/* ----------------------------------------------------------------------
 *
 *                *** Optimal Transportation Meshfree ***
 *
 * This file is part of the USER-OTM package for LAMMPS.
 * Copyright (2020) Lucas Caparini, lucas.caparini@alumni.ubc.ca
 * Department of Mechanical Engineering, University of British Columbia,
 * British Columbia, Canada
 * 
 * Purpose: This fix will move the material points after the nodes have 
 * been repositioned
 * ----------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
// Not all of these will be required!!!
#include "fix_otm_integrate_mp.h"
#include <cstring>
#include <cstdlib>
#include <mpi.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "pair.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

/*
TO-DO:
--> Adjust functions to take multiple atom types for both nodes
    and material points. Important for multiphase/FSI/contact 
    systems
--> Make a separate math file to deal with simple vector/matrix
    operations
*/



/* ----------------------------------------------------------------------
  Initialize pointers and parse the inputs. The args are listed here:
  [0]: FixID
  [1]: groupID - Must include both mp and node style particles
  [2]: Fix name
  [3,4]: keyword (mat_point) and material point style number
  [5,6]: keyword (nodes) and node style number

  More arguments may be needed. We will see
------------------------------------------------------------------------- */

FixOTMIntegrateMP::FixOTMIntegrateMP(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
// fix 1 all otm/integrate_mp mat_points 1 nodes 2
//    [0][1]        [2]          [3]    [4] [5] [6]

int index, ii, i, j;
int ntypes = atom->ntypes;
char *atom_style = atom->atom_style;

if (strcmp(atom_style, "otm") != 0) 
  error->all(FLERR, "Illegal atom_style for material point integration. Use OTM style");
if (narg != 7) 
  error->all(FLERR,"Illegal fix otm/integrate_mp command. Incorrect # of args"); // StC
if (atom->map_style == 0)
  error->all(FLERR, "LME Shape functions require an atom map to evaluate, see atom modify");
if (force->newton_pair) 
  error->all(FLERR, "OTM style cannot be run with newton on"); // Wouldn't make any sense

// Parse the input arguments
for (index = 3; index < narg; index +=2) {
  if (strcmp(arg[index],"mat_points") == 0) {
    // Find atom type of mps
    typeMP = force->numeric(FLERR,arg[index+1]);
    if (typeMP > ntypes) error->all(FLERR,"mat_points type does not exist");
  }
  else if (strcmp(arg[index],"nodes") == 0) {
    // Find atom type of nodes
    typeND = force->numeric(FLERR,arg[index+1]);
    if (typeND > ntypes) error->all(FLERR,"nodes type does not exist");
  }
  else {
    error->all(FLERR,"Unknown keyword identifier for fix otm/integrate_mp");
  }
}

  nevery = 1; // Operation performed every iteration
  time_integrate = 1;

  atom->add_callback(0); // 0 for grow, 1 for restart, 2 for border comm (adds fix to a list of fixes to perform, I think...)

  // Grow and initialize deformation measures
  grow_arrays(atom->nmax);

  double **F = atom->def_grad;
  double **Fdot = atom->def_rate;

  int dim = domain->dimension;
  int nmax = atom->nmax;
  for (ii = 0; ii < nmax; ii++) {
    for (i = 0; i < dim; i++) {
      for (j = 0; j < dim; j++) {
        Fdot[ii][dim*i+j] = 0.0;
        F[ii][dim*i+j] = 0.0;
        if (dim == 2) F[ii][0] = F[ii][3] = 1.0;
        else if (dim == 3) F[ii][0] = F[ii][4] = F[ii][8] = 1.0;
        else error->all(FLERR,"Dimension must be set to 2 or 4. Unrecognized dimension");       
      }
    }
  }

  //DEBUG
  // double **x = atom->x;
  // double **v = atom->v;
  // printf("\n\nx = (%e %e %e)\n",x[216][0],x[216][1],x[216][2]);
  // printf("v = (%e %e %e)\n",v[216][0],v[216][1],v[216][2]);
  // printf("Volume = %e\nMass = %e\n",atom->vfrac[216],atom->rmass[216]);
  // printf("F = |%e %e %e|\n"
  //        "    |%e %e %e|\n"
  //        "    |%e %e %e|\n",F[216][0],F[216][1],F[216][2],F[216][3],F[216][4],F[216][5],F[216][6],F[216][7],F[216][8]);
  //   printf("Fdot = |%e %e %e|\n"
  //          "       |%e %e %e|\n"
  //          "       |%e %e %e|\n\n",Fdot[216][0],Fdot[216][1],Fdot[216][2],Fdot[216][3],Fdot[216][4],Fdot[216][5],Fdot[216][6],Fdot[216][7],Fdot[216][8]);

}

/* ---------------------------------------------------------------------- */

FixOTMIntegrateMP::~FixOTMIntegrateMP()
{
  if (copymode) return;
  
  // memory->destroy(F);
  // memory->destroy(Fdot);
  return;
}

/* ----------------------------------------------------------------------
  Material Points are moved after the nodes have been moved in the 
  initial integration phase. I'm indecisive if I should put this as
  an initial_integrate or post_integrate mask b/c I'm not sure exactly 
  how the order of events happens w/in a stage.
------------------------------------------------------------------------- */

int FixOTMIntegrateMP::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE; 
  return mask;
}

/* ----------------------------------------------------------------------
  Set some flags
------------------------------------------------------------------------- */

void FixOTMIntegrateMP::init()
{
  if (force->pair == NULL)
    error->all(FLERR,"Fix otm/integrate_mp requires a pair style be defined");

  if (atom->tag_enable == 0) {
    error->all(FLERR, "Pair style otm requires atoms have IDs");
  }
  
  // anything else I can think of

  // Need an occasional full neighbor list --> investigate closer
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->pair = 0;  
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->half = 0; // Why is half the default setting?
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 0;
}

/* ---------------------------------------------------------------------- */

void FixOTMIntegrateMP::init_list(int id, NeighList *ptr) 
{
  list = ptr;
}

/* ----------------------------------------------------------------------
  Move the Material Points based on the simple relationship with nodal 
  positions:
    x[mp] = sum{ p_a(x[mp]) * x[nd] }
  The velocity is interpolated with first order backwards difference
  Change the other MP properties through their respective formulas
------------------------------------------------------------------------- */

void FixOTMIntegrateMP::post_integrate(void)
{
  //DEBUG
  // printf("MP Post_integrate %lli",update->ntimestep);

  int i, j, ii, jj, d1, d2, inum, jnum;
  int *ilist, *jlist;
  int itype, jtype;

  int dim = domain->dimension;
  double dt = update->dt;
  double Fincr[3][3], Fold[3][3], Fnew[3][3];

  double x0[3];
  double **x = atom->x;
  double **v = atom->v;
  double **F = atom->def_grad;
  double **Fdot = atom->def_rate;
  double *volume = atom->vfrac;
  int *mask = atom->mask;
  int *type = atom->type;

  double **p = atom->p;
  double **gradp = atom->gradp;
  int *npartner = atom->npartner;
  int **partner = atom->partner;

  inum = list->inum; // # of atoms neighbours are stored for
  ilist = list->ilist; // local indices of I atom


  // Main loop: performs both x and v updates
  for (ii = 0; ii < inum; ii++) { // for every atom w/neighbours
    i = ilist[ii]; // atom index
    itype = type[i];

    if (mask[i] & groupbit && itype == typeMP) { // If atom is in the group and mp type
      jlist = partner[i];
      jnum = npartner[i];
      
      // zero sums
      for (d1 = 0; d1 < dim; d1++) {
        x0[d1] = x[i][d1];
        x[i][d1] = 0.0; // mp positions
        for (d2 = 0; d2 < dim; d2++) Fincr[d1][d2] = 0.0;
      }

      for (jj = 0; jj < jnum; jj++) { // for each neighbour node
        j = jlist[jj]; // neighbour index (local)
        j &= NEIGHMASK; 
        if (type[j] != typeND) error->all(FLERR, "Partner list contains a particle which is not a node!\n");

        for (d1 = 0; d1 < dim; d1++) {
          x[i][d1] += p[i][jj]*x[j][d1]; // position update
          for (d2 = 0; d2 < dim; d2++){
            Fincr[d1][d2] += x[j][d1]*gradp[i][dim*jj+d2]; // incr. def. grad.
          }
        }
      }
      
      // update properties (F, Fdot, volume, velocity)
      double detFincr = determinant(Fincr,dim);
      volume[i] *= detFincr;


      // DEBUG: keeps fucking up on compile
      //vec_to_matrix(F[i],Fold,dim);
      //matrix_mult(Fincr,Fold,Fnew,dim);
      //matrix_to_vec(F[i],Fnew,dim);

      if (dim == 2) {
        Fold[0][0] = F[i][0];
        Fold[0][1] = F[i][1];
        Fold[1][0] = F[i][2];
        Fold[1][1] = F[i][3];

        Fnew[0][0] = Fincr[0][0]*Fold[0][0] + Fincr[0][1]*Fold[1][0];
        Fnew[0][1] = Fincr[0][0]*Fold[0][1] + Fincr[0][1]*Fold[1][1];
        Fnew[1][0] = Fincr[1][0]*Fold[0][0] + Fincr[1][1]*Fold[1][0];
        Fnew[1][1] = Fincr[1][0]*Fold[0][1] + Fincr[1][1]*Fold[1][1];

        F[i][0] = Fnew[0][0];
        F[i][1] = Fnew[0][1];
        F[i][2] = Fnew[1][0];
        F[i][3] = Fnew[1][1];

      }
      else if (dim == 3) {
        Fold[0][0] = F[i][0];
        Fold[0][1] = F[i][1];
        Fold[0][2] = F[i][2];
        Fold[1][0] = F[i][3];
        Fold[1][1] = F[i][4];
        Fold[1][2] = F[i][5];
        Fold[2][0] = F[i][6];
        Fold[2][1] = F[i][7];
        Fold[2][2] = F[i][8];

        Fnew[0][0] = Fincr[0][0]*Fold[0][0] + Fincr[0][1]*Fold[1][0] + Fincr[0][2]*Fold[2][0];
        Fnew[0][1] = Fincr[0][0]*Fold[0][1] + Fincr[0][1]*Fold[1][1] + Fincr[0][2]*Fold[2][1];
        Fnew[0][2] = Fincr[0][0]*Fold[0][2] + Fincr[0][1]*Fold[1][2] + Fincr[0][2]*Fold[2][2];
        Fnew[1][0] = Fincr[1][0]*Fold[0][0] + Fincr[1][1]*Fold[1][0] + Fincr[1][2]*Fold[2][0];
        Fnew[1][1] = Fincr[1][0]*Fold[0][1] + Fincr[1][1]*Fold[1][1] + Fincr[1][2]*Fold[2][1];
        Fnew[1][2] = Fincr[1][0]*Fold[0][2] + Fincr[1][1]*Fold[1][2] + Fincr[1][2]*Fold[2][2];
        Fnew[2][0] = Fincr[2][0]*Fold[0][0] + Fincr[2][1]*Fold[1][0] + Fincr[2][2]*Fold[2][0];
        Fnew[2][1] = Fincr[2][0]*Fold[0][1] + Fincr[2][1]*Fold[1][1] + Fincr[2][2]*Fold[2][1];
        Fnew[2][2] = Fincr[2][0]*Fold[0][2] + Fincr[2][1]*Fold[1][2] + Fincr[2][2]*Fold[2][2];

        F[i][0] = Fnew[0][0];
        F[i][1] = Fnew[0][1];
        F[i][2] = Fnew[0][2];
        F[i][3] = Fnew[1][0];
        F[i][4] = Fnew[1][1];
        F[i][5] = Fnew[1][2];
        F[i][6] = Fnew[2][0];
        F[i][7] = Fnew[2][1];
        F[i][8] = Fnew[2][2];
      }

      for (d1 = 0; d1 < dim; d1++) {
        v[i][d1] = (x[i][d1] - x0[d1])/dt; // Backwards 1st order velocity
        for (d2 = 0; d2 < dim; d2++) {
          Fdot[i][d1*dim+d2] = (Fnew[d1][d2] - Fold[d1][d2])/dt;
        }
      }

      //DEBUG
      // if (dim==2) {
      // printf("________________________\n"
      //        "Timestep = %lli\tMP = %i\n",update->ntimestep,i);
      // printf("x = (%e %e %e)\n",x[216][0],x[216][1],x[216][2]);
      // printf("v = (%e %e %e)\n",v[216][0],v[216][1],v[216][2]);
      // printf("Volume = %e\nMass = %e\n",atom->vfrac[216],atom->rmass[216]);
      // printf("Fincr = |%e %e|\n"
      //        "        |%e %e|\n",Fincr[0][0],Fincr[0][1],Fincr[1][0],Fincr[1][1]);
      // printf("F = |%e %e|\n"
      //        "    |%e %e|\n",F[216][0],F[216][1],F[216][2],F[216][3]);
      // printf("Fdot = |%e %e|\n"
      //        "       |%e %e|\n\n",Fdot[216][0],Fdot[216][1],Fdot[216][2],Fdot[216][3]);
      // }
      // else if (dim==3) {
      // printf("________________________\n"
      //        "Timestep = %lli\tMP = %i\n",update->ntimestep,i);
      // printf("x = (%e %e %e)\n",x[216][0],x[216][1],x[216][2]);
      // printf("v = (%e %e %e)\n",v[216][0],v[216][1],v[216][2]);
      // printf("Volume = %e\nMass = %e\n",atom->vfrac[216],atom->rmass[216]);
      // printf("Fincr = |%e %e %e|\n"
      //        "        |%e %e %e|\n"
      //        "        |%e %e %e|\n",Fincr[0][0],Fincr[0][1],Fincr[0][2],Fincr[1][0],Fincr[1][1],Fincr[1][2],Fincr[2][0],Fincr[2][1],Fincr[2][2]);
      // printf("F = |%e %e %e|\n"
      //        "    |%e %e %e|\n"
      //        "    |%e %e %e|\n",F[216][0],F[216][1],F[216][2],F[216][3],F[216][4],F[216][5],F[216][6],F[216][7],F[216][8]);
      // printf("Fdot = |%e %e %e|\n"
      //        "       |%e %e %e|\n"
      //        "       |%e %e %e|\n\n",Fdot[216][0],Fdot[216][1],Fdot[216][2],Fdot[216][3],Fdot[216][4],Fdot[216][5],Fdot[216][6],Fdot[216][7],Fdot[216][8]);
      // }

    }
  }
  
  return;  
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */
// Change to ignore the nodes
void FixOTMIntegrateMP::grow_arrays(int nmax) 
{
  int dim = domain->dimension;
  memory->grow(atom->def_grad, nmax, dim*dim, "otm_mp_update:F"); // Final arg is just for an error message
  memory->grow(atom->def_rate, nmax, dim*dim, "otm_mp_update:Fdot");
  return;
}

/* ----------------------------------------------------------------------
  Compute the determinant.
  Takes a 3x3 matrix, but will compute the determinant as either a 2x2 or
  3x3 depending on the value of dim.
------------------------------------------------------------------------- */

double FixOTMIntegrateMP::determinant(double F[3][3], int dim) 
{
  double det;

  if (dim == 2) {
    det = F[0][0]*F[1][1] - F[0][1]*F[1][0];
  }
  else if (dim == 3) {
    det = F[0][0] * ( F[2][2]*F[1][1] - F[2][1]*F[1][2] )
        - F[1][0] * ( F[2][2]*F[0][1] - F[2][1]*F[0][2] )
        + F[2][0] * ( F[1][2]*F[0][1] - F[1][1]*F[0][2] );
  }

  return det;
}

/* ----------------------------------------------------------------------
  Compute the matrix inner product A*B = C.
  Takes a 3x3 matrix, but will compute the multiplication as either a 2x2
  or 3x3 depending on the value of dim.

  Should overload later for use with double ** matrices
------------------------------------------------------------------------- */
// void matrix_mult(double A[3][3], double B[3][3], double C[3][3], int dim)
// {
//   if (dim == 2) {
//     C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0];
//     C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1];
//     C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0];
//     C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1];
//   }
//   else if (dim == 3) {
//     C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0];
//     C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1];
//     C[0][2] = A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2];
//     C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0];
//     C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1];
//     C[1][2] = A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2];
//     C[2][0] = A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0];
//     C[2][1] = A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1];
//     C[2][2] = A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2];
//   }

//   return;
// }

// /* ----------------------------------------------------------------------
//   Takes a vec of length dim^2 in the form of a pointer and returns a 
//   matrix populated by its values

//   Overload this later
// ------------------------------------------------------------------------- */
// void vec_to_matrix(double *a, double B[3][3], int dim)
// {
//   if (dim == 2) {
//     B[0][0] = a[0];
//     B[0][1] = a[1];
//     B[1][0] = a[2];
//     B[1][1] = a[3];
//   }
//   else if (dim == 3) {
//     B[0][0] = a[0];
//     B[0][1] = a[1];
//     B[0][2] = a[2];
//     B[1][0] = a[3];
//     B[1][1] = a[4];
//     B[1][2] = a[5];
//     B[2][0] = a[6];
//     B[2][1] = a[7];
//     B[2][2] = a[8];
//   }

//   return;
// }

// /* ----------------------------------------------------------------------
//   Takes a dim x dim matrix and writes the values to a vec of length dim^2 

//   Overload this later
// ------------------------------------------------------------------------- */
// void matrix_to_vec(double *a, double B[3][3], int dim)
// {
//   if (dim == 2) {
//     a[0] = B[0][0];
//     a[1] = B[0][1];
//     a[2] = B[1][0];
//     a[3] = B[1][1];
//   }
//   else if (dim == 3) {
//     a[0] = B[0][0];
//     a[1] = B[0][1];
//     a[2] = B[0][2];
//     a[3] = B[1][0];
//     a[4] = B[1][1];
//     a[5] = B[1][2];
//     a[6] = B[2][0];
//     a[7] = B[2][1];
//     a[8] = B[2][2];
//   }

//   return;
// }
