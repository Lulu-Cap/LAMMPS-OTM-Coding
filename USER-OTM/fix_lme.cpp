/* ----------------------------------------------------------------------
 *
 *                *** Optimal Transportation Meshfree ***
 *
 * This file is part of the USER-OTM package for LAMMPS.
 * Copyright (2020) Lucas Caparini, lucas.caparini@alumni.ubc.ca
 * Department of Mechanical Engineering, University of British Columbia,
 * British Columbia, Canada
 * 
 * Purpose: This fix will update the LME shape function evaluations after
 * the movement of the nodes and material points has been completed for a 
 * timestep.
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
#include "fix_lme.h"
#include <cstring>
#include <cstdlib>
#include <stdio.h> //DEBUG
#include <cmath>
#include <mpi.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "lattice.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "pair.h"
#include "region.h"
#include "update.h"
#include "variable.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;
#define DELTA 16384 // what?
#define TOL 1.0e-16 // machine precision used for cutoff radii
#define LMDA_TOL_SQ 1.0e-8*1.0e-8
#define NEIGH_MIN 5
#define NEIGH_MAX 100
#define MAX(A,B) ((A) > (B) ? (A) : (B))

/*
TO-DO:
--> Adjust functions to take multiple atom types for both nodes
    and material points. Important for multiphase/FSI/contact 
    systems
--> Add a safety to the LME neighbour finding. If numneigh < NEIGH_MIN 
    b/c nodes are too far, take the closest NEIGH_MIN nodes as neighbours. 
    Actually, this should be refined further. Take re-normalize the ball 
    radius based on the value of the closest node.
    i.e. If the closest yields exp(beta/|r|)=1e-20, then the new criterion
    could be set as TOL2 = 1e-20*TOL. This way we still have the same computer 
    precision cutoffs maintained.
*/



/* ----------------------------------------------------------------------
  Initialize pointers and parse the inputs. The args are listed here:
  [0]: FixID
  [1]: groupID - Must include both mp and node style particles
  [2]: Fix name
  [3,4]: keyword (mat_point) and material point style number
  [5,6]: keyword (nodes) and node style number
  [7,8]: keyword (Spacing) and nodal spacing parameter
  [9,10]: keyword (Locality) and locality parameter (gamma)

  More arguments may be needed. We will see
------------------------------------------------------------------------- */

FixLME::FixLME(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
// fix 1 all otm/lme/shape mat_points 1 nodes 2 Spacing ${h} Locality ${gamma}
//    [0][1]      [2]          [3]   [4] [5] [6]  [7]   [8]    [9]      [10]

int index;
int ntypes = atom->ntypes;
char *atom_style = atom->atom_style;

if (strcmp(atom_style, "otm") != 0) {
  error->all(FLERR, "Illegal atom_style for LME Shape function evaluations");
}
if (narg != 11) {
  error->all(FLERR,"Illegal fix otm/lme/shape command"); // Must have at least 10 args 
                 //(may include option for more later, so multiple groups can easily interact)
}
if (atom->map_style == 0){
  error->all(FLERR, "LME Shape functions require an atom map to evaluate, see atom modify");
}
if (force->newton_pair) {
  error->all(FLERR, "OTM style cannot be run with newton on"); // Wouldn't make any sense
}

// Parse the input arguments
for (index = 3; index < narg; index +=2) {
  if (strcmp(arg[index],"mat_points") == 0) {
    // Find atom type of mps
    typeMP = force->numeric(FLERR,arg[index+1]);
    if (typeMP > ntypes) error->all(FLERR,"mat_points type does not exist");
  }
  else if (strcmp(arg[index],"nodes") == 0) {
    // Find atom type of nodes
    typeND = force->numeric(FLERR, arg[index+1]);
    if (typeND > ntypes) error->all(FLERR,"nodes type does not exist");
  }
  else if (strcmp(arg[index],"Spacing") == 0) {
    h = force->numeric(FLERR,arg[index+1]);
  }
  else if (strcmp(arg[index],"Locality") == 0) {
    gamma = force->numeric(FLERR,arg[index+1]);
  }
  else {
    error->all(FLERR,"Unknown keyword identifier for fix otm/lme/shape");
  }
}

  nevery = 1; // Operation performed every iteration
  
  maxpartner = 1; // Initialize one partner

  atom->npartner = NULL; // Vector containing # partners of each atom
  atom->partner = NULL; // Array containing global atom IDs of partners
  atom->p = NULL; // Array of derivatives of shape functions at each partner
  atom->gradp = NULL; // Array of shape functions at each partner

  grow_arrays(atom->nmax); // grow arrays to the corresponding number of atoms
                           // nmax = max # of owned +ghost in arrays on this proc)
  atom->add_callback(0); // 0 for grow, 1 for restart, 2 for border comm (adds fix to a list of fixes to perform, I think...)

  // initialize npartner to 0 so neighbour list creation is OK the 1st time
  int nlocal = atom->nlocal;
  for (int ii = 0; ii < nlocal; ii++) {
    atom->npartner[ii] = 0;
  }

  nmax = 0;
  comm_forward = ntypes; // size from own atoms to ghost atoms

}

/* ---------------------------------------------------------------------- */
// Testing --> may not require much deletion if atom storage used
FixLME::~FixLME()
{
  if (copymode) return; // What is this?
}

/* ----------------------------------------------------------------------
  Shape functions must be evaluated prior to force evaluations 
  (pre_force and setup)
------------------------------------------------------------------------- */

int FixLME::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE; 
  return mask;
}

/* ----------------------------------------------------------------------
  Set some flags
------------------------------------------------------------------------- */
// Investigate closer (when to use full or half neighbor lists)
void FixLME::init()
{
  if (force->pair == NULL)
    error->all(FLERR,"Fix otm/lme/shape requires a pair style be defined");

  if (atom->tag_enable == 0) {
    error->all(FLERR, "Pair style otm requires atoms have IDs");
  }
  
  // Check variables for errors
  if (gamma < 0.01 || gamma > 10.0) {
    error->all(FLERR, "locality parameter (gamma) outside of normal parameters.\n"
                      "Please use a value in the range [0.01,10].\n"
                      "For best results, stay in the range [0.8,4]\n");
  }
  if (h <= 0) {
    error->all(FLERR, "Nodal Spacing parameter (h) cannot be zero or negative.\n"
      "Please adjust value.\n");
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

void FixLME::init_list(int id, NeighList *ptr) 
{
  list = ptr;
}

/* ----------------------------------------------------------------------
  Compute shape functions and shape function gradients before the first
  integration - essentially identical to pre_force()
------------------------------------------------------------------------- */

void FixLME::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    pre_force(vflag);
  else
    error->all(FLERR,"Fix setchempotential requires verlet method");

  int i, j, ii, jj, n, inum, jnum;
  int *ilist, *jlist, *numneigh, **firstneigh;
  int itype, jtype;

  int nlocal = atom->nlocal; // number of owned atoms on this proc
  nmax = atom->nmax; // Maximum number of local + ghost atoms on this proc
  grow_arrays(nmax);

  int *npartner = atom->npartner;
  int **partner = atom->partner;
  double **p = atom->p;
  double **gradp = atom->gradp;

  double **x = atom->x;
  double *radius = atom->radius; // Do I need this?
  int *mask = atom->mask; 
  int *type = atom->type;
  tagint *tag = atom->tag; // wtf?
  //NeighList *list = pair->list; //
  inum = list->inum; // # of atoms neighbours are stored for
  ilist = list->ilist; // local indices of I atoms
  numneigh = list->numneigh; // # of J neighbours for each I atom
  firstneigh = list->firstneigh; // ptr to 1st J int value of each I atom

  // Cutoff Radius
  double beta = gamma / (h*h); 
  double Rcut_sq = -log(TOL)/beta; 
  int dim = domain->dimension;

  // zero npartner for all current atoms
  for (i = 0; i < nlocal; i++)
    npartner[i] = 0;

  // Find the max # of partners --> very important to do before allocating memory!
  maxpartner = NEIGH_MIN;
  for (ii = 0; ii < nlocal; ii++) {
    int count = 0;
    i = ilist[ii];
    itype = type[i];
    if ( (mask[i] & groupbit) && (itype == typeMP) ) {
      jlist = firstneigh[i];
      jnum = numneigh[i];
      for (jj = 0; jj < jnum; jj++) {
        if ( (mask[j] & groupbit) && (jtype == typeND) ) {
          double rsq = 0.0; // Euclidean distance
          for (int d = 0; d < dim; d++)
            rsq += (x[i][d] - x[j][d]) * (x[i][d] - x[j][d]);
          if (rsq <= Rcut_sq) count++;
        }
      }
    }
    maxpartner = MAX(maxpartner, count);
  }
  int maxall;
  MPI_Allreduce(&maxpartner, &maxall, 1, MPI_INT, MPI_MAX, world);
  maxpartner = maxall;
  
  grow_arrays(nmax);
  
  hMin = 1.0e10; // random large number
  // Create a partner list from material points to nodes ONLY
  for (ii = 0; ii < inum; ii++) {//iloop
    i = ilist[ii];
    itype = type[i];

    if (mask[i] & groupbit) { //itest1
      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {//jloop
        j = jlist[jj];
        j &= NEIGHMASK;
        jtype = type[j];

        if ( (mask[j] & groupbit) && (jtype == typeND) ) {//jtest
          double rsq = 0.0; // Euclidean distance
          for (int d = 0; d < dim; d++)
            rsq += (x[i][d] - x[j][d]) * (x[i][d] - x[j][d]);

          if ( (itype == typeMP) && (rsq <= Rcut_sq) ) {
            partner[i][npartner[i]++] = j;
          }

          else if (itype == typeND) { //node
            if (rsq < hMin*hMin && rsq > 0.0) 
              hMin = pow(rsq,0.5); // assign hMin
            if (jj == 1) {
              npartner[i] = -1;
              partner[i] = NULL;
            }
          } //node
        }//jtest
      }//jloop

      {
        // If insufficient neighbours based on TOL criterion, get the 
        // closest ones
        /*Note: Improve this algorithm by renormalizing the TOL instead!
        See notes at top or in logbook for details*/
        if ( (npartner[i] < NEIGH_MIN) && (itype == typeMP) ) {
          npartner[i] = NEIGH_MIN;
          int rsq_closest[NEIGH_MIN];
          int nodal_neigh = 0;

          for (jj = 0; jj < jnum; jj++) { //jloop
            j = jlist[jj];
            j &= NEIGHMASK;
            jtype = type[j];

            if ( (mask[j] & groupbit) && (jtype == typeND) ) { //jtest
              double rsq = 0.0;
              for (int d = 0; d < dim; d++)
                rsq += (x[i][d] - x[j][d]) * (x[i][d] - x[j][d]);
              
              if (nodal_neigh < NEIGH_MIN) { // Assign first values
                partner[i][nodal_neigh] = j;
                rsq_closest[nodal_neigh] = rsq;
                nodal_neigh++;
              }

              for (int kk = 0; kk < NEIGH_MIN; kk++) { // Check for closer nds
                if (rsq < rsq_closest[kk]) {
                  partner[i][kk] = j;
                  rsq_closest[kk] = rsq;
                }

              }

            } //jtest
          } //jloop
          if (nodal_neigh < NEIGH_MIN)
            error->all(FLERR,"Insufficient nodal neighbours found for material point");
        }
      }
    }//itest1
  }//iloop

  // zero shape functions + derivatives
  for (i = 0; i < nlocal; i++) {
    for (jj = 0; jj < maxpartner; jj++) {
      p[i][jj] = 0.0;
      for (int d = 0; d < dim; d++) gradp[i][dim*jj+d] = 0.0;
    }
  }

  // Main Loop: loop though each nd/mp pair and evaluate shape function and 
  //    shape function gradient for each.  
  const int max_iter = 100;
  for (ii = 0; ii < inum; ii++) { // mp loop
    i = ilist[ii]; // mp id
    itype = type[i];

    if (mask[i] & groupbit && itype == typeMP) { // mp test
      jnum = npartner[i]; // number of nodal neighbours
      jlist = partner[i]; // indices of nodal neighbours

      // initial values for optimization problem
      double lambda0[3] = {0,0,0}; // Lagrange multipliers
      double lambda1[3] = {1,1,1}; 
      double r[3] = {0,0,0}; // gradient of shape functions w.r.t lambda
      double H[3][3] = {{0,0,0},{0,0,0},{0,0,0}};// Hessian of shape func. w.r.t lambda
      double invH[3][3] = {{1,0,0},{0,1,0},{0,0,1}}; // Inverse Hessian
      double det, norm_sq;
      int iter = 0;
      

      // LME loop (Regularized Newton's Method)
      do {
        iter++;
        lambda0[0] = lambda1[0];
        lambda0[1] = lambda1[1];
        lambda0[2] = lambda1[2];

        // Guess at shape functions
        double Z = 0.0; // sum of partition functions
        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj]; // nodal number
          
          double dx[3] = { (x[i][0]-x[j][0]),
                           (x[i][1]-x[j][1]),
                           (x[i][2]-x[j][2])};
          double f = -beta * ( dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2] ) +
                       1/h * ( lambda1[0]*dx[0] + lambda1[1]*dx[1] + lambda1[2]*dx[2] );
          p[i][jj] = exp(f);
          Z += p[i][jj];
        }

        // Evaluate Gradient and Hessian (regularized)
        { // Rezero gradient and Hessian
        r[0]=r[1]=r[2]=0.0;
        H[0][0]=H[0][1]=H[0][2]=0.0;
        H[1][0]=H[1][1]=H[1][2]=0.0;
        H[2][0]=H[2][1]=H[2][2]=0.0;}
        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          p[i][jj] /= Z; // Normalized shape function value
          double dx[3] = { (x[i][0]-x[j][0]),
                           (x[i][1]-x[j][1]),
                           (x[i][2]-x[j][2])};
          // Gradient vector
          r[0] += p[i][jj] * dx[0];
          r[1] += p[i][jj] * dx[1];
          r[2] += p[i][jj] * dx[2];

          // Hessian Matrix (summation components)
          H[0][0] += p[i][jj]*dx[0]*dx[0];
          H[0][1] += p[i][jj]*dx[0]*dx[1];
          H[0][2] += p[i][jj]*dx[0]*dx[2];
          H[1][0] += p[i][jj]*dx[1]*dx[0];
          H[1][1] += p[i][jj]*dx[1]*dx[1];
          H[1][2] += p[i][jj]*dx[1]*dx[2];
          H[2][0] += p[i][jj]*dx[2]*dx[0];
          H[2][1] += p[i][jj]*dx[2]*dx[1];
          H[2][2] += p[i][jj]*dx[2]*dx[2];
          
        }

        r[0] /= h; r[1] /= h; r[2] /= h; 
        double norm_r = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
        norm_r = sqrt(norm_r); // 2-norm of gradient
        H[0][0] = H[0][0]/(h*h) - r[0]*r[0] + norm_r; 
        H[0][1] = H[0][1]/(h*h) - r[0]*r[1]; 
        H[0][2] = H[0][2]/(h*h) - r[0]*r[2];
        H[1][0] = H[1][0]/(h*h) - r[1]*r[0];
        H[1][1] = H[1][1]/(h*h) - r[1]*r[1] + norm_r;
        H[1][2] = H[1][2]/(h*h) - r[1]*r[2];
        H[2][0] = H[2][0]/(h*h) - r[2]*r[0];
        H[2][1] = H[2][1]/(h*h) - r[2]*r[1]; 
        H[2][2] = H[2][2]/(h*h) - r[2]*r[2] + norm_r;

        // Invert Hessian Matrix
        if (dim == 2) { // Invert 2D explicitly
        det = H[0][0]*H[1][1] - H[0][1]*H[1][0];
        invH[0][0] = H[1][1]/det;
        invH[0][1] = -H[0][1]/det;
        invH[1][0] = -H[1][0]/det;
        invH[1][1] = H[0][0]/det;

        invH[0][2] = invH[1][2] = 0.0; // identity for z-component
        invH[2][0] = invH[2][1] = 0.0;
        invH[2][2] = 1.0;
        }
        else if (dim == 3) { // Invert 3D explicitly
        det = H[0][0] * ( H[2][2]*H[1][1] - H[2][1]*H[1][2] ) - 
              H[1][0] * ( H[2][2]*H[0][1] - H[2][1]*H[0][2] ) + 
              H[2][0] * ( H[1][2]*H[0][1] - H[1][1]*H[0][2] );

        invH[0][0] = +( H[2][2]*H[1][1] - H[2][1]*H[1][2] ) / det;
        invH[0][1] = -( H[2][2]*H[0][1] - H[2][1]*H[0][2] ) / det;
        invH[0][2] = +( H[1][2]*H[0][1] - H[1][1]*H[0][2] ) / det;

        invH[1][0] = -( H[2][2]*H[1][0] - H[2][0]*H[1][2] ) / det;
        invH[1][1] = +( H[2][2]*H[0][0] - H[2][0]*H[0][2] ) / det;
        invH[1][2] = -( H[1][2]*H[0][0] - H[1][0]*H[0][2] ) / det;

        invH[2][0] = +( H[2][1]*H[1][0] - H[2][0]*H[1][1] ) / det;
        invH[2][1] = -( H[2][1]*H[0][0] - H[2][0]*H[0][1] ) / det;
        invH[2][2] = +( H[1][1]*H[0][0] - H[1][0]*H[0][1] ) / det;

        }
        
        // Increment lambda1: lambda1 = lambda0 - invH*r 
        {
        lambda1[0] = lambda0[0] - ( invH[0][0]*r[0] + invH[0][1]*r[1] + invH[0][2]*r[2] );
        lambda1[1] = lambda0[1] - ( invH[1][0]*r[0] + invH[1][1]*r[1] + invH[1][2]*r[2] );
        lambda1[2] = lambda0[2] - ( invH[2][0]*r[0] + invH[2][1]*r[1] + invH[2][2]*r[2] );
        }

        // Test if max_iter exceeded or convergence otherwise failed
        {
        if (iter > max_iter) error->all(FLERR, "Maximum iterations reached without LME convergence to specified tolerance\n");
        if ((isnormal(lambda1[0]) == 0 && lambda1[0] != 0) || 
            (isnormal(lambda1[1]) == 0 && lambda1[1] != 0) ||
            (isnormal(lambda1[2]) == 0 && lambda1[2] != 0)) {
          error->all(FLERR, "Lagrange multipliers reached undefined value (NaN). LME failed to converge\n");
        }
        }

        // Evaluate change in lambda
        {norm_sq = (lambda0[0]-lambda1[0])*(lambda0[0]-lambda1[0]) +
                  (lambda0[1]-lambda1[1])*(lambda0[1]-lambda1[1]) +
                  (lambda0[2]-lambda1[2])*(lambda0[2]-lambda1[2]);
        }

      } while (norm_sq > LMDA_TOL_SQ); // Convergence while loop
      
      // Spatial Gradient of shape functions
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        double dx[3] = { (x[i][0]-x[j][0]),
                         (x[i][1]-x[j][1]),
                         (x[i][2]-x[j][2]) };
        if (dim == 2) {
          gradp[i][dim*jj] += -p[i][jj]/(h*h) * ( invH[0][0]*dx[0] + invH[0][1]*dx[1] ); // Add p[i][jj]*K_a*grad(beta) term if beta becomes nonconstant
          gradp[i][dim*jj+1] += -p[i][jj]/(h*h) * ( invH[1][0]*dx[0] + invH[1][1]*dx[1] );
        }
        else if (dim == 3) {
          gradp[i][dim*jj] += -p[i][jj]/(h*h) * ( invH[0][0]*dx[0] + invH[0][1]*dx[1] + invH[0][2]*dx[2] ); // Add p[i][jj]*K_a*grad(beta) term if beta becomes nonconstant
          gradp[i][dim*jj+1] += -p[i][jj]/(h*h) * ( invH[1][0]*dx[0] + invH[1][1]*dx[1] + invH[1][2]*dx[2] );
          gradp[i][dim*jj+2] += -p[i][jj]/(h*h) * ( invH[2][0]*dx[0] + invH[2][1]*dx[1] + invH[2][2]*dx[2] );
        }
      }
    
    } // mp test
  } // mp loop

  // Adjust the below statistics --> print shape function statistics to the terminal
  
  // bond statistics
  if (update->ntimestep > -1) {
    n = 0;
    int count = 0;
    for (i = 0; i < nlocal; i++) {
      itype = type[i];
      if (mask[i] & groupbit && itype == typeMP) {
        n += npartner[i];
        count += 1;
      }
    }
    int nall, countall;
    MPI_Allreduce(&n, &nall, 1, MPI_INT, MPI_SUM, world);
    MPI_Allreduce(&count, &countall, 1, MPI_INT, MPI_SUM, world);
    if (countall < 1) countall = 1;

    // if (comm->me == 0) {
    //   if (screen) {
    //     printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
    //     fprintf(screen, "OTM neighbors:\n");
    //     fprintf(screen, "  max # of neighbors for a single mp = %d\n", maxpartner);
    //     fprintf(screen, "  average # of neighbors/particle in group tlsph = %g\n", (double) nall / countall);
    //     printf(">>========>>========>>========>>========>>========>>========>>========>>========\n\n");
    //   }
    //   if (logfile) {
    //     fprintf(logfile, "\nOTM neighbors:\n");
    //     fprintf(logfile, "  max # of neighbors for a single particle = %d\n", maxpartner);
    //     fprintf(logfile, "  average # of neighbors/particle in group tlsph = %g\n", (double) nall / countall);
    //   }
    // }
  } 
}

/* ----------------------------------------------------------------------
  Compute shape functions and shape function gradients before any forces
  are computed
------------------------------------------------------------------------- */

void FixLME::pre_force(int vflag)
{
  int i, j, ii, jj, n, inum, jnum;
  int *ilist, *jlist, *numneigh, **firstneigh;
  int itype, jtype;

  int nlocal = atom->nlocal; // number of owned atoms on this proc
  nmax = atom->nmax; // Maximum number of local + ghost atoms on this proc

  int *npartner = atom->npartner;
  int **partner = atom->partner;
  double **p = atom->p;
  double **gradp = atom->gradp;

  double **x = atom->x;
  double *radius = atom->radius; // Do I need this?
  int *mask = atom->mask; 
  int *type = atom->type;
  tagint *tag = atom->tag; // wtf?
  //NeighList *list = pair->list; //
  inum = list->inum; // # of atoms neighbours are stored for
  ilist = list->ilist; // local indices of I atoms
  numneigh = list->numneigh; // # of J neighbours for each I atom
  firstneigh = list->firstneigh; // ptr to 1st J int value of each I atom

  // Cutoff Radius
  double beta = gamma / (h*h); 
  double Rcut_sq = -log(TOL)/beta; 
  int dim = domain->dimension;

  // zero npartner for all current atoms
  for (i = 0; i < nlocal; i++)
    npartner[i] = 0;

  // Find the max # of partners --> very important to do before allocating memory!
  maxpartner = NEIGH_MIN;
  for (ii = 0; ii < nlocal; ii++) {
    int count = 0;
    i = ilist[ii];
    itype = type[i];
    if ( (mask[i] & groupbit) && (itype == typeMP) ) {
      jlist = firstneigh[i];
      jnum = numneigh[i];
      for (jj = 0; jj < jnum; jj++) {
        if ( (mask[j] & groupbit) && (jtype == typeND) ) {
          double rsq = 0.0; // Euclidean distance
          for (int d = 0; d < dim; d++)
            rsq += (x[i][d] - x[j][d]) * (x[i][d] - x[j][d]);
          if (rsq <= Rcut_sq) count++;
        }
      }
    }
    maxpartner = MAX(maxpartner, count);
  }
  int maxall;
  MPI_Allreduce(&maxpartner, &maxall, 1, MPI_INT, MPI_MAX, world);
  maxpartner = maxall;
  
  grow_arrays(nmax);
  
  hMin = 1.0e10; // random large number
  // Create a partner list from material points to nodes ONLY
  for (ii = 0; ii < inum; ii++) {//iloop
    i = ilist[ii];
    itype = type[i];

    if (mask[i] & groupbit) { //itest1
      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {//jloop
        j = jlist[jj];
        j &= NEIGHMASK;
        jtype = type[j];

        if ( (mask[j] & groupbit) && (jtype == typeND) ) {//jtest
          double rsq = 0.0; // Euclidean distance
          for (int d = 0; d < dim; d++)
            rsq += (x[i][d] - x[j][d]) * (x[i][d] - x[j][d]);

          if ( (itype == typeMP) && (rsq <= Rcut_sq) ) {
            partner[i][npartner[i]++] = j;
          }

          else if (itype == typeND) { //node
            if (rsq < hMin*hMin && rsq > 0.0) 
              hMin = pow(rsq,0.5); // assign hMin
            if (jj == 1) {
              npartner[i] = -1;
              partner[i] = NULL;
            }
          } //node
        }//jtest
      }//jloop

      {
        // If insufficient neighbours based on TOL criterion, get the 
        // closest ones
        /*Note: Improve this algorithm by renormalizing the TOL instead!
        See notes at top or in logbook for details*/
        if ( (npartner[i] < NEIGH_MIN) && (itype == typeMP) ) {
          npartner[i] = NEIGH_MIN;
          int rsq_closest[NEIGH_MIN];
          int nodal_neigh = 0;

          for (jj = 0; jj < jnum; jj++) { //jloop
            j = jlist[jj];
            j &= NEIGHMASK;
            jtype = type[j];

            if ( (mask[j] & groupbit) && (jtype == typeND) ) { //jtest
              double rsq = 0.0;
              for (int d = 0; d < dim; d++)
                rsq += (x[i][d] - x[j][d]) * (x[i][d] - x[j][d]);
              
              if (nodal_neigh < NEIGH_MIN) { // Assign first values
                partner[i][nodal_neigh] = j;
                rsq_closest[nodal_neigh] = rsq;
                nodal_neigh++;
              }

              for (int kk = 0; kk < NEIGH_MIN; kk++) { // Check for closer nds
                if (rsq < rsq_closest[kk]) {
                  partner[i][kk] = j;
                  rsq_closest[kk] = rsq;
                }

              }

            } //jtest
          } //jloop
          if (nodal_neigh < NEIGH_MIN)
            error->all(FLERR,"Insufficient nodal neighbours found for material point");
        }
      }
    }//itest1
  }//iloop

  // zero shape functions + derivatives
  for (i = 0; i < nlocal; i++) {
    for (jj = 0; jj < maxpartner; jj++) {
      p[i][jj] = 0.0;
      for (int d = 0; d < dim; d++) gradp[i][dim*jj+d] = 0.0;
    }
  }

  // Main Loop: loop though each nd/mp pair and identify evaluate shape function and 
  //    shape function gradient for each.  
  const int max_iter = 100;
  for (ii = 0; ii < inum; ii++) { // mp loop
    i = ilist[ii]; // mp id
    itype = type[i];

    if (mask[i] & groupbit && itype == typeMP) { // mp test
      jnum = npartner[i]; // number of nodal neighbours
      jlist = partner[i]; // indices of nodal neighbours

      // initial values for optimization problem
      double lambda0[3] = {0,0,0}; // Lagrange multipliers
      double lambda1[3] = {1,1,1}; 
      double r[3] = {0,0,0}; // gradient of shape functions w.r.t lambda
      double H[3][3] = {{0,0,0},{0,0,0},{0,0,0}};// Hessian of shape func. w.r.t lambda
      double invH[3][3] = {{1,0,0},{0,1,0},{0,0,1}}; // Inverse Hessian
      double det, norm_sq;
      int iter = 0;
      

      // LME loop (Regularized Newton's Method)
      do {
        iter++;
        lambda0[0] = lambda1[0];
        lambda0[1] = lambda1[1];
        lambda0[2] = lambda1[2];

        // Guess at shape functions
        double Z = 0.0; // sum of partition functions
        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj]; // nodal number

          double dx[3] = { (x[i][0]-x[j][0]),
                           (x[i][1]-x[j][1]),
                           (x[i][2]-x[j][2])};
          double f = -beta * ( dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2] ) +
                       1/h * ( lambda1[0]*dx[0] + lambda1[1]*dx[1] + lambda1[2]*dx[2] );
          p[i][jj] = exp(f);
          Z += p[i][jj];
        }

        // Evaluate Gradient and Hessian (regularized)
        { // Rezero gradient and Hessian
        r[0]=r[1]=r[2]=0.0;
        H[0][0]=H[0][1]=H[0][2]=0.0;
        H[1][0]=H[1][1]=H[1][2]=0.0;
        H[2][0]=H[2][1]=H[2][2]=0.0;}
        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          p[i][jj] /= Z; // Normalized shape function value
          double dx[3] = { (x[i][0]-x[j][0]),
                           (x[i][1]-x[j][1]),
                           (x[i][2]-x[j][2])};
          // Gradient vector
          r[0] += p[i][jj] * dx[0];
          r[1] += p[i][jj] * dx[1];
          r[2] += p[i][jj] * dx[2];

          // Hessian Matrix (summation components)
          H[0][0] += p[i][jj]*dx[0]*dx[0];
          H[0][1] += p[i][jj]*dx[0]*dx[1];
          H[0][2] += p[i][jj]*dx[0]*dx[2];
          H[1][0] += p[i][jj]*dx[1]*dx[0];
          H[1][1] += p[i][jj]*dx[1]*dx[1];
          H[1][2] += p[i][jj]*dx[1]*dx[2];
          H[2][0] += p[i][jj]*dx[2]*dx[0];
          H[2][1] += p[i][jj]*dx[2]*dx[1];
          H[2][2] += p[i][jj]*dx[2]*dx[2];
          
        }

        r[0] /= h; r[1] /= h; r[2] /= h; 
        double norm_r = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
        norm_r = sqrt(norm_r); // 2-norm of gradient
        H[0][0] = H[0][0]/(h*h) - r[0]*r[0] + norm_r; 
        H[0][1] = H[0][1]/(h*h) - r[0]*r[1]; 
        H[0][2] = H[0][2]/(h*h) - r[0]*r[2];
        H[1][0] = H[1][0]/(h*h) - r[1]*r[0];
        H[1][1] = H[1][1]/(h*h) - r[1]*r[1] + norm_r;
        H[1][2] = H[1][2]/(h*h) - r[1]*r[2];
        H[2][0] = H[2][0]/(h*h) - r[2]*r[0];
        H[2][1] = H[2][1]/(h*h) - r[2]*r[1]; 
        H[2][2] = H[2][2]/(h*h) - r[2]*r[2] + norm_r;

        // Invert Hessian Matrix
        if (dim == 2) { // Invert 2D explicitly
        det = H[0][0]*H[1][1] - H[0][1]*H[1][0];
        invH[0][0] = H[1][1]/det;
        invH[0][1] = -H[0][1]/det;
        invH[1][0] = -H[1][0]/det;
        invH[1][1] = H[0][0]/det;

        invH[0][2] = invH[1][2] = 0.0; // identity for z-component
        invH[2][0] = invH[2][1] = 0.0;
        invH[2][2] = 1.0;
        }
        else if (dim == 3) { // Invert 3D explicitly
        det = H[0][0] * ( H[2][2]*H[1][1] - H[2][1]*H[1][2] ) - 
              H[1][0] * ( H[2][2]*H[0][1] - H[2][1]*H[0][2] ) + 
              H[2][0] * ( H[1][2]*H[0][1] - H[1][1]*H[0][2] );

        invH[0][0] = +( H[2][2]*H[1][1] - H[2][1]*H[1][2] ) / det;
        invH[0][1] = -( H[2][2]*H[0][1] - H[2][1]*H[0][2] ) / det;
        invH[0][2] = +( H[1][2]*H[0][1] - H[1][1]*H[0][2] ) / det;

        invH[1][0] = -( H[2][2]*H[1][0] - H[2][0]*H[1][2] ) / det;
        invH[1][1] = +( H[2][2]*H[0][0] - H[2][0]*H[0][2] ) / det;
        invH[1][2] = -( H[1][2]*H[0][0] - H[1][0]*H[0][2] ) / det;

        invH[2][0] = +( H[2][1]*H[1][0] - H[2][0]*H[1][1] ) / det;
        invH[2][1] = -( H[2][1]*H[0][0] - H[2][0]*H[0][1] ) / det;
        invH[2][2] = +( H[1][1]*H[0][0] - H[1][0]*H[0][1] ) / det;

        }

        // Increment lambda1: lambda1 = lambda0 - invH*r 
        {
        lambda1[0] = lambda0[0] - ( invH[0][0]*r[0] + invH[0][1]*r[1] + invH[0][2]*r[2] );
        lambda1[1] = lambda0[1] - ( invH[1][0]*r[0] + invH[1][1]*r[1] + invH[1][2]*r[2] );
        lambda1[2] = lambda0[2] - ( invH[2][0]*r[0] + invH[2][1]*r[1] + invH[2][2]*r[2] );
        }

        // Evaluate change in lambda
        {norm_sq = (lambda0[0]-lambda1[0])*(lambda0[0]-lambda1[0]) +
                  (lambda0[1]-lambda1[1])*(lambda0[1]-lambda1[1]) +
                  (lambda0[2]-lambda1[2])*(lambda0[2]-lambda1[2]);
        }

        // Test if max_iter exceeded or convergence otherwise failed
        {
        if (iter > max_iter) {
          printf("\n\nLME Failed on particle i=%i\n\n",i);
          for (jj = 0; jj < jnum; jj++) 
            printf("%i ",jlist[jj]);
          error->all(FLERR, "Maximum iterations reached without LME convergence to specified tolerance\n");
        }
        if ((isnormal(lambda1[0]) == 0 && lambda1[0] != 0) || 
            (isnormal(lambda1[1]) == 0 && lambda1[1] != 0) ||
            (isnormal(lambda1[2]) == 0 && lambda1[2] != 0)) {

          error->all(FLERR, "Lagrange multipliers reached undefined value (NaN). LME failed to converge\n");
        }
        }

      } while (norm_sq > LMDA_TOL_SQ); // Convergence while loop
      
      // Spatial Gradient of shape functions
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        double dx[3] = { (x[i][0]-x[j][0]),
                         (x[i][1]-x[j][1]),
                         (x[i][2]-x[j][2]) };
        if (dim == 2) {
          gradp[i][dim*jj] = p[i][jj]/(h*h) * ( invH[0][0]*dx[0] + invH[0][1]*dx[1] ); // Add p[i][jj]*K_a*grad(beta) term if beta becomes nonconstant
          gradp[i][dim*jj+1] = p[i][jj]/(h*h) * ( invH[1][0]*dx[0] + invH[1][1]*dx[1] );

        }
        else if (dim == 3) {
          gradp[i][dim*jj] += p[i][jj]/(h*h) * ( invH[0][0]*dx[0] + invH[0][1]*dx[1] + invH[0][2]*dx[2] ); // Add p[i][jj]*K_a*grad(beta) term if beta becomes nonconstant
          gradp[i][dim*jj+1] += p[i][jj]/(h*h) * ( invH[1][0]*dx[0] + invH[1][1]*dx[1] + invH[1][2]*dx[2] );
          gradp[i][dim*jj+2] += p[i][jj]/(h*h) * ( invH[2][0]*dx[0] + invH[2][1]*dx[1] + invH[2][2]*dx[2] );
        }
      }
    } // mp test
  } // mp loop

  // Adjust the below statistics --> print shape function statistics to the terminal
  
  // bond statistics
  if (update->ntimestep > -1) {
    n = 0;
    int count = 0;
    for (i = 0; i < nlocal; i++) {
      itype = type[i];
      if (mask[i] & groupbit && itype == typeMP) {
        n += npartner[i];
        count += 1;
      }
    }
    int nall, countall;
    MPI_Allreduce(&n, &nall, 1, MPI_INT, MPI_SUM, world);
    MPI_Allreduce(&count, &countall, 1, MPI_INT, MPI_SUM, world);
    if (countall < 1) countall = 1;

    // if (comm->me == 0) {
    //   if (screen) {
    //     printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
    //     fprintf(screen, "OTM neighbors:\n");
    //     fprintf(screen, "  max # of neighbors for a single mp = %d\n", maxpartner);
    //     fprintf(screen, "  average # of neighbors/particle in group tlsph = %g\n", (double) nall / countall);
    //     printf(">>========>>========>>========>>========>>========>>========>>========>>========\n\n");
    //   }
    //   if (logfile) {
    //     fprintf(logfile, "\nOTM neighbors:\n");
    //     fprintf(logfile, "  max # of neighbors for a single particle = %d\n", maxpartner);
    //     fprintf(logfile, "  average # of neighbors/particle in group tlsph = %g\n", (double) nall / countall);
    //   }
    // }
  } 


  {
  // //DEBUG: print a file to compare the LME evaluations to those from MATLAB
  // printf("timestep: %lli\n",update->ntimestep);

  // // Files originally printed positions of nodes and mps. These are input to MATLAB to calc LME for comparison
  // // Now files
  // if (update->ntimestep == 1) {
  //   FILE *MP_FILE_X;
  //   FILE *ND_FILE_X;
  //   FILE *FILE_LME;
  //   FILE *FILE_LME_GRAD;
  //   MP_FILE_X = fopen("mp_locations.txt","w");
  //   ND_FILE_X = fopen("nd_locations.txt","w");
  //   FILE_LME = fopen("lme.txt","w");
  //   FILE_LME_GRAD = fopen("lme_grad.txt","w");

  //   if (MP_FILE_X!=NULL && ND_FILE_X!=NULL && FILE_LME!=NULL && FILE_LME_GRAD!=NULL) {
  //     fprintf(MP_FILE_X,"ID\tType\tx\ty\tz\n");
  //     fprintf(ND_FILE_X,"ID\tType\tx\ty\tz\n");
  //     // fprintf(FILE_LME,"%% MP order is the same as in the mp_locations.txt file.\n"
  //     //                 "%% I have printed up to the maxpartner value to make the array uniform.\n"
  //     //                 "%% Non-partnered positions assigned a value of zero\n"
  //     //                 "%% One line represents one mp\n");
  //     // fprintf(FILE_LME_GRAD,"%% MP order is the same as in the mp_locations.txt file.\n"
  //     //                 "%% I have printed up to the maxpartner value to make the array uniform. The order of gradients is x y x y ...\n"
  //     //                 "%% Non-partnered positions assigned a value of zero\n"
  //     //                 "%% One line represents one mp\n");

  //     for (ii = 0; ii < inum; ii++) { // iloop
  //       i = ilist[ii];
  //       itype = type[i];

  //       if (itype == typeND)  // print nodal positions
  //         fprintf(ND_FILE_X,"%i\t%i\t%lf\t%lf\t%lf\n",ii,itype,x[i][0],x[i][1],x[i][2]);
        
  //       else if (itype == typeMP) {
  //         jnum = npartner[i];
  //         jlist = partner[i];

  //         // print mp positions 
  //         fprintf(MP_FILE_X,"%i\t%i\t%lf\t%lf\t%lf\n",ii,itype,x[i][0],x[i][1],x[i][2]);
  //         //print npartner
  //         fprintf(FILE_LME,"%i ",jnum);
  //         fprintf(FILE_LME_GRAD,"%i ",jnum);
  //         for (jj = 0; jj < maxpartner; jj++) {
  //           if (jj < jnum) {
  //             // print LME
  //             fprintf(FILE_LME,"%.8e ",p[i][jj]);
  //             fprintf(FILE_LME_GRAD,"%.8e %.8e ",gradp[i][jj],gradp[i][jj+1]);
  //             }
  //           else {
  //             fprintf(FILE_LME,"%.8e ",0.0);
  //             fprintf(FILE_LME_GRAD,"%.8e %.8e ",0.0,0.0);
  //           }
  //         }
  //         fprintf(FILE_LME,";\n");
  //         fprintf(FILE_LME_GRAD,";\n");
  //       }
  //     }
  //     fclose(MP_FILE_X);
  //     fclose(ND_FILE_X);
  //     fclose(FILE_LME);
  //     fclose(FILE_LME_GRAD);
  //   }
  // }
  }



}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixLME::memory_usage()
{
  int dim = domain->dimension;
  int nmax = atom->nmax;
  int bytes = nmax*sizeof(int);
  bytes += nmax*maxpartner * sizeof(tagint); // partner array
  bytes += nmax * maxpartner * sizeof(double); // p
  bytes += nmax * maxpartner * sizeof(double) * dim; // gradp
  bytes += nmax * sizeof(int); // npartner array
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */
// Change to ignore the nodes
void FixLME::grow_arrays(int nmax) 
{
  memory->grow(atom->npartner, nmax, "otm_lme:npartner"); // Final arg is just for an error message
  memory->grow(atom->partner, nmax, maxpartner, "otm_lme:partner");
  memory->grow(atom->p, nmax, maxpartner, "otm_lme:p");
  memory->grow(atom->gradp, nmax, maxpartner*domain->dimension, "otm_lme:gradp");
}

/* ----------------------------------------------------------------------
 copy values within local atom-based arrays
 ------------------------------------------------------------------------- */
// ERROR: segmentation fault caused by this function, copy_arrays(int, int, int);
// Not sure why exactly
void FixLME::copy_arrays(int i, int j, int /*delflag*/) {
  // int dim = domain->dimension;
  // npartner[j] = npartner[i];
  // for (int m = 0; m < npartner[i]; m++) {
  //   partner[j][m] = partner[i][m];
  //   p[j][m] = p[i][m];
  //   for (int jj = 0; jj < dim; jj++)
  //     gradp[j][dim*m+jj] = gradp[i][dim*m+jj];
  // }

  return;
}

/* ----------------------------------------------------------------------
 pack values in local atom-based arrays for exchange with another proc
 ------------------------------------------------------------------------- */

int FixLME::pack_exchange(int i, double *buf) {
// NOTE: how do I know comm buf is big enough if extreme # of touching neighs
// Comm::BUFEXTRA may need to be increased

//printf("pack_exchange ...\n");
        // int dim = domain->dimension;
        // int m = 0;
        // buf[m++] = npartner[i];
        // for (int n = 0; n < npartner[i]; n++) {
        //         buf[m++] = partner[i][n];
        //         buf[m++] = p[i][n];
        //         for (int jj = 0; jj < dim; jj++) {
        //           buf[m++] = gradp[i][dim*n+jj];
        //         }
        // }
        // return m;
  return 0;
}

/* ----------------------------------------------------------------------
 unpack values in local atom-based arrays from exchange with another proc
 ------------------------------------------------------------------------- */

int FixLME::unpack_exchange(int nlocal, double *buf) {
  // if (nlocal == nmax) {
  //   //printf("nlocal=%d, nmax=%d\n", nlocal, nmax);
  //   nmax = nmax / DELTA * DELTA;
  //   nmax += DELTA;
  //   grow_arrays(nmax);

  //   error->message(FLERR,
  //     "in FixLME::unpack_exchange: local arrays too small for receiving partner information; growing arrays");
  // }
  // //printf("nlocal=%d, nmax=%d\n", nlocal, nmax);

  // int dim = domain->dimension;
  // int m = 0;
  // npartner[nlocal] = static_cast<int>(buf[m++]);
  // for (int n = 0; n < npartner[nlocal]; n++) {
  //   partner[nlocal][n] = static_cast<tagint>(buf[m++]);
  //   p[nlocal][n] = static_cast<float>(buf[m++]);
  //   for (int jj = 0; jj < dim; jj++) {
  //     gradp[nlocal][dim*n+jj] = static_cast<float>(buf[m++]);
  //   }
  // }
  // return m;
  return 0;
}

/* ----------------------------------------------------------------------
 pack values in local atom-based arrays for restart file
 ------------------------------------------------------------------------- */
// I'm not 100% sure this is correct
int FixLME::pack_restart(int i, double *buf) {

  int *npartner = atom->npartner;
  int **partner = atom->partner;
  double **p = atom->p;
  double **gradp = atom->gradp;
  int dim = domain->dimension;
  int nlocal = atom->nlocal;
  int m = 0;

  buf[m++] = (2+dim) * npartner[i] + 2; // Is this the correct numbering? 
  buf[m++] = npartner[i];
  for (int n = 0; n < npartner[i]; n++) {
    buf[m++] = partner[i][n]; // +1
    buf[m++] = p[i][n]; // +2
    for (int jj = 0; jj < dim; jj++) {
      gradp[nlocal][dim*n+jj] = static_cast<float>(buf[m++]); 
    } // +dim
  }
  return m;
 //return 0;
}

/* ----------------------------------------------------------------------
 unpack values from atom->extra array to restart the fix
 ------------------------------------------------------------------------- */
// Not finished yet...
void FixLME::unpack_restart(int nlocal, int nth) 
{
//ipage = NULL if being called from granular pair style init()

// skip to Nth set of extra values

  int *npartner = atom->npartner;
  int **partner = atom->partner;

  double **extra = atom->extra;

  int m = 0;
  for (int i = 0; i < nth; i++)
          m += static_cast<int>(extra[nlocal][m]);
  m++;

  // allocate new chunks from ipage,dpage for incoming values

  npartner[nlocal] = static_cast<int>(extra[nlocal][m++]);
  for (int n = 0; n < npartner[nlocal]; n++) {
          partner[nlocal][n] = static_cast<tagint>(extra[nlocal][m++]);
  }

 return;

}

/* ----------------------------------------------------------------------
 maxsize of any atom's restart data
 ------------------------------------------------------------------------- */

int FixLME::maxsize_restart() {
// maxtouch_all = max # of touching partners across all procs

  int maxtouch_all;
  int dim = domain->dimension;
  MPI_Allreduce(&maxpartner, &maxtouch_all, 1, MPI_INT, MPI_MAX, world);

  return (2+dim) * maxtouch_all + 2;
}

/* ----------------------------------------------------------------------
 size of atom nlocal's restart data
 ------------------------------------------------------------------------- */

int FixLME::size_restart(int nlocal) {
  int dim = domain->dimension;
  int *npartner = atom->npartner;

  return (2 + dim) * npartner[nlocal] + 2;
}


