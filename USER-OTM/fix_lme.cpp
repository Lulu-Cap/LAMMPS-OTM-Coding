/* ----------------------------------------------------------------------
 *
 *                *** Optimal Transportation Meshfree ***
 *
 * This file is part of the USER-OTM package for LAMMPS.
 * Copyright (2020) Lucas Caparini, lucas.caparini@alumni.ubc.ca
 * Department of Mechanical Engineering, University of British Columbia,
 * British Columbia, Canada
 *
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
#include <cmath>
#include <Eigen/Eigen>
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
#include "otm_math.h"
//#include "otm_kernels.h" //cut out

using namespace Eigen;
using namespace LAMMPS_NS;
using namespace FixConst;
using namespace SMD_Kernels; // Can be cut out 
using namespace std;
using namespace SMD_Math;
#define DELTA 16384 // what?
#define TOL 1.0e-16 // machine precision used for cutoff radii
#define LMDA_TOL_SQ 1.0e-8*1.0e-8

/*
TO-DO:
1. Modify to use init_list. It would be more efficient, since there is no reason
  I need a totally new set of list variables (I don't think)
2. Make groupIDs as args. Identify group->bitmask[nd] and group->bitmask[mp] to 
  use in if() statement
3. Add gradp to atom_vec files
4. Eliminate reliance on Eigen in this file (should be simple)
5. Add correlation_index to atom_vec files and clarify meaning. (may be unnecessary)
*/



/* ----------------------------------------------------------------------
  Initialize pointers and parse the inputs. The args are listed here:
  [0]: FixID
  [1]: groupID - Union of material point and nodal groups
  [2]: Fix name
  [3,4]: keyword (mat_point) and material point groupID
  [5,6]: keyword (nodes) and node groupID
  [7,8]: keyword (Spacing) and nodal spacing parameter
  [9,10]: keyword (Locality) and locality parameter (gamma)

  More arguments may be needed. We will see
------------------------------------------------------------------------- */

FixLME::FixLME(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
// fix 1 water otm/lme/shape mat_points water_mp nodes water_nd Spacing ${h} Locality ${gamma}
//    [0] [1]      [2]          [3]        [4]    [5]     [6]     [7]   [8]    [9]      [10]

int index, ntypes;
char *atom_style = atom->atom_style;

if (strcmp(atom_style, "otm") != 0) {
  error->all(FLERR< "Illegal atom_style for LME Shape function evaluations");
}
if (narg != 10) {
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
    // Find group index corresponding to this id
    groupMP = group->find(arg[index+1]);
    if (groupMP == -1) error->all(FLERR,"mat_points group ID does not exist");
  }
  else if (strcmp(arg[index],"nodes") == 0) {
    groupND = group->find(arg[index+1]);
    if (groupND == -1) error->all(FLERR,"nodes group ID does not exist");
  }
  else if (strcmp(arg[index],"Spacing") == 0) {
    h = force->numeric(FLERR,arg[index+1]);
  }
  else if (strcmp(arg[index],"Locality") == 0) {
    gamma = force->numeric(FLERR,arg[index+1]);
  }
}

  nevery = 1; // Operation performed every iteration
  
  maxpartner = 1; // Initialize one partner
  npartner = NULL; // Vector containing # partners of each atom
  partner = NULL; // Array containing global atom IDs of partners
  p = NULL; // Array of derivatives of shape functions at each partner
  gradp = NULL; // Array of shape functions at each partner

  grow_arrays(atom->nmax); // grow arrays to the corresponding number of atoms
                           // nmax = max # of owned +ghost in arrays on this proc)
  atom->add_callback(0); // 0 for grow, 1 for restart, 2 for border comm (adds fix to a list of fixes to perform, I think...)

  // initialize npartner to 0 so neighbour list creation is OK the 1st time
  int nlocal = atom->nlocal;
  for (int ii = 0; ii < nlocal; ii++) {
    npartner[ii] = 0;
  }

  nmax = 0;
  ntypes = atom->ntypes;
  comm_forward = ntypes; // size from own atoms to ghost atoms

}

/* ---------------------------------------------------------------------- */

FixLME::~FixLME()
{
  if (copymode) return; // What is this?

  /* Caution: memory class may need additional overloads for ragged 3D tensors in the case of the gradp values */
  memory->destroy(npartner);
  memory->destroy(partner);
  memory->destroy(p);
  memory->destroy(gradp); // Must adjust memory->destroy for ragged 3D arrays if I use that approach (which would be more readable)

}

/* ----------------------------------------------------------------------
  Shape functions must be evaluated prior to force evaluations
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
    error->all(FLERR,"Fix otm/lme/shape requires a pair style be defined");)

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
  neighbor->requests[irequest]->pair = 0  
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->half = 1; // Why is half the default setting?
  neighbor->requests[irequest]->full = 0;
  neighbor->requests[irequest]->occasional = 0;

/*
  // check variables

  // if (xstr) {
  //   xvar = input->variable->find(xstr);
  //   if (xvar < 0)
  //     error->all(FLERR,"Variable name for fix setforce does not exist");
  //   if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
  //   else if (input->variable->atomstyle(xvar)) xstyle = ATOM;
  //   else error->all(FLERR,"Variable for fix setforce is invalid style");
  // }
  // if (ystr) {
  //   yvar = input->variable->find(ystr);
  //   if (yvar < 0)
  //     error->all(FLERR,"Variable name for fix setforce does not exist");
  //   if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
  //   else if (input->variable->atomstyle(yvar)) ystyle = ATOM;
  //   else error->all(FLERR,"Variable for fix setforce is invalid style");
  // }
  // if (zstr) {
  //   zvar = input->variable->find(zstr);
  //   if (zvar < 0)
  //     error->all(FLERR,"Variable name for fix setforce does not exist");
  //   if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
  //   else if (input->variable->atomstyle(zvar)) zstyle = ATOM;
  //   else error->all(FLERR,"Variable for fix setforce is invalid style");
  // }

  // // set index and check validity of region

  // if (iregion >= 0) {
  //   iregion = domain->find_region(idregion);
  //   if (iregion == -1)
  //     error->all(FLERR,"Region ID for fix setforce does not exist");
  // }

  // if (xstyle == ATOM || ystyle == ATOM || zstyle == ATOM)
  //   varflag = ATOM;
  // else if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
  //   varflag = EQUAL;
  // else varflag = CONSTANT;
*/
}

/* ----------------------------------------------------------------------
  Compute shape functions and shape function gradients before the first
  integration
------------------------------------------------------------------------- */

void FixLME::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    pre_force(vflag);
  else
    error->all(FLERR,"Fix setchempotential requires verlet method");
}

/* ----------------------------------------------------------------------
  Compute shape functions and shape function gradients before any forces
  are computed
------------------------------------------------------------------------- */

void FixLME::pre_force(int vflag)
{
  int i, j, ii, jj, n, inum, jnum;
  int *ilist, *jlist, *numneigh, **firstneigh;

  int nlocal = atom->nlocal; // number of owned atoms on this proc
  nmax = atom->nmax; // Maximum number of local + ghost atoms on this proc
  grow_arrays(nmax);

  double **x = atom->x;
  double *radius = atom->radius; // Do I need this?
  int *mask = atom->mask; // wtf?
  tagint *tag = atom->tag; // wtf?
  NeighList *list = pair->list; //
  inum = list->inum; // # of atoms neighbours are stored for
  ilist = list->ilist; // local indices of I atoms
  numneigh = list->numneigh; // # of J neighbours for each I atom
  firstneigh = list->firstneigh; // ptr to 1st J int value of each I atom

  // Group bitmasks for material points and nodes
  int groupbitMP = group->bitmask[groupMP];
  int groupbitND = group->bitmask[groupND];

  // Cutoff Radius
  double beta = gamma / (h*h); 
  double Rcut_sq = -log(TOL/beta); 
  int dim = domain->dimension;

  // zero npartner for all current atoms
  for (i = 0; i < nlocal; i++)
    npartner[i] = 0;
  
  // Create a partner list from material points to nodes ONLY
  for (ii = 0; ii < inum; ii++) { // for every atom w/neighbours
    i = ilist[ii]; // atom index

    if (mask[i] & groupbitMP) { // If atom is an mp
      jlist = firstneigh[i]; // pointer to neighbour list J atoms for atom I
      jnum = numneigh[i]; // Number of neighbours of atom I

      for (jj = 0; jj < jnum; jj++) { // for each neighbour
        j = jlist[jj]; // neighbour index (local)
        j &= NEIGHMASK; // NEIGHMASK = 0x3FFFFFFF which eliminates the highest 2 bits in j, 
                      //since they are reserved for a bonds flag, and don't contribute to 
                      //the actual number

        double rsq = 0.0; // Euclidean distance between particles
        for (int d = 0; d < dim; d++) {
          r += (x[i][d] - x[j][d]) * (x[i][d] - x[j][d]);
        }

        if ( (mask[j] & groupbitND) && (rsq <= Rcut_sq)) { // If neigh is node and within cutoff radius
          partner[i][npartner[i]] = j; // Add particle j to partner list of i
          npartner[i]++; // increment npartner
        }
      
      }
    }

    else if (mask[i] & groupbitND) {
      // Only concerned about the mps. Assign non-values
      npartner[i] = -1;
      partner[i] = NULL;
    }
  }


  // Find the max # of partners (could combine with the above)
  {
  maxpartner = 0;
  for (i = 0; i < nlocal; i++)
    maxpartner = MAX(maxpartner, npartner[i]);
  int maxall;
  MPI_Allreduce(&maxpartner, &maxall, 1, MPI_INT, MPI_MAX, world);
  maxpartner = maxall;

  grow_arrays(nmax); // Didn't I already do this?

  // zero shape functions + derivatives
  for (i = 0; i < nlocal; i++) {
    for (jj = 0; jj < maxpartner; jj++) {
      p[i][jj] = 0.0;
      for (int d = 0; d < dim; d++) gradp[i][dim*jj+d] = 0.0;
    }
  }
  }

  // Main Loop: loop though each nd/mp pair and identify evaluate shape function and 
  //    shape function gradient for each.  
  const int max_iter = 100;
  for (ii = 0; ii < inum; ii++) { // mp loop
    i = ilist[ii]; // mp id
    if (mask[i] & groupbitMP) { // mp test
      jnum = npartner[i]; // number of nodal neighbours
      jlist = partner[i]; // indices of nodal neighbours

      // initial values for optimization problem
      double lambda0[3] = {0,0,0}; // Lagrange multipliers
      double lambda1[3] = {0,0,0}; 
      double r[3] = {0,0,0}; // gradient of shape functions w.r.t lambda
      double H[3][3] = {{0,0,0},{0,0,0},{0,0,0}};// Hessian of shape func. w.r.t lambda
      double invH[3][3] = {{1,0,0},{0,1,0},{0,0,1}}; // Inverse Hessian
      double det = 1.0;
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
          double f = -beta * ( (x[i][0]-x[j][0])*(x[i][0]-x[j][0]) + 
                               (x[i][1]-x[j][1])*(x[i][1]-x[j][1]) + 
                               (x[i][2]-x[j][2])*(x[i][2]-x[j][2]) ) + 
                       1/h * ( lambda1[0]*(x[i][0]-x[j][0]) +
                               lambda1[1]*(x[i][1]-x[j][1]) +
                               lambda1[2]*(x[i][2]-x[j][2]) );
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
        if (isnormal(lambda1[0]) == 0 && lambda1[0] != 0 || 
            isnormal(lambda1[1]) == 0 && lambda1[1] != 0 ||
            isnormal(lambda1[2]) == 0 && lambda1[2] != 0) {
          error->all(FLERR, "Lagrange multipliers reached undefined value (NaN). LME failed to converge\n");
        }
        }

        // Evaluate change in lambda
        norm_sq = (lambda0[0]-lambda1[0])*(lambda0[0]-lambda1[0]) +
                  (lambda0[1]-lambda1[1])*(lambda0[1]-lambda1[1]) +
                  (lambda0[2]-lambda1[2])*(lambda0[2]-lambda1[2]);

      } while (norm_sq > LMDA_TOL_SQ); // Convergence while loop
      
      // Spatial Gradient of shape functions
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        double dx[3] = { (x[i][0]-x[j][0]),
                         (x[i][1]-x[j][1]),
                         (x[i][2]-x[j][2]) };
        if (dim == 2) {
          gradp[i][dim*jj] += p[i][jj]/(h*h) * ( invH[0][0]*dx[0] + invH[0][1]*dx[1] ); // Add p[i][jj]*K_a*grad(beta) term if beta becomes nonconstant
          gradp[i][dim*jj+1] += p[i][jj]/(h*h) * ( invH[1][0]*dx[0] + invH[1][1]*dx[1] );
        }
        else if (dim == 3) {
          gradp[i][dim*jj] += p[i][jj]/(h*h) * ( invH[0][0]*dx[0] + invH[0][1]*dx[1] + invH[0][2]*dx[2] ); // Add p[i][jj]*K_a*grad(beta) term if beta becomes nonconstant
          gradp[i][dim*jj+1] += p[i][jj]/(h*h) * ( invH[1][0]*dx[0] + invH[1][1]*dx[1] + invH[1][2]*dx[2] );
          gradp[i][dim*jj+2] += p[i][jj]/(h*h) * ( invH[2][0]*dx[0] + invH[2][1]*dx[1] + invH[2][2]*dx[2] );
        }
      }
      
      // Transfer to atom variables
      for (jj = 0; jj < jnum; jj++) {
        atom->npartner[i] = npartner[i];
        atom->p[i][jj] = p[i][jj];
        for (int d = 0; d < dim; d++) atom->gradp[i][dim*jj+d] = gradp[i][dim*jj+d];
      }
    } // mp test
  } // mp loop


  // Adjust the below statistics --> print shape function statistics to the terminal
  /*
  // bond statistics
  if (update->ntimestep > -1) {
    n = 0;
    int count = 0;
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) { // Wrong if statement. If i is an mp and jj is node (separate groups)
        n += npartner[i];
        count += 1;
      }
    }
    int nall, countall;
    MPI_Allreduce(&n, &nall, 1, MPI_INT, MPI_SUM, world);
    MPI_Allreduce(&count, &countall, 1, MPI_INT, MPI_SUM, world);
    if (countall < 1) countall = 1;

    if (comm->me == 0) {
      if (screen) {
        printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
        fprintf(screen, "OTM neighbors:\n");
        fprintf(screen, "  max # of neighbors for a single mp = %d\n", maxpartner);
        fprintf(screen, "  average # of neighbors/particle in group tlsph = %g\n", (double) nall / countall);
        printf(">>========>>========>>========>>========>>========>>========>>========>>========\n\n");
      }
      if (logfile) {
        fprintf(logfile, "\nTLSPH neighbors:\n");
        fprintf(logfile, "  max # of neighbors for a single particle = %d\n", maxpartner);
        fprintf(logfile, "  average # of neighbors/particle in group tlsph = %g\n", (double) nall / countall);
      }
    }
  } */
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixLME::memory_usage()
{
  int nmax = atom->nmax;
  int bytes = nmax*sizeof(int);
  bytes += nmax*maxpartner * sizeof(tagint); // partner array
  bytes += nmax * maxpartner * sizeof(float); // p
  bytes += nmax * maxpartner * sizeof(float) * domain->dimension; // gradp
  bytes += nmax * sizeof(int); // npartner array
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */
// Change to ignore the nodes
void FixLME::grow_arrays(int nmax) 
{
  memory->grow(npartner, nmax, "otm_lme:npartner"); // Final arg is just for an error message
  memory->grow(partner, nmax, maxpartner, "otm_lme:partner");
  memory->grow(p, nmax, maxpartner, "otm_lme:p");
  memory->grow(gradp, nmax, maxpartner*domain->dimension, "otm_lme:gradp");
}

/* ----------------------------------------------------------------------
 copy values within local atom-based arrays
 ------------------------------------------------------------------------- */

void FixLME::copy_arrays(int i, int j, int /*delflag*/) {
  int dim = domain->dimension;
  npartner[j] = npartner[i];
  for (int m = 0; m < npartner[j]; m++) {
    partner[j][m] = partner[i][m];
    p[j][m] = p[i][m];
    for (int jj = 0; jj < dim; jj++)
      gradp[j][dim*m+jj] = gradp[i][dim*m+jj];
  }
}

/* ----------------------------------------------------------------------
 pack values in local atom-based arrays for exchange with another proc
 ------------------------------------------------------------------------- */

int FixLME::pack_exchange(int i, double *buf) {
// NOTE: how do I know comm buf is big enough if extreme # of touching neighs
// Comm::BUFEXTRA may need to be increased

//printf("pack_exchange ...\n");
        int dim = domain->dimension;
        int m = 0;
        buf[m++] = npartner[i];
        for (int n = 0; n < npartner[i]; n++) {
                buf[m++] = partner[i][n];
                buf[m++] = p[i][n];
                for (int jj = 0; jj < dim; jj++) {
                  buf[m++] = gradp[i][dim*n+jj];
                }
        }
        return m;

}

/* ----------------------------------------------------------------------
 unpack values in local atom-based arrays from exchange with another proc
 ------------------------------------------------------------------------- */

int FixLME::unpack_exchange(int nlocal, double *buf) {
  if (nlocal == nmax) {
    //printf("nlocal=%d, nmax=%d\n", nlocal, nmax);
    nmax = nmax / DELTA * DELTA;
    nmax += DELTA;
    grow_arrays(nmax);

    error->message(FLERR,
      "in FixLME::unpack_exchange: local arrays too small for receiving partner information; growing arrays");
  }
  //printf("nlocal=%d, nmax=%d\n", nlocal, nmax);

  int dim = domain->dimension;
  int m = 0;
  npartner[nlocal] = static_cast<int>(buf[m++]);
  for (int n = 0; n < npartner[nlocal]; n++) {
    partner[nlocal][n] = static_cast<tagint>(buf[m++]);
    p[nlocal][n] = static_cast<float>(buf[m++]);
    for (int jj = 0; jj < dim; jj++) {
      gradp[nlocal][dim*n+jj] = static_cast<float>(buf[m++]);
    }
  }
  return m;
}

/* ----------------------------------------------------------------------
 pack values in local atom-based arrays for restart file
 ------------------------------------------------------------------------- */
// I'm not 100% sure this is correct
int FixLME::pack_restart(int i, double *buf) {
  int dim = domain->dimension;
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
}

/* ----------------------------------------------------------------------
 unpack values from atom->extra array to restart the fix
 ------------------------------------------------------------------------- */
// Not finished yet...
void FixLME::unpack_restart(int nlocal, int nth) 
{
//ipage = NULL if being called from granular pair style init()

// skip to Nth set of extra values

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
  return (2 + dim) * npartner[nlocal] + 2;
}


