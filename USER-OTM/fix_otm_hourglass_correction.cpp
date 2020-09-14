/* ----------------------------------------------------------------------
 *
 *                *** Optimal Transportation Meshfree ***
 *
 * This file is part of the USER-OTM package for LAMMPS.
 * Copyright (2020) Lucas Caparini, lucas.caparini@alumni.ubc.ca
 * Department of Mechanical Engineering, University of British Columbia,
 * British Columbia, Canada
 * 
 * Purpose: This fix will move the nodes then material points. 
 *    - Nodes are moved with standard velocity-verlet time integration 
 *      scheme
 *    - Material points are convected with the nodes after the former's 
 *      integration
 *    - Material point properties are updated after mp movement 
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
#include "fix_otm_hourglass_correction.h"
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

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

/* ----------------------------------------------------------------------
  Initialize pointers and parse the inputs. The args are listed here:
  [0]: FixID
  [1]: groupID - Generally applied to all
  [2]: Fix name
  [3,4]: keyword (MP) and material point style number
  [5,6]: keyword (ND) and node style number

  More arguments may be needed. We will see
------------------------------------------------------------------------- */

FixOTMHGCorrection::FixOTMHGCorrection(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
// fix 1 all otm/integrate MP 1 ND 2 eps ${epsilon}
//    [0][1]        [2]   [3, 4][5,6][7]     [8]

int index, ii, i, j;
int ntypes = atom->ntypes;
char *atom_style = atom->atom_style;

if (strcmp(atom_style, "otm") != 0) 
  error->all(FLERR, "Illegal atom_style for otm hourglass correction. Use OTM style");
if (narg != 9) 
  error->all(FLERR,"Illegal fix otm/hourglass_correction command. Incorrect # of args"); // StC
if (atom->map_style == 0)
  error->all(FLERR, "LME Shape functions require an atom map to evaluate, see atom modify");
if (force->newton_pair) 
  error->all(FLERR, "OTM style cannot be run with newton on"); 

// Parse the input arguments
for (index = 3; index < narg; index +=2) {
  if (strcmp(arg[index],"MP") == 0) {
    // Find atom type of mps
    typeMP = force->numeric(FLERR,arg[index+1]);
    if (typeMP > ntypes) error->all(FLERR,"MP type does not exist");
  }
  else if (strcmp(arg[index],"ND") == 0) {
    // Find atom type of nodes
    typeND = force->numeric(FLERR,arg[index+1]);
    if (typeND > ntypes) error->all(FLERR,"ND type does not exist");
  }
  else if (strcmp(arg[index],"eps") == 0) {
    // Find atom type of nodes
    epsilon = force->numeric(FLERR,arg[index+1]);
    if (epsilon < 0.0) error->all(FLERR,"Hourglass correction parameter must be non-negative");
  }
  else {
    error->all(FLERR,"Unknown keyword identifier for fix otm/integrate_mp");
  }
}

  nevery = 1; // Operation performed every iteration
  time_integrate = 1;
  

  atom->add_callback(0); // 0 for grow, 1 for restart, 2 for border comm (adds fix to a list of fixes to perform, I think...)
}

/* ---------------------------------------------------------------------- */

int FixOTMHGCorrection::setmask()
{
  int mask = 0;
  mask |= POST_FORCE; //Does this need to occur before the setvel cmd?
  return mask;
}

/* ----------------------------------------------------------------------
  Set some flags
------------------------------------------------------------------------- */

void FixOTMHGCorrection::init()
{
  if (force->pair == NULL)
    error->all(FLERR,"Fix otm/integrate_mp requires a pair style be defined");

  if (atom->tag_enable == 0) {
    error->all(FLERR, "Pair style otm requires atoms have IDs");
  }

  // Need an occasional full neighbor list --> investigate closer
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->pair = 0;  
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->half = 0; // Why is half the default setting?
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 0;

}

/* ---------------------------------------------------------------------- */

void FixOTMHGCorrection::init_list(int id, NeighList *ptr) 
{
  list = ptr;
}

/* ----------------------------------------------------------------------
  Add an hourglass error correction to the force on each node. Based on 
  the work by Weibenfels & Wriggers (2018).
------------------------------------------------------------------------- */

void FixOTMHGCorrection::post_force(int /*vflag*/)
{
  int i, j, ii, jj, d1, d2, inum, jnum;
  int *ilist, *jlist;
  int itype, jtype;

  double dx[3], dx0[3], dx2[3];
  double err[3];
  double fincr[3][3];
  double norm_dx0;

  int dim = domain->dimension;

  double **x0 = atom->x0;
  double **x = atom->x;
  double **f = atom->f;
  double **Fincr = atom->def_incr;
  int *mask = atom->mask;
  int *type = atom->type;

  double **p = atom->p;
  double **gradp = atom->gradp;
  int *npartner = atom->npartner;
  int **partner = atom->partner;

  inum = list->inum; 
  ilist = list->ilist; 

  // Updates material point positions and properties
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii]; 
    itype = type[i];

    if (mask[i] & groupbit && itype == typeMP) {
      jlist = partner[i];
      jnum = npartner[i];

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj]; 
        if (type[j] != typeND) error->all(FLERR, "Partner list contains a particle which is not a node!\n");

        dx[0] = x[j][0] - x[i][0];
        dx[1] = x[j][1] - x[i][1];
        dx[2] = x[j][2] - x[i][2];

        dx0[0] = x0[j][0] - x0[i][0];
        dx0[1] = x0[j][1] - x0[i][1];
        dx0[2] = x0[j][2] - x0[i][2];

        // Calculate scaled linear error
        if (dim == 2) {
          norm_dx0 = pow( (dx0[0]*dx0[0] + dx0[1]*dx0[1]) , 0.5);

          dx2[0] = Fincr[i][0]*dx0[0] + Fincr[i][1]*dx0[1];
          dx2[1] = Fincr[i][2]*dx0[0] + Fincr[i][3]*dx0[1];

          err[0] = (dx[0] - dx2[0]) / norm_dx0;
          err[1] = (dx[1] - dx2[1]) / norm_dx0;
        }
        else if (dim == 3) {
          norm_dx0 = pow( (dx0[0]*dx0[0] + dx0[1]*dx0[1] + dx0[2]*dx0[2]) , 0.5);

          dx2[0] = Fincr[i][0]*dx0[0] + Fincr[i][1]*dx0[1] + Fincr[i][2]*dx0[2];
          dx2[1] = Fincr[i][3]*dx0[0] + Fincr[i][4]*dx0[1] + Fincr[i][5]*dx0[2];
          dx2[2] = Fincr[i][6]*dx0[0] + Fincr[i][7]*dx0[1] + Fincr[i][8]*dx0[2];

          err[0] = (dx[0] - dx2[0]) / norm_dx0;
          err[1] = (dx[1] - dx2[1]) / norm_dx0;
          err[2] = (dx[2] - dx2[2]) / norm_dx0;
        }

        // Correct nodal forces
        f[j][0] -= epsilon*p[i][jj]*err[0];
        f[j][1] -= epsilon*p[i][jj]*err[1];
        if (dim == 3) f[j][2] -= epsilon*p[i][jj]*err[2];

      }

    }
  }
  
  return;  
}


