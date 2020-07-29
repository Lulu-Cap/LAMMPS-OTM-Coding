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
--> Eliminate reliance on Eigen in this file
--> When to use half vs. full neighbour lists?
--> reset_dt() function not completed
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

int index;
int ntypes = atom->ntypes;
char *atom_style = atom->atom_style;

if (strcmp(atom_style, "otm") != 0) {
  error->all(FLERR, "Illegal atom_style for material point integration");
}
if (narg != 7) {
  error->all(FLERR,"Illegal fix otm/integrate_mp command"); // Must have at least 7 args 
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
  else {
    error->all(FLERR,"Unknown keyword identifier for fix otm/integrate_mp");
  }
}

  nevery = 1; // Operation performed every iteration
  time_integrate = 1;

  atom->add_callback(0); // 0 for grow, 1 for restart, 2 for border comm (adds fix to a list of fixes to perform, I think...)

}

/* ---------------------------------------------------------------------- */
/*
FixOTMIntegrateMP::~FixOTMIntegrateMP()
{
  if (copymode) return; // What is this?
  else return; // Nothing to delete for this one
}*/

/* ----------------------------------------------------------------------
  Material Points are moved after the nodes have been moved in the 
  initial integration phase. I'm indecisive if I should put this as
  an initial_integrate or post_integrate mask b/c I'm not sure exactly 
  how the order of events happens w/in a stage.
------------------------------------------------------------------------- */
// INITIAL_INTEGRATE or POST_INTEGRATE? Which makes more sense?
int FixOTMIntegrateMP::setmask()
{
  int mask = 0;
  //mask |= INITIAL_INTEGRATE;
  mask |= POST_INTEGRATE; // Should I do this post integrate instead? Would this ensure the correct order? 
  return mask;
}

/* ----------------------------------------------------------------------
  Set some flags
------------------------------------------------------------------------- */
// Investigate closer (when to use full or half neighbor lists)
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
------------------------------------------------------------------------- */

void FixOTMIntegrateMP::post_integrate(int vflag)
{
  int i, j, ii, jj, d, inum, jnum;
  int *ilist, *jlist;
  int itype, jtype;

  double dim = domain->dimension;
  double dt = update->dt;

  double x0[3];
  double **x = atom->x;
  double **v = atom->v;
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
      jlist = partner[i]; // pointer to list of partner node IDs
      jnum = npartner[i]; // Number of nodal partners
      
      for (d = 0; d < dim; d++) {
        x0[d] = x[i][d];
        x[i][d] = 0.0; // zero the mp positions
      }

      for (jj = 0; jj < jnum; jj++) { // for each neighbour node
        j = jlist[jj]; // neighbour index (local)
        j &= NEIGHMASK; 
        
        if (type[j] != typeND) error->all(FLERR, "Partner list contains a particle which is not a node!\n");

        for (d = 0; d < dim; d++) x[i][d] += p[i][jj]*x[j][d];

      }
    }

    for (d = 0; d < dim; d++) v[i][d] = (x[i][d] - x0[d])/dt; // backwards 1st order velocity

  }

  return;  
}

