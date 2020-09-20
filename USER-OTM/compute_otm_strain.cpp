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

#include "compute_otm_strain.h"
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "update.h"


using namespace std;
using namespace LAMMPS_NS;


/* DEBUG + NOTES:
For now, this must be implemented like this - which is just a repeat of calculations already done in previous files
(e.g. pair_otm_linear_elastic.cpp). If these eventually are concatenated into one pair style file, with multiple 
options, as was done by SMD, then it would be more efficient to store this information, along with the stress
information, inside the fix, and simply copy the stored values to strainVector (or later, stressVector).

For options that don't compute the strain, I would set a flag to show that the strain has no meaning in this case.

An example of how the values could be accessed is shown below:

  double **strain = (double **) force->pair->extract("otm/elastic/linear/strain",itmp);
  if (strain == NULL) {
    error->all(FLERR, "compute otm/strain failed to access strain array");
  } 

*/


/* ---------------------------------------------------------------------- */

ComputeOTMStrain::ComputeOTMStrain(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg) 
{
  if (narg == 3) {
    strain_type = 0; // Lagrange strain is default
  }
  else if (narg == 4) { // read strain type to compute
    strain_type = force->numeric(FLERR,arg[3]);
    if (!(strain_type == 0 || strain_type == 1))
      error->all(FLERR,"Illegal strain type argument for compute otm/strain");
  }
  else {
    error->all(FLERR, "Illegal compute otm/strain command");
  }
  
  peratom_flag = 1;

  int dim = domain->dimension;
  if (dim == 2) size_peratom_cols = 3;
  else if (dim == 3) size_peratom_cols = 6;

  nmax = 0;
  strainVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeOTMStrain::~ComputeOTMStrain() 
{
  memory->sfree(strainVector);
}

/* ---------------------------------------------------------------------- */

void ComputeOTMStrain::init() {
  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style, "otm/strain") == 0)
      count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR, "More than one compute otm/strain");
}

/* ---------------------------------------------------------------------- */

void ComputeOTMStrain::compute_peratom() {
  int dim = domain->dimension;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double **F = atom->def_grad;

  invoked_peratom = update->ntimestep;

  // grow vector array if necessary
  if (atom->nmax > nmax) {
    memory->destroy(strainVector);
    nmax = atom->nmax;
    memory->create(strainVector,nmax,size_peratom_cols,"strainVector");
    array_atom = strainVector;
  }

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {

      if (dim == 2) {
        if (strain_type == 0) { // Lagrange strain
          strainVector[i][0] = 0.5 * ( F[i][0]*F[i][0] + F[i][2]*F[i][2] - 1 ); //xx
          strainVector[i][1] = 0.5 * ( F[i][1]*F[i][1] + F[i][3]*F[i][3] - 1 ); //yy
          strainVector[i][2] = 0.5 * ( F[i][0]*F[i][1] + F[i][2]*F[i][3] ); //xy
        }
        else if (strain_type == 1) { // Infinitesimal strain
          strainVector[i][0] = F[i][0] - 1; //xx
          strainVector[i][1] = F[i][3] - 1; //yy
          strainVector[i][2] = 0.5 * (F[i][1] + F[i][2]); //xy
        }
      }

      else if (dim == 3) {
        if (strain_type == 0) { // Lagrange strain
          strainVector[i][0] = 0.5 * ( F[i][0]*F[i][0] + F[i][3]*F[i][3] + F[i][6]*F[i][6] - 1 ); // xx
          strainVector[i][1] = 0.5 * ( F[i][1]*F[i][1] + F[i][4]*F[i][4] + F[i][7]*F[i][7] - 1 ); // yy
          strainVector[i][2] = 0.5 * ( F[i][2]*F[i][2] + F[i][5]*F[i][5] + F[i][8]*F[i][8] - 1 ); // zz
          strainVector[i][3] = 0.5 * ( F[i][0]*F[i][1] + F[i][3]*F[i][4] + F[i][6]*F[i][7] ); // xy
          strainVector[i][4] = 0.5 * ( F[i][0]*F[i][2] + F[i][3]*F[i][5] + F[i][6]*F[i][8] ); // xz
          strainVector[i][5] = 0.5 * ( F[i][1]*F[i][2] + F[i][4]*F[i][5] + F[i][7]*F[i][8] ); // yz
        }
        else if (strain_type == 1) { // Infinitesimal strain
          strainVector[i][0] = F[i][0] - 1; // xx
          strainVector[i][1] = F[i][4] - 1; // yy
          strainVector[i][2] = F[i][8] - 1; // zz
          strainVector[i][3] = 0.5 * (F[i][1] + F[i][3]); // xy
          strainVector[i][4] = 0.5 * (F[i][2] + F[i][6]); // xz
          strainVector[i][5] = 0.5 * (F[i][5] + F[i][7]); // yz
          
        }
      }

    }

    else {
      for (int j = 0; j < size_peratom_cols; j++) {
        strainVector[i][j] = 0.0;
      }
    }
  }
  
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeOTMStrain::memory_usage() {
  double bytes = size_peratom_cols * nmax * sizeof(double);
  return bytes;
}
