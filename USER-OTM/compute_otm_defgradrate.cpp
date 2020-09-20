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

#include "compute_otm_defgradrate.h"
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"


using namespace std;
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeOTMDefgradrate::ComputeOTMDefgradrate(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg) 
{
  if (narg != 3)
    error->all(FLERR, "Illegal compute otm/defgradrate command");

  peratom_flag = 1;
  
  int dim = domain->dimension;
  if (dim == 2) size_peratom_cols = 4;
  else if (dim == 3) size_peratom_cols = 9;

  nmax = 0;
  defgradrateVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeOTMDefgradrate::~ComputeOTMDefgradrate() 
{
  memory->sfree(defgradrateVector);
}

/* ---------------------------------------------------------------------- */

void ComputeOTMDefgradrate::init() 
{
  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style, "otm/defgradrate") == 0)
      count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR, "More than one compute otm/defgradrate");
}

/* ---------------------------------------------------------------------- */

void ComputeOTMDefgradrate::compute_peratom() 
{
  int dim = domain->dimension;

  double **Fdot = atom->def_rate;
  invoked_peratom = update->ntimestep;

  // grow vector array if necessary
  if (atom->nmax > nmax) {
    memory->destroy(defgradrateVector);
    nmax = atom->nmax;
    memory->create(defgradrateVector,nmax,size_peratom_cols,"defgradrateVector");
    array_atom = defgradrateVector;
  }

  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (dim == 2) {
        defgradrateVector[i][0] = Fdot[i][0]; // xx
        defgradrateVector[i][1] = Fdot[i][1]; // xy
        defgradrateVector[i][2] = Fdot[i][2]; // yx
        defgradrateVector[i][3] = Fdot[i][3]; // yy
      }
      else if (dim == 3) {
        defgradrateVector[i][0] = Fdot[i][0];
        defgradrateVector[i][1] = Fdot[i][1];
        defgradrateVector[i][2] = Fdot[i][2];
        defgradrateVector[i][3] = Fdot[i][3];
        defgradrateVector[i][4] = Fdot[i][4];
        defgradrateVector[i][5] = Fdot[i][5];
        defgradrateVector[i][6] = Fdot[i][6];
        defgradrateVector[i][7] = Fdot[i][7];
        defgradrateVector[i][8] = Fdot[i][8];
      }
    }
    else {
      if (dim == 2) {
        defgradrateVector[i][0] = 0.0; 
        defgradrateVector[i][1] = 0.0;
        defgradrateVector[i][2] = 0.0;
        defgradrateVector[i][3] = 0.0;
      }
      else if (dim == 3) {
        defgradrateVector[i][0] = 0.0; 
        defgradrateVector[i][1] = 0.0;
        defgradrateVector[i][2] = 0.0;
        defgradrateVector[i][3] = 0.0;
        defgradrateVector[i][4] = 0.0; 
        defgradrateVector[i][5] = 0.0;
        defgradrateVector[i][6] = 0.0;
        defgradrateVector[i][7] = 0.0;
        defgradrateVector[i][8] = 0.0;
      }
    }
  }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeOTMDefgradrate::memory_usage() 
{
  double bytes = size_peratom_cols * nmax * sizeof(double);
  return bytes;
}