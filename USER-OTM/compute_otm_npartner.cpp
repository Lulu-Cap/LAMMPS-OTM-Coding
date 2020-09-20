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

#include "compute_otm_npartner.h"
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeOTMNPartner::ComputeOTMNPartner(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg)
{
  if (narg != 3)
    error->all(FLERR, "Illegal compute otm/npartner command");
  if (atom->npartner_flag != 1)
    error->all(FLERR, "compute otm/npartner command requires atom_style with material point - node pairs");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  npartnerVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeOTMNPartner::~ComputeOTMNPartner() 
{
  memory->sfree(npartnerVector);
}

/* ---------------------------------------------------------------------- */

void ComputeOTMNPartner::init() 
{
  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style, "otm/npartner") == 0)
      count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR, "More than one compute otm/npartner");
}

/* ---------------------------------------------------------------------- */

void ComputeOTMNPartner::compute_peratom() {
  invoked_peratom = update->ntimestep;

  // grow npartnerVector array if necessary
  if (atom->nmax > nmax) {
    memory->destroy(npartnerVector);
    nmax = atom->nmax;
    memory->create(npartnerVector,nmax,"npartnerVector");
    vector_atom = npartnerVector;
  }

  int *npartner = atom->npartner;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit)
      npartnerVector[i] = npartner[i];
    else
      npartner[i] = -1;
  }

}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeOTMNPartner::memory_usage() {
        double bytes = nmax * sizeof(double);
        return bytes;
}
