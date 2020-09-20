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

#include "compute_otm_defgrad.h"
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

ComputeOTMDefgrad::ComputeOTMDefgrad(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg) 
{
  if (narg != 3)
    error->all(FLERR, "Illegal compute otm/defgrad command");

  peratom_flag = 1;
  
  int dim = domain->dimension;
  if (dim == 2) size_peratom_cols = 5; // F + det(F)
  else if (dim == 3) size_peratom_cols = 10;

  nmax = 0;
  defgradVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeOTMDefgrad::~ComputeOTMDefgrad() 
{
  memory->sfree(defgradVector);
}

/* ---------------------------------------------------------------------- */

void ComputeOTMDefgrad::init() 
{
  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style, "otm/defgrad") == 0)
      count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR, "More than one compute otm/defgrad");
}

/* ---------------------------------------------------------------------- */

void ComputeOTMDefgrad::compute_peratom() 
{
  int dim = domain->dimension;

  double **F = atom->def_grad;
  invoked_peratom = update->ntimestep;

  // grow vector array if necessary
  if (atom->nmax > nmax) {
    memory->destroy(defgradVector);
    nmax = atom->nmax;
    memory->create(defgradVector,nmax,size_peratom_cols,"defgradVector");
    array_atom = defgradVector;
  }

  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (dim == 2) {
        defgradVector[i][0] = F[i][0]; // xx
        defgradVector[i][1] = F[i][1]; // xy
        defgradVector[i][2] = F[i][2]; // yx
        defgradVector[i][3] = F[i][3]; // yy
        defgradVector[i][4] = determinant(F[i]); // J
      }
      else if (dim == 3) {
        defgradVector[i][0] = F[i][0];
        defgradVector[i][1] = F[i][1];
        defgradVector[i][2] = F[i][2];
        defgradVector[i][3] = F[i][3];
        defgradVector[i][4] = F[i][4];
        defgradVector[i][5] = F[i][5];
        defgradVector[i][6] = F[i][6];
        defgradVector[i][7] = F[i][7];
        defgradVector[i][8] = F[i][8];
        defgradVector[i][9] = determinant(F[i]);
      }
    }
    else {
      if (dim == 2) {
        defgradVector[i][0] = 1.0; 
        defgradVector[i][1] = 0.0;
        defgradVector[i][2] = 0.0;
        defgradVector[i][3] = 1.0;
        defgradVector[i][4] = 1.0;
      }
      else if (dim == 3) {
        defgradVector[i][0] = 1.0; 
        defgradVector[i][1] = 0.0;
        defgradVector[i][2] = 0.0;
        defgradVector[i][3] = 0.0;
        defgradVector[i][4] = 1.0; 
        defgradVector[i][5] = 0.0;
        defgradVector[i][6] = 0.0;
        defgradVector[i][7] = 0.0;
        defgradVector[i][8] = 1.0;
        defgradVector[i][9] = 1.0;
      }
    }
  }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeOTMDefgrad::memory_usage() 
{
  double bytes = size_peratom_cols * nmax * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
 Determinant calculation
 ------------------------------------------------------------------------- */

double ComputeOTMDefgrad::determinant(double *def_grad) 
{
  int dim = domain->dimension;
  int det;

  if (dim == 2) {
    double F[2][2] = { {def_grad[0],def_grad[1]},
                        {def_grad[2],def_grad[3]} };
    det = F[0][0]*F[1][1] - F[0][1]*F[1][0];
  }
  else if (dim == 3) {
    double F[3][3] = { {def_grad[0],def_grad[1],def_grad[2]},
                       {def_grad[3],def_grad[4],def_grad[5]},
                       {def_grad[6],def_grad[7],def_grad[8]} };
    det = F[0][0] * ( F[2][2]*F[1][1] - F[2][1]*F[1][2] )
        - F[1][0] * ( F[2][2]*F[0][1] - F[2][1]*F[0][2] )
        + F[2][0] * ( F[1][2]*F[0][1] - F[1][1]*F[0][2] );
  }
  return det;
}