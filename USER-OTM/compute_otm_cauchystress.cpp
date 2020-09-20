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

#include "compute_otm_cauchystress.h"
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

#define NMAT_SYMM 6

/* To Do:
  - Von Mises equivalent stress could be added as a final term in the stress vector
*/

/* ---------------------------------------------------------------------- */

ComputeOTMCauchyStress::ComputeOTMCauchyStress(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg) 
{
  if (narg != 3)
    error->all(FLERR, "Illegal compute otm/CauchyStress command");

  peratom_flag = 1;
  
  size_peratom_cols = NMAT_SYMM + 1; 

  nmax = 0;
  CauchyStressVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeOTMCauchyStress::~ComputeOTMCauchyStress() 
{
  memory->sfree(CauchyStressVector);
}

/* ---------------------------------------------------------------------- */

void ComputeOTMCauchyStress::init() 
{
  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style, "otm/CauchyStress") == 0)
      count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR, "More than one compute otm/CauchyStress");
}

/* ---------------------------------------------------------------------- */

void ComputeOTMCauchyStress::compute_peratom() 
{
  double **Cauchy = atom->smd_stress;
  invoked_peratom = update->ntimestep;

  // grow vector array if necessary
  if (atom->nmax > nmax) {
    memory->destroy(CauchyStressVector);
    nmax = atom->nmax;
    memory->create(CauchyStressVector,nmax,size_peratom_cols,"CauchyStressVector");
    array_atom = CauchyStressVector;
  }

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double Svm = 0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      CauchyStressVector[i][0] = Cauchy[i][0]; // Sxx
      CauchyStressVector[i][1] = Cauchy[i][1]; // Syy
      CauchyStressVector[i][2] = Cauchy[i][2]; // Szz
      CauchyStressVector[i][3] = Cauchy[i][3]; // Sxy
      CauchyStressVector[i][4] = Cauchy[i][4]; // Sxz
      CauchyStressVector[i][5] = Cauchy[i][5]; // Syz
      Svm = 0.5 * ( (Cauchy[i][0] - Cauchy[i][1]) * (Cauchy[i][0] - Cauchy[i][1]) +
                    (Cauchy[i][0] - Cauchy[i][2]) * (Cauchy[i][0] - Cauchy[i][2]) +
                    (Cauchy[i][1] - Cauchy[i][2]) * (Cauchy[i][1] - Cauchy[i][2]) ) +
            3.0 * ( Cauchy[i][3] * Cauchy[i][3] + 
                    Cauchy[i][4] * Cauchy[i][4] + 
                    Cauchy[i][5] * Cauchy[i][5] );
      CauchyStressVector[i][6] = Svm; // Von Mises Equivalent stress, Svm
    }
    else {
      CauchyStressVector[i][0] = 0.0; 
      CauchyStressVector[i][1] = 0.0;
      CauchyStressVector[i][2] = 0.0;
      CauchyStressVector[i][3] = 0.0;
      CauchyStressVector[i][4] = 0.0; 
      CauchyStressVector[i][5] = 0.0;
      CauchyStressVector[i][6] = 0.0;
    }
  }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeOTMCauchyStress::memory_usage() 
{
  double bytes = size_peratom_cols * nmax * sizeof(double);
  return bytes;
}
