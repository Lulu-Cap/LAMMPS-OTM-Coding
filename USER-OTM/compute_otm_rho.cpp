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

#include "compute_otm_rho.h"
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeOTMRho::ComputeOTMRho(LAMMPS *lmp, int narg, char **arg) :
                Compute(lmp, narg, arg) {
        if (narg != 3)
                error->all(FLERR, "Illegal compute otm/rho command");
        if (atom->vfrac_flag != 1)
                error->all(FLERR, "compute otm/rho command requires atom_style with volume (e.g. otm)");

        peratom_flag = 1;
        size_peratom_cols = 0;

        nmax = 0;
        rhoVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeOTMRho::~ComputeOTMRho() {
        memory->sfree(rhoVector);
}

/* ---------------------------------------------------------------------- */

void ComputeOTMRho::init() {

        int count = 0;
        for (int i = 0; i < modify->ncompute; i++)
                if (strcmp(modify->compute[i]->style, "otm/rho") == 0)
                        count++;
        if (count > 1 && comm->me == 0)
                error->warning(FLERR, "More than one compute otm/rho");
}

/* ---------------------------------------------------------------------- */

void ComputeOTMRho::compute_peratom() {
        invoked_peratom = update->ntimestep;

        // grow rhoVector array if necessary

        if (atom->nmax > nmax) {
                memory->sfree(rhoVector);
                nmax = atom->nmax;
                rhoVector = (double *) memory->smalloc(nmax * sizeof(double), "atom:rhoVector");
                vector_atom = rhoVector;
        }

        double *vfrac = atom->vfrac;
        double *rmass = atom->rmass;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;

        for (int i = 0; i < nlocal; i++) {
                if (mask[i] & groupbit) {
                        rhoVector[i] = rmass[i] / vfrac[i];
                } else {
                        rhoVector[i] = 0.0;
                }
        }

}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeOTMRho::memory_usage() {
        double bytes = nmax * sizeof(double);
        return bytes;
}
