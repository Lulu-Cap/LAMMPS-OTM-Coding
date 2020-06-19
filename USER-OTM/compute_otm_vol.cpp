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

#include "compute_otm_vol.h"
#include <mpi.h>
#include <cstring>
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSMDVol::ComputeSMDVol(LAMMPS *lmp, int narg, char **arg) :
                Compute(lmp, narg, arg) {
        if (narg != 3)
                error->all(FLERR, "Illegal compute smd/volume command");
        if (atom->vfrac_flag != 1)
                error->all(FLERR, "compute smd/volume command requires atom_style with density (e.g. smd)");

        scalar_flag = 1;
        peratom_flag = 1;
        size_peratom_cols = 0;

        nmax = 0;
        volVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDVol::~ComputeSMDVol() {
        memory->sfree(volVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDVol::init() {

        int count = 0;
        for (int i = 0; i < modify->ncompute; i++)
                if (strcmp(modify->compute[i]->style, "smd/volume") == 0)
                        count++;
        if (count > 1 && comm->me == 0)
                error->warning(FLERR, "More than one compute smd/volume");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDVol::compute_peratom() {
        invoked_peratom = update->ntimestep;

        // grow volVector array if necessary

        if (atom->nmax > nmax) {
                memory->sfree(volVector);
                nmax = atom->nmax;
                volVector = (double *) memory->smalloc(nmax * sizeof(double), "atom:volVector");
                vector_atom = volVector;
        }

        double *vfrac = atom->vfrac;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;

        for (int i = 0; i < nlocal; i++) {
                if (mask[i] & groupbit) {
                        volVector[i] = vfrac[i];
                } else {
                        volVector[i] = 0.0;
                }
        }
}

/* ---------------------------------------------------------------------- */

double ComputeSMDVol::compute_scalar() {

        invoked_scalar = update->ntimestep;
        double *vfrac = atom->vfrac;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;

        double this_proc_sum_volumes = 0.0;
        for (int i = 0; i < nlocal; i++) {
                if (mask[i] & groupbit) {
                        this_proc_sum_volumes += vfrac[i];
                }
        }

        //printf("this_proc_sum_volumes = %g\n", this_proc_sum_volumes);
        MPI_Allreduce(&this_proc_sum_volumes, &scalar, 1, MPI_DOUBLE, MPI_SUM, world);
        //if (comm->me == 0) printf("global sum_volumes = %g\n", scalar);

        return scalar;

}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDVol::memory_usage() {
        double bytes = nmax * sizeof(double);
        return bytes;
}
