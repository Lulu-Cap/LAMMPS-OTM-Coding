/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef COMPUTE_CLASS

ComputeStyle(smd/ulsph/effm,ComputeSMD_Ulsph_Effm)

#else

#ifndef LMP_COMPUTE_SMD_ULSPH_EFFM_H
#define LMP_COMPUTE_SMD_ULSPH_EFFM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeSMD_Ulsph_Effm : public Compute {
 public:
  ComputeSMD_Ulsph_Effm(class LAMMPS *, int, char **);
  ~ComputeSMD_Ulsph_Effm();
  void init();
  void compute_peratom();
  double memory_usage();

 private:
  int nmax;
  double *dt_vector;
};

}

#endif
#endif
