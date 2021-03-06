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

ComputeStyle(otm/vol,ComputeOTMVol)

#else

#ifndef LMP_COMPUTE_OTM_VOL_H
#define LMP_COMPUTE_OTM_VOL_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeOTMVol : public Compute {
 public:
  ComputeOTMVol(class LAMMPS *, int, char **);
  ~ComputeOTMVol();
  void init();
  void compute_peratom();
  double compute_scalar();
  double memory_usage();

 private:
  int nmax;
  double *volVector;
};

}

#endif
#endif
