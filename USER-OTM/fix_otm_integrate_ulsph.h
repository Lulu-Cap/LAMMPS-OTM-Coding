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

#ifdef FIX_CLASS

FixStyle(smd/integrate_ulsph,FixSMDIntegrateUlsph)

#else

#ifndef LMP_FIX_SMD_INTEGRATE_ULSPH_H
#define LMP_FIX_SMD_INTEGRATE_ULSPH_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSMDIntegrateUlsph : public Fix {
 public:
  FixSMDIntegrateUlsph(class LAMMPS *, int, char **);
  int setmask();
  virtual void init();
  virtual void initial_integrate(int);
  virtual void final_integrate();
  void reset_dt();

 private:
  class NeighList *list;
 protected:
  double dtv,dtf, vlimit, vlimitsq;
  int mass_require;
  bool xsphFlag;
  bool adjust_radius_flag;
  double adjust_radius_factor;
  int min_nn, max_nn; // number of SPH neighbors should lie within this interval

  class Pair *pair;
};

}

#endif
#endif
