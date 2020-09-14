/* -*- c++ -*- ----------------------------------------------------------
 *
 *                *** Optimal Transportation Meshfree ***
 *
 * This file is part of the USER-OTM package for LAMMPS.
 * Copyright (2020) Lucas Caparini, lucas.caparini@alumni.ubc.ca
 * Department of Mechanical Engineering, University of British Columbia,
 * British Columbia, Canada
 * 
 * Purpose: This fix will move the material points after the nodes have 
 * been repositioned. It will then update the following properties of the 
 * material points:
 *    - Deformation Gradient tensor, F
 *    - Deformation Rate tensor, Fdot
 *    - Volume/density, v and rho
 * More properties may be added as necessary.
 * ----------------------------------------------------------------------- */

/* -*- c++ -*- ----------------------------------------------------------
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

FixStyle(otm/hourglass_correction,FixOTMHGCorrection)

#else

#ifndef LMP_FIX_HGCORRECTION_OTM_H
#define LMP_FIX_HGCORRECTION_OTM_H

#include "fix.h"

namespace LAMMPS_NS {

class FixOTMHGCorrection : public Fix {
 public:
  FixOTMHGCorrection(class LAMMPS *, int, char **); // Parse input parameters, check for errors, add callback, initialize values and pointers
  virtual ~FixOTMHGCorrection(){return;}; 
  int setmask();
  virtual void init(); // Primarily to set some flags and prevent errors
  void init_list(int, class NeighList *);
  void post_force(int);

 protected:
  int typeND, typeMP; // group indexes of nodal and material point groups 
  double epsilon;

  class NeighList *list;
};

}

#endif
#endif

