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
 * been repositioned
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

FixStyle(otm/integrate_mp,FixOTMIntegrateMP)

#else

#ifndef LMP_FIX_INTEGRATE_MP_H
#define LMP_FIX_INTEGRATE_MP_H

#include "fix.h"

namespace LAMMPS_NS {

class FixOTMIntegrateMP : public Fix {
 public:
  FixOTMIntegrateMP(class LAMMPS *, int, char **); // Parse input parameters, check for errors, add callback, initialize values and pointers
  virtual ~FixOTMIntegrateMP() {return;}; 
  int setmask(); // make it preforce 
  virtual void init(); // Primarily to set some flags and prevent errors
  void init_list(int id, NeighList *ptr);
  void post_integrate();
  //void initial_integrate(int);
  void reset_dt() {return;};

 protected:
 
 // I may not need most of these variables actually... Maybe none of them but *list
//  double **x; // position of particles

//   double **p; // Shape function evaluations
//   double **gradp; // Gradient of shape functions (Should I use 3D array?)
  int typeND, typeMP; // group indexes of nodal and material point groups 
  class NeighList *list;

  // I will likely need storage for neighbour list stuff.
//   int nmax; // Maximum number of owned+ghost atoms in arrays on this proc
//   int maxpartner; // The maximum partners of any atom on this proc
//   int *npartner; // # of touching partners of each atom
//   tagint **partner; // global atom IDs for the partners

};

}

#endif
#endif

