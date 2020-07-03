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

FixStyle(otm/lme/shape,FixLME)

#else

#ifndef LMP_FIX_LME_H
#define LMP_FIX_LME_H

#include "fix.h"

namespace LAMMPS_NS {

class FixLME : public Fix {
  FixLME(class LAMMPS *, int, char **); // Parse input parameters, check for errors, add callback, initialize values and pointers
  virtual ~FixLME(); // delete callback, destroy pointers (memory.h/.cpp may need adjustment for ragged 3D arrays)
  int setmask(); // make it preforce 
  virtual void init(); // Primarily to set some flags and prevent errors
  //void pre_exchange(); // Adds or deletes atoms before neighbour build steps, and after initial integration
  void setup(int); // Does stuff prior to first integration -> it will look nearly identical to preforce
  virtual void pre_force(int vflag); // Where all the computation takes place


  double memory_usage(); // Computes memory usage by this fix
  void grow_arrays();
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  int pack_restart(int, int);
  int unpack_restart(int,int);
  int size_restart(int);
  int maxsize_restart();


 protected:
  double **p; // Shape function evaluations
  double **gradp; // Gradient of shape functions (Should I use 3D array?)
  double gamma, h; // Locality parameter, average spacing
  int groupND, groupMP; // group indexes of nodal and material point groups 

  // I will likely need storage for neighbour list stuff.
  int nmax // Maximum number of owned+ghost atoms in arrays on this proc
  int maxpartner; // The maximum partners of any atom on this proc
  int *npartner; // # of touching partners of each atom
  tagint **partner; // global atom IDs for the partners

  // Actually, don't think I need this
  //tagint **correlation_index; // index that a node holds in the partner list of an mp

 
/*  protected:
  double xvalue,yvalue,zvalue;
  int varflag,iregion;
  char *xstr,*ystr,*zstr;
  char *idregion;
  int xvar,yvar,zvar,xstyle,ystyle,zstyle;
  double foriginal[3],foriginal_all[3];
  int force_flag;

  int maxatom;
  double **sforce; */
};

}

#endif
#endif

