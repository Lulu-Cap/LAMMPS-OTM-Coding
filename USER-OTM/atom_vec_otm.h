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

#ifdef ATOM_CLASS

AtomStyle(otm,AtomVecOTM)

#else

#ifndef LMP_ATOM_VEC_OTM_H
#define LMP_ATOM_VEC_OTM_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecOTM : public AtomVec {
 public:
  AtomVecOTM(class LAMMPS *);
  ~AtomVecOTM() {}
  void init();
  void grow(int);
  void grow_reset();
  void copy(int, int, int);
  void force_clear(int, size_t);
  int pack_comm(int, int *, double *, int, int *);
  int pack_comm_vel(int, int *, double *, int, int *);
  int pack_comm_hybrid(int, int *, double *);
  void unpack_comm(int, int, double *);
  void unpack_comm_vel(int, int, double *);
  int unpack_comm_hybrid(int, int, double *);
  int pack_reverse(int, int, double *);
  int pack_reverse_hybrid(int, int, double *);
  void unpack_reverse(int, int *, double *);
  int unpack_reverse_hybrid(int, int *, double *);
  int pack_border(int, int *, double *, int, int *);
  int pack_border_vel(int, int *, double *, int, int *);
  int pack_border_hybrid(int, int *, double *);
  void unpack_border(int, int, double *);
  void unpack_border_vel(int, int, double *);
  int unpack_border_hybrid(int, int, double *);
  int pack_exchange(int, double *);
  int unpack_exchange(double *);
  int size_restart();
  int pack_restart(int, double *);
  int unpack_restart(double *);
  void create_atom(int, double *);
  void data_atom(double *, imageint, char **);
  int data_atom_hybrid(int, char **);
  void data_vel(int, char **);
  int data_vel_hybrid(int, char **);
  void pack_data(double **);
  int pack_data_hybrid(int, double *);
  void write_data(FILE *, int, double **);
  int write_data_hybrid(FILE *, double *);
  void pack_vel(double **);
  int pack_vel_hybrid(int, double *);
  void write_vel(FILE *, int, double **);
  int write_vel_hybrid(FILE *, double *);
  bigint memory_usage();

 private:
  imageint *image;
  double *radius; 

  tagint *molecule;
  double *contact_radius, **smd_data_9, *e, *de, **vest;
  double *eff_plastic_strain;
  double *damage;
  double *eff_plastic_strain_rate;

  // USER-OTM
  tagint *tag;
  int *type,*mask;
  double **x,**v,**f;
  double *rmass;

  double *vfrac,**x0;

  int *npartner; // Number of nodal partners to mps
  int **partner; // Order of partners
  double **p; // Shape function evaluations
  double **gradp; // Shape function derivatives

  double **def_grad; // Deformation Gradient tensor, F
  double **def_incr; // Incremental Def. Grad. tensor, Fincr
  double **def_rate; // Deformation rate tensor, Fdot

  double **smd_stress; // Cauchy stress tensor 2D:(Sxx Sxy Syy)  3D:(Sxx Syy Szz Sxy Sxz Syz)

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Per-processor system is too big

The number of owned atoms plus ghost atoms on a single
processor must fit in 32-bit integer.

E: Invalid atom type in Atoms section of data file

Atom types must range from 1 to specified # of types.

E: Invalid radius in Atoms section of data file

Radius must be >= 0.0.

E: Invalid density in Atoms section of data file

Density value cannot be <= 0.0.

*/
