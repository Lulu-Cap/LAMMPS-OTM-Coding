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

/*
TO-DO:
 --> Appropriately grow() the arrays which are allocated for material properties 
      (i.e. detF, CauchyStress, etc.)
 --> 
*/

#ifdef PAIR_CLASS

PairStyle(otm/elastic/linear,PairOTMLinearElastic)

#else

#ifndef LMP_PAIR_OTM_LE_H
#define LMP_PAIR_OTM_LE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairOTMLinearElastic : public Pair {
 public:
  PairOTMLinearElastic(class LAMMPS *);
  virtual ~PairOTMLinearElastic();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_list(int, class NeighList *);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &); // Not implemented yet
  void *extract(const char *, int &);


 protected:

  void grow_arrays(int); // grows the below arrays
  void DefGrad2Cauchy(double *, double *, double, double, int, int); // converts F --> CauchyStress w/ linear elastic relationship

  int typeMP, typeND;
  int strain_measure; // 0 Lagrange, 1 Infinitesimal
  int stress_measure; // 0 plane strain, 1 plane stress
  double *detF;
  double hNom,hMin; // Nominal and minimum nodal spacing
  double dtCFL; // Minimum allowable dt based on CFL and elastic properties.

  double **E, **nu; // Elastic modulus, Poisson ratio

  class NeighList *list;

  // double *particle_dt;

  // int nmax; // max number of atoms on this proc
  // double hMin; // minimum kernel radius for two particles
  // double dtCFL;
  // double dtRelative; // relative velocity of two particles, divided by sound speed
  // int updateFlag;
  // double update_threshold; // updateFlag is set to one if the relative displacement of a pair exceeds update_threshold
  // double cut_comm;

  virtual void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
