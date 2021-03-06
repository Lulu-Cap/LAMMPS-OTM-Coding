/* -*- c++ -*- ----------------------------------------------------------
 *
 *                *** Optimal Transportation Meshfree ***
 *
 * This file is part of the USER-OTM package for LAMMPS.
 * Copyright (2020) Lucas Caparini, lucas.caparini@alumni.ubc.ca
 * Department of Mechanical Engineering, University of British Columbia,
 * British Columbia, Canada
 * 
 * Purpose: This pair style applies to linear elastic solids. It uses the 
 *  LME shape functions already established through fix otm/lme to 
 *  transfer forces onto the nodes.
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

/*
TO-DO:
--> Check in settings() method if the LME fix has already run. It MUST run 
    prior to this pairstyle operating
--> Multiple material type interactions, or materials w/ different moduli, 
    don't make any sense. Need to think about this prior to attempting to 
    use. Maybe it makes more sense to make the stress matrix a particle 
    property? Anyway, it can't work to have different stresses for the same 
    mp depending on the node pair. Something must change.
--> Set up energy and virial computation options!!! For now just ignoring them
--> Add a CFL criterion like was done with the SMD package
--> Interprocessor communication (no comm_forward stuff complete yet)
*/

#include "pair_otm_linear_elastic.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "fix.h"
#include "fix_lme.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "respa.h"
#include "math_const.h"
#include "modify.h"
#include "memory.h"
#include "error.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define NMAT_FULL 9
#define NMAT_SYMM 6
#define MAX(A,B) ((A) > (B) ? (A) : (B))

/* ---------------------------------------------------------------------- */

PairOTMLinearElastic::PairOTMLinearElastic(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1; // Write coeffs to data file
  centroidstressflag = 4; // centroid/stress/atom compute not implemented

  detF = NULL;
  strain_measure = stress_measure = -1;

  E = nu = NULL;

  dtCFL = 0.0; // initialize so safe if extracted on zeroth timestep

  comm_forward = 10; // 9 for Deformation Gradient Tensor, 1 for mass //DEBUG: I haven't done any comm stuff on this file

  // Grow and initialize arrays
  int nmax = atom->nmax;

  grow_arrays(atom->nmax);
  double **Cauchy = atom->smd_stress;
  for (int ii = 0; ii < nmax; ii++) {
    detF[ii] = 1.0;
    for (int jj = 0; jj < NMAT_SYMM; jj++){
      Cauchy[ii][jj] = 0.0;
    }
  }

}

/* ---------------------------------------------------------------------- */
// TODO: add variables as necessary.
PairOTMLinearElastic::~PairOTMLinearElastic()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(detF);

    memory->destroy(E);
    memory->destroy(nu);

  }
  return;
}

/* ----------------------------------------------------------------------
   Compute the forces acting on each node
------------------------------------------------------------------------- */
// TO DO: set up energy and virial calculation options
void PairOTMLinearElastic::compute(int eflag, int vflag)
{
  if (eflag != 0 || vflag != 0) {
    eflag = 0;
    vflag = 0;
    // error->all(FLERR,"\n\nEnergy and Virial computations are not set up yet. Do not use fixes which require them. \n"
    //            "Refer to integrate.h/.cpp for details\n\n");
    printf("\n\nEnergy and Virial computations are not currently enabled\n\n");
  }

  int i, j, ii, jj, d, inum, jnum;
  int *ilist, *jlist;
  int itype, jtype;

  int dim = domain->dimension;
  
  double **x = atom->x;
  double **f = atom->f;
  double **F = atom->def_grad;
  double *volume = atom->vfrac;
  int *mask = atom->mask;
  int *type = atom->type;

  if (update->ntimestep == 0) { // LME must occur first
  int LME_index = modify->find_fix_by_style("otm/lme/shape");
  modify->fix[LME_index]->pre_force(0);
  }

  double **p =  atom->p;
  double **gradp = atom->gradp;
  int *npartner = atom->npartner;
  int **partner = atom->partner;

  grow_arrays(atom->nmax);

  double **Cauchy = atom->smd_stress;

  //ev_init(eflag,vflag); 

  inum = list->inum;
  ilist = list->ilist;
  
  // Assign forces to each node
  for (ii = 0; ii < inum; ii++) { // mp loop
    i = ilist[ii];
    itype = type[i];

    int Stress_flag;
    double WvSpeed = 0.0; //longitudinal wave speed for CFL criterion

    if (itype == typeMP) { // mp test
      jlist = partner[i];
      jnum = npartner[i];
      Stress_flag = 0;

      for (jj = 0; jj < jnum; jj++) { // node loop
        j = jlist[jj];
        jtype = type[j];
        if (jtype != typeND) error->all(FLERR, "Partner list contains a particle which is not a node!\n");

        // Convert F to Cauchy Stress 
        if (Stress_flag == 0) {
          DefGrad2Cauchy(F[i],Cauchy[i],E[itype][jtype],nu[itype][jtype],strain_measure,stress_measure); 
          Stress_flag = 1; // Only calculate stress once per mp
        }

        // double dens = m[i]/volume[i];
        // WvSpeed = ( E[itype][jtype] * (1-nu[itype][jtype]) ) / (dens * (1+nu[itype][jtype]) * (1-2*nu[itype][jtype]) ); // DEBUG: something is really fucked up here. Not giving the right answer at all.
        // WvSpeed = pow(WvSpeed,0.5);

        // Stress contribution to force: f += -volume * dot(Cauchy,gradp) 
        if (dim == 2) {
          f[j][0] -= volume[i] * ( gradp[i][dim*jj]*Cauchy[i][0] + gradp[i][dim*jj+1]*Cauchy[i][3] );
          f[j][1] -= volume[i] * ( gradp[i][dim*jj]*Cauchy[i][3] + gradp[i][dim*jj+1]*Cauchy[i][1] );
        }
        else if (dim == 3) {
          f[j][0] -= volume[i] * ( gradp[i][dim*jj]*Cauchy[i][0] 
                                 + gradp[i][dim*jj+1]*Cauchy[i][3] 
                                 + gradp[i][dim*jj+2]*Cauchy[i][4] );
          f[j][1] -= volume[i] * ( gradp[i][dim*jj]*Cauchy[i][3] 
                                 + gradp[i][dim*jj+1]*Cauchy[i][1] 
                                 + gradp[i][dim*jj+2]*Cauchy[i][5] );
          f[j][2] -= volume[i] * ( gradp[i][dim*jj]*Cauchy[i][4] 
                                 + gradp[i][dim*jj+1]*Cauchy[i][5] 
                                 + gradp[i][dim*jj+2]*Cauchy[i][2] );
        }

        // Body force contribution
        /* Current implementation does not include this term. Use gravity fix instead
            If gravity must be included here at a future date, the term may be written 
              f[j][*] += m[i]*p[i][jj]*g[*];
        */

        // CFL check based on wavespeed
        // int LME_index = modify->find_fix_by_style("otm/lme/shape");
        // hMin = modify->fix[LME_index]->compute_scalar();
        // //double h = hNom; // DEBUG
        // double SF = 0.1;
        // double particle_dt = hMin/WvSpeed * SF;
        // dtCFL = (dtCFL < particle_dt) ? dtCFL : particle_dt;
        
        // double dtNom = hNom/WvSpeed * SF;
        // update->dt = dtNom;
        // if (update->ntimestep == 0) {
        //         update->dt = 1.0e-16;
        // }
        // update->dt = 1.0e-6;

      } // node loop
    } // mp test
  } // mp loop
  
  // // DEBUG
  // double **S = Cauchy;
  // double **Fincr = atom->def_incr;
  // printf("\ntimestep = %lli\n",update->ntimestep);
  // printf("p = ");
  // for (int k = 0; k < npartner[16]; k++) {
  //   printf("(%i %.13e)\n",partner[16][k],p[16][k]);
  // }
  // printf("\ngradp = ");
  // for (int k = 0; k < npartner[16]; k++) {
  //   printf("(%.13e %.13e %.13e)\n",gradp[16][2*k],gradp[16][2*k+1],0.0);
  // }
  // printf("\nFincr = |%.13e %.13e %.13e|\n"
  //        "          |%.13e %.13e %.13e|\n"
  //        "          |%.13e %.13e %.13e|\n",Fincr[16][0],Fincr[16][1],0.0,Fincr[16][2],Fincr[16][3],0.0,0.0,0.0,1.0);
  // printf("F = |%.13e %.13e %.13e|\n"
  //        "    |%.13e %.13e %.13e|\n"
  //        "    |%.13e %.13e %.13e|\n",F[16][0],F[16][1],0.0,F[16][2],F[16][3],0.0,0.0,0.0,1.0);
  // printf("S = |%.13e %.13e %.13e|\n"
  //        "    |%.13e %.13e %.13e|\n"
  //        "    |%.13e %.13e %.13e|\n",S[16][0],S[16][3],S[16][4],S[16][3],S[16][1],S[16][5],S[16][4],S[16][5],S[16][2]);
  // printf("f[0][*] = [%.13e %.13e %.13e]\n\n\n\n",f[0][0],f[0][1],f[0][2]);

}

/* ----------------------------------------------------------------------
   allocate all arrays
   This is primarily for arrays which depend on pair interactions. It is 
   not used for 'growing' arrays like the Fix::grow() method is. 
------------------------------------------------------------------------- */

void PairOTMLinearElastic::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(E,n+1,n+1,"pair:E");
  memory->create(nu,n+1,n+1,"pair:nu");
}

/* ----------------------------------------------------------------------
   global settings:
   These args are input directly after the initial pair_style declaration
   They are useful for settings which will apply to all instances of that
   pairstyle. NOT for material parameters!

   Parse the mp and nd styles as inputs w/ the format

      pair_style otm/elastic/linear MP 1 ND 2 hNom ${hNom} strain ${strain_meas} stress ${stress_meas}

   This is similar format to other OTM commands. The stress argument is only 
   required for 2D simulations.
------------------------------------------------------------------------- */

void PairOTMLinearElastic::settings(int narg, char **arg)
{
  int ntypes = atom->ntypes;
  char *atom_style = atom->atom_style;

  if (strcmp(atom_style, "otm") != 0)
    error->all(FLERR, "Illegal atom_style for otm pair_style. Use otm atoms");
  if ( !(narg == 8 || narg == 10) ) error->all(FLERR,"Illegal pair_style command");

  for (int index = 0; index < narg; index += 2) {
    if (strcmp(arg[index],"MP") == 0) {
      typeMP = force->numeric(FLERR,arg[index+1]);
      if (typeMP > ntypes) error->all(FLERR,"mat_points type does not exist");
    }
    else if (strcmp(arg[index],"ND") == 0) {
      typeND = force->numeric(FLERR,arg[index+1]);
      if (typeND > ntypes) error->all(FLERR,"nodes type does not exist");
    }
    else if (strcmp(arg[index],"hNom") == 0) {
      hNom = force->numeric(FLERR,arg[index+1]);
      if (hNom <= 0.0) error->all(FLERR,"Node spacing cannot be < 0");
    }
    else if (strcmp(arg[index],"strain") == 0) {
      strain_measure = force->numeric(FLERR,arg[index+1]);
      if (strain_measure != 0 && strain_measure != 1) error->all(FLERR,"Invalid strain measure");
    }
    else if (strcmp(arg[index],"stress") == 0) {
      stress_measure = force->numeric(FLERR,arg[index+1]);
      if (stress_measure != 0 && stress_measure != 1) error->all(FLERR,"Invalid stress measure");
    }
    else {
      error->all(FLERR,"Unknown keyword identifier for fix otm/integrate_mp");
    }
  }

  // Check if an LME computation has already occurred.
  int LME_index = modify->find_fix_by_style("otm/lme/shape");
  if (LME_index == -1)
    error->all(FLERR,"Must compute LME Shape functions prior to pair interactions");


  return;


}

/* ----------------------------------------------------------------------
  set coeffs for one or more type pairs
  coefficients of the form
  pair_coeff [type1] [type2] [Elastic Modulus] [Poisson Ratio]
      -        [0]     [1]          [2]              [3]

  Mods: Must allow user to specify * * ${E} ${nu} and inside of this script
  work it out to eliminate neighbour list redundancies. (i.e. Since it knows
  which are nds and mps already, it takes those inputs and only builds a 
  list between nd-mp and nd-nd (for the purpose of finding hMin. May 
  eliminate later b/c of its inefficiency)).

------------------------------------------------------------------------- */
// Both the nodes and mps must belong to the same material with this setup!!!
void PairOTMLinearElastic::coeff(int narg, char **arg)
{
  if (narg != 4)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double E_one = force->numeric(FLERR,arg[2]);
  double nu_one = force->numeric(FLERR,arg[3]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      E[i][j] = E_one;
      E[j][i] = E_one;
      nu[i][j] = nu_one;
      nu[j][i] = nu_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   initialize neighbour list
------------------------------------------------------------------------- */

void PairOTMLinearElastic::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------
   init specific to a pair style. Requires a full neighbour list
------------------------------------------------------------------------- */

void PairOTMLinearElastic::init_style()
{
  // Request full neighbour list
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->pair = 1;  
  neighbor->requests[irequest]->fix = 0;
  neighbor->requests[irequest]->half = 0; // Why is half the default setting?
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 0;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
// No clue if this one is correct
double PairOTMLinearElastic::init_one(int i, int j)
{
  if (!allocated)
    allocate();

  if (setflag[i][j] == 0)
    error->all(FLERR, "All pair coeffs are not set");

  if (force->newton == 1)
    error->all(FLERR, "Pair style otm/elastic/linear requires newton off");

// cutoff = sum of max I,J radii
  double *radius = atom->radius;
  double cutoff = radius[i] + radius[j];
  
  return cutoff;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */
// Add more material constants if necessary
void PairOTMLinearElastic::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&E[i][j],sizeof(double),1,fp);
        fwrite(&nu[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairOTMLinearElastic::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,NULL,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&E[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&nu[i][j],sizeof(double),1,fp,NULL,error);
        }
        MPI_Bcast(&E[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&nu[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairOTMLinearElastic::write_restart_settings(FILE *fp)
{
  fwrite(&typeMP,sizeof(int),1,fp);
  fwrite(&typeND,sizeof(int),1,fp);
  fwrite(&hNom,sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairOTMLinearElastic::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR,&typeMP,sizeof(int),1,fp,NULL,error);
    utils::sfread(FLERR,&typeND,sizeof(int),1,fp,NULL,error);
    utils::sfread(FLERR,&hNom,sizeof(double),1,fp,NULL,error);
  }
  MPI_Bcast(&typeMP,1,MPI_INT,0,world);
  MPI_Bcast(&typeND,1,MPI_INT,0,world);
  MPI_Bcast(&hNom,1,MPI_DOUBLE,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairOTMLinearElastic::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,E[i][i],nu[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairOTMLinearElastic::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g\n",i,j,E[i][j],nu[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairOTMLinearElastic::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                         double /*factor_coul*/, double factor_lj,
                         double &fforce)
{
  error->all(FLERR,"The single function for PairOTMLinearElastic is not yet implemented for OTM");
  return 0;
//   double r2inv,r6inv,forcelj,philj;
//
//   r2inv = 1.0/rsq;
//   r6inv = r2inv*r2inv*r2inv;
//   forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
//   fforce = factor_lj*forcelj*r2inv;
//
//   philj = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
//     offset[itype][jtype];
//   return factor_lj*philj;
}

/* ---------------------------------------------------------------------- */
//TODO: add variables as necessary
void *PairOTMLinearElastic::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"detF") == 0) return (void *) detF;
  else if (strcmp(str,"smd/tlsph/dtCFL_ptr") == 0) return (void *) &dtCFL; //string format maintains compatibility with fix_otm_adjust_dt.h/.cpp for now
  else if (strcmp(str,"otm/hMin") == 0) return (void *) &hMin;

  return NULL;
}

/* ----------------------------------------------------------------------
   Grows the atom based arrays the appropriate amount
------------------------------------------------------------------------- */
// TODO: grow more variables as needed.
void PairOTMLinearElastic::grow_arrays(int nmax)
{
  memory->grow(atom->smd_stress, nmax, NMAT_SYMM, "otm_pair_linear_elastic:Cauchy");
  memory->grow(detF, nmax, "otm_pair_linear_elastic:detF");
}

/* ----------------------------------------------------------------------
   Convert the deformation gradient tensor into the Cauchy stress tensor
   assuming a linear elastic isotropic material
   args:
   F: Deformation gradient tensor (For 3d, 11,12,13,21,22,23,31,32,33. For 2d, 11,12,21,22)
   C: Cauchy stress tensor (For 3d, 11,12,13,22,23,33. For 2d, 11,12,22)
   E: Young's modulus
   nu: Poisson's ratio
   strain_type: Specifies the strain measure to use. 
        0 --> Lagrange strain
        1 --> Infinitesimal strain
   stress_type: Specifies the stress measure to use (2D only).
        0 --> Plane strain
        1 --> Plane Stress
------------------------------------------------------------------------- */

void PairOTMLinearElastic::DefGrad2Cauchy(double *F, double *Cauchy, double E, double nu, int strain_type = -1, int stress_type = -1)
{
  int dim = domain->dimension;
  double strain[3][3]; // strain matrix

  if (dim == 2) {
    // Make strain matrix
    if (strain_type == 0) { // Lagrange strain = 0.5*(F^T * F - I)
      strain[0][0] = 0.5 * ( F[0]*F[0] + F[2]*F[2] - 1 );
      strain[0][1] = 0.5 * ( F[0]*F[1] + F[2]*F[3] );
      strain[1][0] = 0.5 * ( F[0]*F[1] + F[2]*F[3] );
      strain[1][1] = 0.5 * ( F[1]*F[1] + F[3]*F[3] - 1 );
    }
    else if (strain_type == 1) { // Infinitesimal strain = 0.5*(F^T + F) - I
      strain[0][0] = F[0] - 1;
      strain[0][1] = 0.5 * (F[1] + F[2]);
      strain[1][0] = 0.5 * (F[1] + F[2]);
      strain[1][1] = F[3] - 1;
    }
    else {
      error->all(FLERR,"Strain type must be specified as Lagrange or Infinitesimal "
                       "with otm/elastic/linear pairstyle");
    }
    
    // Make stress matrix
    if (stress_type == 0) { // Plane strain
      double C = E / ((1+nu)*(1-2*nu));
      Cauchy[0] = C * ( (1-nu)*strain[0][0] + nu*strain[1][1] ); // Sxx
      Cauchy[1] = C * ( nu*strain[0][0] + (1-nu)*strain[1][1] ); // Syy
      Cauchy[2] = C * nu * (strain[0][0] + strain[1][1]); // Szz
      Cauchy[3] = C * (1 - 2*nu) * strain[0][1]; // Sxy
      Cauchy[4] = 0.0; // Sxz
      Cauchy[5] = 0.0; // Syz
    }
    else if (strain_type == 1) { // Plane stress
      double C = E / (1 - nu*nu);
      Cauchy[0] = C * ( strain[0][0] + nu*strain[1][1] ); // Sxx
      Cauchy[1] = C * ( nu*strain[0][0] + strain[1][1] ); // Syy
      Cauchy[2] = 0.0; // Szz - Really could be any value though. Make it an optional input (TO DO/debug)
      Cauchy[3] = C * (1 - 2*nu) * strain[0][1]; // Sxy
      Cauchy[4] = 0.0; // Sxz
      Cauchy[5] = 0.0; // Syz
    }
    else { // simply ignore 3D dimension. No physical meaning
      error->all(FLERR,"Stress type must be specified as plane strain or plane stress for 2D simulations "
                       "with otm/elastic/linear pairstyle");
    }

  }
  else if (dim == 3) {
    // Make strain matrix
    if (strain_type == 0) { // Lagrange strain E = 0.5*(F^T * F - I)
      strain[0][0] = 0.5 * ( F[0]*F[0] + F[3]*F[3] + F[6]*F[6] - 1 );
      strain[0][1] = 0.5 * ( F[0]*F[1] + F[3]*F[4] + F[6]*F[7] );
      strain[0][2] = 0.5 * ( F[0]*F[2] + F[3]*F[5] + F[6]*F[8] );
      strain[1][0] = strain[0][1];
      strain[1][1] = 0.5 * ( F[1]*F[1] + F[4]*F[4] + F[7]*F[7] - 1 );
      strain[1][2] = 0.5 * ( F[1]*F[2] + F[4]*F[5] + F[7]*F[8] );
      strain[2][0] = strain[0][2];
      strain[2][1] = strain[1][2];
      strain[2][2] = 0.5 * ( F[2]*F[2] + F[5]*F[5] + F[8]*F[8] - 1 );
    }
    else if (strain_type == 1) { // Infinitesimal strain E = 0.5*(F^T + F) - I
      strain[0][0] = F[0] - 1;
      strain[0][1] = 0.5 * (F[1]+F[3]);
      strain[0][2] = 0.5 * (F[2]+F[6]);
      strain[1][0] = strain[0][1];
      strain[1][1] = F[4] - 1;
      strain[1][2] = 0.5 * (F[5] + F[7]);
      strain[2][0] = strain[0][2];
      strain[2][1] = strain[1][2];
      strain[2][2] = F[8] - 1;
    }
    else {
      error->all(FLERR,"Strain type must be specified as Lagrange or Infinitesimal "
                       "with otm/elastic/linear pairstyle");
    }

    // Make stress matrix
    double C = E / ((1+nu)*(1-2*nu));
    Cauchy[0] = C * ( (1-nu)*strain[0][0] + nu*strain[1][1] + nu*strain[2][2] ); // Sxx
    Cauchy[1] = C * ( nu*strain[0][0] + (1-nu)*strain[1][1] + nu*strain[2][2] ); // Syy
    Cauchy[2] = C * ( nu*strain[0][0] + nu*strain[1][1] + (1-nu)*strain[2][2] ); // Szz
    Cauchy[3] = C * (1-2*nu) * strain[0][1]; // Sxy
    Cauchy[4] = C * (1-2*nu) * strain[0][2]; // Sxz
    Cauchy[5] = C * (1-2*nu) * strain[1][2]; // Syz
  }

  return;

}
