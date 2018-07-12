#include "conduction.hpp"
 
extern "C"
void conductiont_(
  int * nz, double const z[], double * dt, double T[], double * Tsurf,
  double * Tsurfp1, double const ti[], double const rhoc[], double * Fgeotherm,
  double * Fsurf);

void conductionT(
  int nz, double const * z, double dt, double * T, double Tsurf,
  double Tsurfp1, double const * ti, double const * rhoc, double Fgeotherm,
  double * Fsurf)
{
  conductiont_(&nz, z, &dt, T, &Tsurf, &Tsurfp1, ti, rhoc, &Fgeotherm, Fsurf);
}

extern "C"
void conductionq_(
  int * nz, double const z[], double * dt, double * Qn, double * Qnp1,
  double T[], double const ti[], double const rhoc[], double * emiss,
  double * Tsurf, double * Fgeotherm, double * Fsurf);

void conductionQ(
  int nz, double const * z, double dt, double Qn, double Qnp1,
  double * T, double const * ti, double const * rhoc, double emiss,
  double * Tsurf, double Fgeotherm, double * Fsurf)
{
  conductionq_(&nz, z, &dt, &Qn, &Qnp1, T, ti, rhoc, &emiss, Tsurf,
               &Fgeotherm, Fsurf);
}
