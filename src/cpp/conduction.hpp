#ifndef __CONDUCTION_HPP__
#define __CONDUCTION_HPP__

void conductionT(
  int nz,              // number of points in z grid
  double const * z,    // z grid points
  double dt,           // time step
  double * T,          // vertical temperature profile [K], updated inplace
  double Tsurf,        // surface temperature at time n [K]
  double Tsurfp1,    // ... at time n+1 [K]
  double const * ti,   // thermal inertia [J m^-2 K^-1 s^-1/2]
  double const * rhoc, // heat capacity per volume [J m^-3]
  double Fgeotherm,    // heat flux b.c.
  double * Fsurf);     // heat flux into surface

void conductionQ(
  int nz,
  double const * z,
  double dt,
  double Qn,
  double * Qnp1,
  double * T,
  double const * ti,
  double const * rhoc,
  double emiss,
  double * Tsurf,
  double Fgeotherm,
  double * Fsurf);

#endif // __CONDUCTION_HPP__
