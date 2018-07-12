#ifndef __CONDUCTION_HPP__
#define __CONDUCTION_HPP__

void conductionT(
  int nz,              // number of points in z grid
  double const * z,    // z grid points
  double dt,           // time step
  double * T,          // vertical temperature profile [K], updated inplace
  double Tsurf,        // surface temperature at time n [K]
  double Tsurfp1,      // ... at time n+1 [K]
  double const * ti,   // thermal inertia [J m^-2 K^-1 s^-1/2]
  double const * rhoc, // heat capacity per volume [J m^-3]
  double Fgeotherm,    // heat flux b.c.
  double * Fsurf);     // heat flux into surface

void conductionQ(
  int nz,              // number of points in z grid
  double const * z,    // z grid points
  double dt,           // time step
  double Qn,           // net solar irradiance at time step n [W m^-2]
  double Qnp1,         // ... at time step n + 1 [W m^-2]
  double * T,          // vertical temperature profile [K] (output)
  double const * ti,   // thermal inertia [J m^-2 K^-1 s^-1/2]
  double const * rhoc, // rho = density, c = spec. heat [J K^-1 m^-3]
  double emiss,        // emissivity
  double * Tsurf,      // surface temperature [K] (output)
  double Fgeotherm,    // geothermal flux at bottom boundary [W/m^2]
  double * Fsurf);     // heat flux at surface [W/m^2]

#endif // __CONDUCTION_HPP__
