#ifndef __THERMAL_HPP__
#define __THERMAL_HPP__

#include "conduction.hpp"

struct thermal_model {
  int nz;
  double t;
  arma::vec z, ti, rhoc, Qprev, F;
  arma::mat T;

  thermal_model(
    int nfaces,
    double thermal_inertia = 70.0,
    double init_temp = 233.0)
  {
    z = {
      0.00197, 0.00434, 0.00718, 0.01060, 0.01469, 0.01960, 0.02549, 0.03257,
      0.04105, 0.05124, 0.06346, 0.07812, 0.09572, 0.11684, 0.14218, 0.17259,
      0.20908, 0.25287, 0.30542
    };

    nz = z.n_elem;

    t = 0;
    
    ti.set_size(nz);
    ti.fill(thermal_inertia);

    rhoc = {
      1.42491e+06, 1.41942e+06, 1.41294e+06, 1.40533e+06, 1.39643e+06,
      1.38608e+06, 1.37414e+06, 1.36048e+06, 1.34503e+06, 1.32780e+06,
      1.30894e+06, 1.28872e+06, 1.26762e+06, 1.24631e+06, 1.22562e+06,
      1.20649e+06, 1.18977e+06, 1.17615e+06, 1.16592e+06
    };

    Qprev.zeros(nfaces);

    F.zeros(nfaces);

    T.set_size(nz + 1, nfaces);
    T.fill(init_temp);
  }

  arma::vec get_radiosity() const {
    return -F;
  }

  void step(double dt, arma::vec const & Q) {
    assert(T.n_cols == Q.n_elem);
    for (auto i = 0ull; i < T.n_cols; ++i) {
      conductionQ(
        nz,            // number of points in z grid
        z.memptr(),    // z grid points
        dt,            // time step
        Qprev(i),      // net solar irradiance at time step n [W m^-2]
        Q(i),          // ... at time step n + 1 [W m^-2]
        &T(1, i),      // vertical temperature profile [K] (output)
        ti.memptr(),   // thermal inertia [J m^-2 K^-1 s^-1/2]
        rhoc.memptr(), // rho = density, c = spec. heat [J K^-1 m^-3]
        0.999,         // emissivity
        &T(0, i),      // surface temperature [K] (output) 
        0.0,           // geothermal flux at bottom boundary [W/m^2]
        &F(i)          // heat flux at surface [W/m^2]
        );
    }
    Qprev = Q;
    t += dt;
  }
};

#endif // __THERMAL_HPP__
