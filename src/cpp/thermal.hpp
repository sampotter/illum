#ifndef __THERMAL_HPP__
#define __THERMAL_HPP__

#include "common.hpp"
#include "conduction.hpp"

struct thermal_model {
  arma::uword nz;
  double t {0};
  arma::vec z, ti, rhoc, Qprev, F;
  arma::mat T;

  thermal_model(int nfaces,
                std::vector<double> const & depths,
                double thermal_inertia,
                var_t<double, std::vector<double>> const & rhoc,
                var_t<double, std::string> const & init_temp)
  {
    nz = depths.size();

    z = arma::vec(depths);

    ti.set_size(nz);
    ti.fill(thermal_inertia);

    Qprev = arma::vec(nfaces, arma::fill::zeros);

    F = arma::vec(nfaces, arma::fill::zeros);

    /**
     * Set volumetric heat capacity from variant
     */
    if (auto r = boost::get<double>(&rhoc)) {
      this->rhoc.set_size(nz);
      this->rhoc.fill(*r);
    } else {
      this->rhoc = arma::vec(*boost::get<std::vector<double>>(&rhoc));
    }

    /**
     * Set initial temperature from variant
     */
    if (double const * T0 = boost::get<double>(&init_temp)) {
      T.set_size(nz + 1, nfaces);
      T.fill(*T0);
    }
    else if (std::string const * path = boost::get<std::string>(&init_temp)) {
      T.load(*path);
      if (T.n_rows != static_cast<arma::uword>(nz + 1) ||
          T.n_cols != static_cast<arma::uword>(nfaces)) {
        throw std::runtime_error {
          "initial temperature profile is an incompatible size"};
      }
    }
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
