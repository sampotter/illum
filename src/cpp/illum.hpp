#ifndef __ILLUM_HPP__
#define __ILLUM_HPP__

#include <armadillo>
#include <memory>

struct illum_context {
  illum_context(char const * path, int shape_index = 0);
  ~illum_context();

  void make_A(arma::sp_umat & A, double offset = 1e-5);
  void make_horizons(arma::mat & horizons, int nphi = 361, double theta_eps = 1e-3,
                     double offset = 1e-5);
  void compute_visibility_ratios(arma::mat const & horizons,
                                 arma::vec const & sun_position,
                                 arma::mat const & disk_xy,
                                 arma::vec & ratios,
                                 double sun_radius);
  
private:
  struct impl;
  std::unique_ptr<impl> pimpl;
};

void compute_V(arma::sp_umat const & A, arma::sp_umat & V);

void fib_spiral(arma::mat & xy, int n);

#endif // __ILLUM_HPP__
