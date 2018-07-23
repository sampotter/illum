#ifndef __ILLUM_HPP__
#define __ILLUM_HPP__

#include "config.hpp"

#include <armadillo>
#include <memory>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

template <class T>
using opt_t = boost::optional<T>;

struct illum_context {
  illum_context(char const * path, int shape_index = 0);
  ~illum_context();

  void make_A(arma::sp_umat & A, double offset = 1e-5);

  arma::sp_mat compute_F(double offset = 1e-5);

  void make_horizons(
    int nphi = 361,
    double theta_eps = 1e-3,
    double offset = 1e-5,
    opt_t<int> j0 = opt_t<int> {},
    opt_t<int> j1 = opt_t<int> {});

  void save_horizons(
    boost::filesystem::path const & path,
    opt_t<int> i0,
    opt_t<int> i1) const;
  
  void load_horizons(
    boost::filesystem::path const & path,
    opt_t<int> i0,
    opt_t<int> i1);
  
  arma::vec get_direct_illum(
    arma::vec const & sun_position,
    arma::mat const & disk_xy,
    double sun_radius,
    opt_t<int> j0,
    opt_t<int> j1);

  int get_num_faces() const;
  
private:
  struct impl;
  std::unique_ptr<impl> pimpl;
};

void compute_V(arma::sp_umat const & A, arma::sp_umat & V);

void fib_spiral(arma::mat & xy, int n);

#endif // __ILLUM_HPP__
