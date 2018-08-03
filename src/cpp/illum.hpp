#ifndef __ILLUM_HPP__
#define __ILLUM_HPP__

#include "config.hpp"

#include <armadillo>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <fastbvh>
#include <memory>

template <class T>
using opt_t = boost::optional<T>;

struct illum_context {
  illum_context();

  illum_context(
    boost::filesystem::path const & obj_path);

  illum_context(
    boost::filesystem::path const & obj_path,
    boost::filesystem::path const & horizon_obj_path);

  ~illum_context();

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
  
  arma::vec get_direct_radiosity(
    arma::vec const & sun_position,
    arma::mat const & disk_xy,
    opt_t<int> j0,
    opt_t<int> j1);

  inline int get_num_faces() const {
    return num_faces;
  }

  inline void set_horizon_obj_file(boost::filesystem::path const & path) {
    horizon_obj_path = path;
  }

private:
  
  std::vector<Object *> objects;
  std::vector<Object *> bvh_objects;
  BVH bvh;

  opt_t<boost::filesystem::path> horizon_obj_path;

  size_t num_faces;
  arma::mat horizons;
};

void fib_spiral(arma::mat & xy, int n);

#endif // __ILLUM_HPP__
