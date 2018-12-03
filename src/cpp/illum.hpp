#ifndef __ILLUM_HPP__
#define __ILLUM_HPP__

#include "common.hpp"

struct illum_context {
  illum_context();

  illum_context(
    boost::filesystem::path const & obj_path);

  illum_context(
    boost::filesystem::path const & obj_path,
    boost::filesystem::path const & horizon_obj_path);

  ~illum_context();

  arma::sp_mat compute_F(
    var_t<double, std::string> const & albedo,
    double offset = 1e-5);

  void make_horizons(int nphi = 361, double theta_eps = 1e-3,
                     double offset = 1e-5);

  void save_horizons(boost::filesystem::path const & path) const;
  
  void load_horizons(boost::filesystem::path const & path);
  
  arma::vec get_direct_radiosity(arma::vec const & sun_position,
                                 arma::mat const & disk_xy);

  inline int get_num_faces() const {
    return num_faces;
  }

  inline void set_horizon_obj_file(boost::filesystem::path const & path) {
    horizon_obj_path = path;
  }

  std::vector<Object *> objects;
  std::vector<Object *> bvh_objects;
  BVH bvh;

  opt_t<boost::filesystem::path> horizon_obj_path;

  size_t num_faces;
  arma::mat horizons;
};

void fib_spiral(arma::mat & xy, int n);

#endif // __ILLUM_HPP__
