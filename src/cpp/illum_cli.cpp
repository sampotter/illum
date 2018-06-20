#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "illum.hpp"
#include "timer.hpp"

template <typename T>
std::pair<arma::uvec, arma::uvec> get_rowinds_and_colptrs(arma::SpMat<T> const & S) {
  arma::uvec rowinds(S.n_nonzero), colptrs(S.n_cols + 1);
  int i = 0, j = 0;
  colptrs(0) = 0;
  for (auto it = S.begin(); it != S.end(); ++it) {
    rowinds(i++) = it.row();
    if (j != it.col()) {
      colptrs(j = it.col()) = i;
    }
  }
  colptrs(++j) = S.n_nonzero;
  return {rowinds, colptrs};
}

template <typename T>
void write_csc_inds(arma::SpMat<T> const & S, const char * path) {
  auto values = arma::nonzeros(S);
  values.save(arma::hdf5_name(path, "values", arma::hdf5_opts::append));

  arma::uvec rowinds, colptrs;
  std::tie(rowinds, colptrs) = get_rowinds_and_colptrs(S);

  rowinds.save(arma::hdf5_name(path, "rowinds", arma::hdf5_opts::append));
  colptrs.save(arma::hdf5_name(path, "colptrs", arma::hdf5_opts::append));
}

void write_coo(arma::sp_umat const & S, const char * path) {
  std::ofstream f;
  f.open(path);
  for (int j = 0; j < S.n_cols; ++j) {
    for (int i = 0; i < S.n_rows; ++i) {
      if (S(i, j)) {
        f << i << ", " << j << ", " << S(i, j) << std::endl;
      }
    }
  }
  f.close();
}

void usage(std::string const & argv0) {
  std::cout << "usage: " << argv0 << std::endl;
}

int main(int argc, char * argv[])
{

  cxxopts::Options options("phobos_test", "Work in progress...");

  options.add_options()
    ("h,help", "Display usage")
    ("t,task", "Task to do", cxxopts::value<std::string>())
    ("p,path", "Path to input OBJ file", cxxopts::value<std::string>())
    ("s,shape", "Shape index in OBJ file",
     cxxopts::value<int>()->default_value("0"))
    ("o,offset",
     "Offset when calculating visibility from a triangle",
     cxxopts::value<double>()->default_value("1e-5"))
    ("e,eps",
     "Tolerance for theta in calculating horizons",
     cxxopts::value<double>()->default_value("1e-3"))
    ("n,nphi",
     "Number of phi values (linearly spaced in [0, 2pi])",
     cxxopts::value<int>()->default_value("361"));

  options.parse_positional({"task"});

  auto args = options.parse(argc, argv);

  if (args["help"].as<bool>() || args["task"].count() == 0) {
    std::cout << options.help() << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  auto task = args["task"].as<std::string>();
  auto path = args["path"].as<std::string>();
  auto shape_index = args["shape"].as<int>();
  auto offset = args["offset"].as<double>();
  auto theta_eps = args["eps"].as<double>();
  auto nphi = args["nphi"].as<int>();

  illum_context context {path.c_str(), shape_index};

  if (task == "visibility") {

    std::cout << "- assembling A";
    tic();
    arma::sp_umat A_before;
    context.make_A(A_before);
    std::cout << " [" << toc() << "s]" << std::endl;

    std::cout << "- pruning A";
    tic();
    arma::sp_umat A_after;
    context.prune_A(A_before, A_after, offset);
    std::cout << " [" << toc() << "s]" << std::endl;

    std::cout << "- computing V";
    tic();
    arma::sp_umat V;
    compute_V(A_after, V);
    std::cout << " [" << toc() << "s]" << std::endl;

    std::cout << "- writing HDF5 files";
    tic();
    write_csc_inds(A_before, "A_before.h5");
    write_csc_inds(A_after, "A_after.h5");
    write_csc_inds(V, "V.h5");
    std::cout << " [" << toc() << "s]" << std::endl;

  } else if (task == "horizon") {

    std::cout << "- building horizon map";
    tic();
    arma::mat horizons;
    context.make_horizons(horizons, nphi, theta_eps, offset);
    std::cout << " [" << toc() << "s]" << std::endl;

    horizons.save(arma::hdf5_name("horizons.h5", "horizons"));

  } else if (task == "ratios") {

    std::cout << "- building horizon map";
    tic();

    arma::mat horizons;
    context.make_horizons(horizons, nphi, theta_eps, offset);

    std::cout << " [" << toc() << "s]" << std::endl;

    std::cout << "- computing ratios";
    tic();

    auto d_sun = 227390024000.; // m
    auto r_sun = 1391400000./2.; // m
    auto sun_position = r_sun*normalise(arma::randn<arma::vec>(3));

    arma::mat disk_xy;
    fib_spiral(disk_xy, 100);

    arma::vec ratios;

    context.compute_visibility_ratios(
      horizons,
      sun_position,
      disk_xy,
      ratios,
      r_sun);

    std::cout << " [" << toc() << "s]" << std::endl;
  }
}
