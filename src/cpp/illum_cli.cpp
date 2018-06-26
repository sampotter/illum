#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <experimental/optional>

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

void timed(std::string const & message, std::function<void()> func) {
  std::cout << message << std::flush;
  tic();
  func();
  std::cout << " [" << toc() << "s]" << std::endl;
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
     cxxopts::value<int>()->default_value("361"))
    ("output_file", "Output file", cxxopts::value<std::string>())
    ("horizon_file", "Horizon file", cxxopts::value<std::string>())
    ("sun_pos_file", "File containing sun positions",
     cxxopts::value<std::string>());

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

  std::experimental::optional<std::string> output_file;
  if (args.count("output_file") != 0) {
    output_file = args["output_file"].as<std::string>();
  }

  std::experimental::optional<std::string> horizon_file;
  if (args.count("horizon_file") != 0) {
    horizon_file = args["horizon_file"].as<std::string>();
  }

  std::experimental::optional<std::string> sun_pos_file;
  if (args.count("sun_pos_file") != 0) {
    sun_pos_file = args["sun_pos_file"].as<std::string>();
  }

  std::set<std::string> tasks = {
    "visibility",
    "horizons",
    "ratios"
  };

  if (tasks.find(task) == tasks.end()) {
    std::cout << options.help() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  illum_context context {path.c_str(), shape_index};

  if (task == "visibility") {

    arma::sp_umat A_before, A_after, V;

    timed("- assembling A", [&] () { context.make_A(A_before); });

    timed("- pruning A", [&] () { context.prune_A(A_before, A_after, offset); });

    timed("- computing V", [&] () { compute_V(A_after, V); });

    timed("- writing HDF5 files", [&] () {
      write_csc_inds(A_before, "A_before.h5");
      write_csc_inds(A_after, "A_after.h5");
      write_csc_inds(V, "V.h5");
    });

  } else if (task == "horizons") {

    arma::mat horizons;

    timed(
      "- building horizon map",
      [&] () {
        context.make_horizons(horizons, nphi, theta_eps, offset);
      });

    if (!output_file) {
      *output_file = "horizons.h5";
    }

    timed(
      "- saving horizon map to " + *output_file,
      [&] () {
        horizons.save(arma::hdf5_name(*output_file, "horizons"));
      });

  } else if (task == "ratios") {

    arma::mat horizons;

    if (horizon_file) {
      timed(
        "- loading horizon map from " + *horizon_file,
        [&] () {
          horizons.load(arma::hdf5_name(*horizon_file, "horizons"));
        });
    } else {
      timed(
        "- building horizon map",
        [&] () {
          context.make_horizons(horizons, nphi, theta_eps, offset);
        });
    }

    auto r_sun = 1391400000./2.; // m

    arma::mat sun_positions;

    if (sun_pos_file) {
      timed("- loading sun positions", [&] () {
        sun_positions.load(*sun_pos_file, arma::raw_ascii);
        sun_positions = sun_positions.t();
      });
    } else {
      auto d_sun = 227390024000.; // m
      sun_positions.resize(3, 1);
      sun_positions.col(0) = d_sun*normalise(arma::randn<arma::vec>(3));
    }

    arma::mat disk_xy;
    fib_spiral(disk_xy, 100);

    arma::mat ratios(horizons.n_cols, sun_positions.n_cols);
    arma::vec tmp_ratio(horizons.n_cols);

    for (int j = 0; j < sun_positions.n_cols; ++j) {
      auto sun_pos = sun_positions.col(j);

      timed("- computing ratios", [&] () {
        context.compute_visibility_ratios(
          horizons,
          sun_pos,
          disk_xy,
          tmp_ratio,
          r_sun);
      });

      ratios.col(j) = tmp_ratio;
    }

    ratios.save(arma::hdf5_name("ratios.h5", "ratios"));
  }
}
