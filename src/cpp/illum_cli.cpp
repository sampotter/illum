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

#define ARMA_USE_SUPERLU 1
#include <armadillo>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

template <class T>
using opt_t = boost::optional<T>;

#include <cxxopts.hpp>

#include <config.hpp>

#include "constants.hpp"
#include "illum.hpp"
#include "thermal.hpp"
#include "timer.hpp"

#if USE_MPI
#include <mpi.h>
int mpi_size, mpi_rank;
MPI_Comm comm = MPI_COMM_WORLD;
MPI_Info info = MPI_INFO_NULL;
#endif

template <typename T>
std::pair<arma::uvec, arma::uvec> get_rowinds_and_colptrs(arma::SpMat<T> const & S) {
  arma::uvec rowinds(S.n_nonzero), colptrs(S.n_cols + 1);
  size_t i = 0, j = 0;
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
void write_csc_inds(arma::SpMat<T> const & S, std::string const & path) {
  auto values = arma::nonzeros(S);
  arma::uvec rowinds, colptrs;
  std::tie(rowinds, colptrs) = get_rowinds_and_colptrs(S);
  values.save(path + "_values.bin");
  rowinds.save(path + "_values.bin");
  colptrs.save(path + "_values.bin");
}

void write_coo(arma::sp_umat const & S, const char * path) {
  std::ofstream f;
  f.open(path);
  for (size_t j = 0; j < S.n_cols; ++j) {
    for (size_t i = 0; i < S.n_rows; ++i) {
      if (S(i, j)) {
        f << i << ", " << j << ", " << S(i, j) << std::endl;
      }
    }
  }
  f.close();
}

void
load_mat(
  std::string const & path,
  arma::mat & mat,
  opt_t<int> i0,
  opt_t<int> i1)
{
#if USE_MPI
  // TODO: this very probably doesn't work... will probably need to
  // cook something up using MPI-IO or disallow this
  std::string node_path = path + "_" + std::to_string(*i0) + "_" +
    std::to_string(*i1) + ".bin";
  mat.load(node_path);
#else
  (void) i0;
  (void) i1;
  mat.load(path);
#endif
}

void
save_mat(
  boost::filesystem::path const & path,
  arma::mat const & mat,
  opt_t<int> i0,
  opt_t<int> i1)
{
#if USE_MPI
  std::string path node_path = path.string() + "_" + std::to_string(*i0) + "_" +
    std::to_string(*i1) + ".bin";
  mat.save(node_path);
#else
  (void) i0;
  (void) i1;
  mat.save(path.string() + ".bin");
#endif
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

bool exists_and_is_file(std::string const & filename) {
  boost::filesystem::path p {filename};
  return boost::filesystem::exists(p) && boost::filesystem::is_regular_file(p);
}

void file_exists_or_die(std::string const & filename) {
  if (!exists_and_is_file(filename)) {
    std::cerr << "- file " << filename << "doesn't exist" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

struct job_params {
  double offset, theta_eps;
  int nphi;
  opt_t<std::string> output_dir, horizon_file, sun_pos_file;

  // TODO: should we use boost units for this eventually?
  std::string sun_unit, mesh_unit;
};

#if USE_MPI
void set_i0_and_i1(int num_faces, opt_t<int> & i0, opt_t<int> & i1) {
  i0 = (static_cast<double>(mpi_rank)/mpi_size)*num_faces;
  i1 = (static_cast<double>(mpi_rank + 1)/mpi_size)*num_faces;
}
#endif

void do_visibility_task(job_params & params, illum_context & context) {
  arma::sp_umat A, V;
  timed("- assembling A", [&] () { context.make_A(A, params.offset); });
  timed("- computing V", [&] () { compute_V(A, V); });
  timed("- writing files", [&] () { write_csc_inds(V, "V"); });
}

void do_horizons_task(job_params & params, illum_context & context) {
  opt_t<int> i0, i1;
#if USE_MPI
  set_i0_and_i1(context.get_num_faces(), i0, i1);
#endif
  arma::mat horizons;
  timed("- building horizon map", [&] () {
    context.make_horizons(
      horizons, params.nphi, params.theta_eps, params.offset, i0, i1);
  });
  boost::filesystem::path output_path {"horizons"};
  if (!params.output_dir) {
    output_path = *params.output_dir/output_path;
  }
  timed("- saving horizon map to " + *params.output_dir, [&] () {
    save_mat(output_path, horizons, i0, i1);
  });
}

void do_direct_illum_task(job_params & params, illum_context & context) {
  int nfaces = context.get_num_faces();
  opt_t<int> i0, i1, nhoriz;

#if USE_MPI
  set_i0_and_i1(nfaces, i0, i1);
#endif

  /*
   * Load or create a matrix containing the horizons.
   */
  arma::mat horizons;
  if (params.horizon_file) {
    file_exists_or_die(*params.horizon_file);
    timed("- loading horizon map from " + *params.horizon_file, [&] () {
      load_mat(*params.horizon_file, horizons, i0, i1);
    });
  } else {
    timed("- building horizon map", [&] () {
      context.make_horizons(
        horizons, params.nphi, params.theta_eps, params.offset, i0, i1);
    });
    nhoriz = nfaces;
  }

  arma::mat sun_positions;
  if (params.sun_pos_file) {
    file_exists_or_die(*params.sun_pos_file);
    timed("- loading sun positions", [&] () {
      sun_positions.load(*params.sun_pos_file, arma::raw_ascii);
      sun_positions = sun_positions.t();
      if (sun_positions.n_rows > 3) {
        sun_positions.shed_rows(3, sun_positions.n_rows - 1);
      }
    });
  } else {
    // A little test problem using made-up numbers
    auto d_sun = 227390024000.; // m
    sun_positions.resize(3, 1);
    sun_positions.col(0) = d_sun*normalise(arma::randn<arma::vec>(3));
  }

  // Rescale the sun positions as necessary
  if (params.sun_unit == "m") {
    sun_positions /= 1000.;
  }

  arma::mat disk_xy;
  fib_spiral(disk_xy, 100);

  int nsunpos = sun_positions.n_cols;

  arma::mat direct(horizons.n_cols, nsunpos);
  arma::mat therm(horizons.n_cols, nsunpos);
  arma::mat rad(horizons.n_cols, nsunpos);
  arma::vec avgdir(direct.n_rows);

  thermal_model therm_model {nfaces};

  // TODO: just temporarily constructing the kernel matrix here to try
  // this out...

  arma::sp_mat F = context.compute_F();
  assert(F.n_rows == static_cast<arma::uword>(nfaces));
  assert(F.n_cols == static_cast<arma::uword>(nfaces));

  arma::sp_mat K = arma::speye(nfaces, nfaces) - 0.12*F;

  for (int j = 0; j < nsunpos; ++j) {
    std::string const frame_str {
      std::to_string(j + 1) + "/" + std::to_string(nsunpos)
    };

    timed("- " + frame_str + ": computing direct illumination", [&] () {
      direct.col(j) = context.get_direct_illum(
        horizons, sun_positions.col(j), disk_xy,
        constants::SUN_RADIUS, i0, i1);

      if (j == 0) {
        avgdir = direct.col(j);
      } else {
        avgdir += (direct.col(j) - avgdir)/(j + 1);
      }
    });

    timed("- " + frame_str + ": stepping thermal model", [&] () {
      therm.col(j) = therm_model.T.row(0).t();

      // TODO: what is 586.2 again?
      // TODO: the sun is a parallel light source
      arma::vec E = 586.2*direct.col(j);
      arma::vec R = arma::spsolve(K, E, "superlu");

      rad.col(j) = R;

      // TODO: check that this is actually what we want to use here...
      therm_model.step(600., R);
    });
  }

  boost::filesystem::path output_dir_path = params.output_dir ?
    *params.output_dir : ".";

  timed("- saving direct illumination", [&] () {
    save_mat(output_dir_path/"direct", direct, i0, i1);
  });

  timed("- saving thermal", [&] () {
    save_mat(output_dir_path/"thermal", therm, i0, i1);
  });

  timed("- saving radiance", [&] () {
    save_mat(output_dir_path/"rad", rad, i0, i1);
  });
}

int main(int argc, char * argv[])
{
  std::set<std::string> tasks = {
    "visibility",
    "horizons",
    "direct"
  };

  auto tasks_to_string = [&] () {
    std::string s;
    for (auto task: tasks) s += "  " + task + '\n';
    return s;
  };

  cxxopts::Options options(
	// boost::filesystem::path {std::string(argv[0])}.filename().c_str(),
    argv[0],
    ("Available tasks:\n\n" + tasks_to_string()).c_str());

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
    ("output_dir", "Output file", cxxopts::value<std::string>())
    ("horizon_file", "Horizon file", cxxopts::value<std::string>())
    ("sun_pos_file", "File containing sun positions",
     cxxopts::value<std::string>())
    ("sun_unit", "Units used for sun positions",
     cxxopts::value<std::string>()->default_value("m"))
    ("mesh_unit", "Units used by OBJ file vertices",
     cxxopts::value<std::string>()->default_value("km"));

  options.parse_positional({"task"});

  auto args = options.parse(argc, argv);
  if (args["help"].as<bool>() || args["task"].count() == 0) {
    std::cout << options.help() << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  /**
   * Parse options related to loading the model from the OBJ file.
   */
  auto task = args["task"].as<std::string>();
  auto path = args["path"].as<std::string>();
  auto shape_index = args["shape"].as<int>();

  /**
   * Parse options which are job parameters.
   */
  job_params params;
  params.offset = args["offset"].as<double>();
  params.theta_eps = args["eps"].as<double>();
  params.nphi = args["nphi"].as<int>();
  if (args.count("output_dir") != 0) {
    params.output_dir = args["output_dir"].as<std::string>();
  }
  if (args.count("horizon_file") != 0) {
    params.horizon_file = args["horizon_file"].as<std::string>();
  }
  if (args.count("sun_pos_file") != 0) {
    params.sun_pos_file = args["sun_pos_file"].as<std::string>();
  }
  params.sun_unit = args["sun_unit"].as<std::string>();
  params.mesh_unit = args["mesh_unit"].as<std::string>();

  assert(params.sun_unit == "m" || params.sun_unit == "km");
  assert(params.mesh_unit == "m" || params.mesh_unit == "km");

  if (params.mesh_unit == "m") {
    std::cout << "TODO: mesh_unit == m isn't implemented yet" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  /**
   * Check that the task that the user request is valid.
   */
  if (tasks.find(task) == tasks.end()) {
    std::cout << options.help() << std::endl;
    std::exit(EXIT_FAILURE);
  }

#if USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &mpi_rank);  
#endif

  illum_context context {path.c_str(), shape_index};

  if (task == "visibility") do_visibility_task(params, context);
  else if (task == "horizons") do_horizons_task(params, context);
  else if (task == "direct") do_direct_illum_task(params, context);

#if USE_MPI
  MPI_Finalize();
#endif
}
