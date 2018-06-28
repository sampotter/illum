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

#include <boost/filesystem.hpp>
#include <experimental/optional>

template <class T>
using opt_t = std::experimental::optional<T>;

#include <cxxopts.hpp>

#include <config.hpp>

#include "constants.hpp"
#include "illum.hpp"
#include "timer.hpp"

#if USE_MPI

#include <hdf5.h>
#include <mpi.h>

int mpi_size, mpi_rank;
MPI_Comm comm = MPI_COMM_WORLD;
MPI_Info info = MPI_INFO_NULL;

#endif

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

#if USE_MPI
void
par_load_mat(
  std::string const & hdf5_path,
  std::string const & dataset_name,
  arma::mat & mat,
  opt_t<int> & j0,
  opt_t<int> & j1,
  opt_t<int> & n_cols)
{
  hid_t file, dset, plist;
  hid_t dspace, memspace;
  hsize_t dims[2];
  hsize_t count[2], offset[2];
  herr_t status;

  plist = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist, comm, info);
  file = H5Fopen(hdf5_path.c_str(), H5F_ACC_RDONLY, plist);
  H5Pclose(plist);

  plist = H5Pcreate(H5P_DATASET_ACCESS);
  dset = H5Dopen(file, dataset_name.c_str(), plist);
  H5Pclose(plist);

  dspace = H5Dget_space(dset);
  assert(H5Sget_simple_extent_ndims(dspace) == 2);
  H5Sget_simple_extent_dims(dspace, dims, nullptr);

  // NOTE: HDF5 is row-major and Armadillo is column-major---we
  // transpose here (and elsewhere)
  n_cols = dims[0];
  // n_rows = dims[1];

  j0 = (static_cast<double>(mpi_rank)/mpi_size)*dims[0];
  j1 = (static_cast<double>(mpi_rank + 1)/mpi_size)*dims[0];

  int ncols = *j1 - *j0;
  int nrows = dims[1];

  mat.set_size(nrows, ncols);

  count[0] = ncols;
  count[1] = nrows;

  offset[0] = *j0;
  offset[1] = 0;

  H5Sselect_hyperslab(dspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);

  memspace = H5Screate_simple(2, count, nullptr);

  plist = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
  status = H5Dread(dset, H5T_IEEE_F64LE, memspace, dspace, plist, &mat(0));
  H5Pclose(plist);

  H5Sclose(memspace);
  H5Sclose(dspace);
  H5Dclose(dset);

  H5Fclose(file);
}
#endif

void
load_mat(
  std::string const & hdf5_path,
  std::string const & dataset_name,
  arma::mat & mat,
  opt_t<int> & j0,
  opt_t<int> & j1,
  opt_t<int> & n_cols)
{
#if USE_MPI
  par_load_mat(hdf5_path, dataset_name, mat, j0, j1, n_cols);
#else
  mat.load(arma::hdf5_name(hdf5_path, dataset_name));
#endif
}

#if USE_MPI
void
par_save_mat(
  std::string const & hdf5_path,
  std::string const & dataset_name,
  arma::mat const & mat,
  opt_t<int> & n_cols,
  opt_t<int> & n_rows,
  opt_t<int> & j0)
{
  hid_t file, dset, plist;
  hid_t filespace, memspace;
  hsize_t dims[2];
  hsize_t count[2], offset[2];
  herr_t status;

  plist = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist, comm, info);
  file = H5Fcreate(hdf5_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist);
  H5Pclose(plist);

  if (mat.is_vec()) {
    dims[0] = *n_cols;
  } else {
    dims[0] = *n_cols;
    dims[1] = *n_rows;
  }

  int rank = mat.is_vec() ? 1 : 2;

  filespace = H5Screate_simple(rank, dims, nullptr);
  dset = H5Dcreate(file, dataset_name.c_str(), H5T_IEEE_F64LE, filespace,
                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Sclose(filespace);

  if (mat.is_vec()) {
    count[0] = mat.n_elem;
  } else {
    count[0] = mat.n_cols;
    count[1] = mat.n_rows;
  }

  if (mat.is_vec()) {
    offset[0] = *j0;
  } else {
    offset[0] = *j0;
    offset[1] = 0;
  }

  memspace = H5Screate_simple(rank, count, nullptr);

  filespace = H5Dget_space(dset);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);

  plist = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
  status = H5Dwrite(dset, H5T_IEEE_F64LE, memspace, filespace, plist, &mat(0));
  H5Pclose(plist);

  H5Dclose(dset);
  H5Sclose(filespace);
  H5Sclose(memspace);

  H5Fclose(file);
}
#endif

void
save_mat(
  std::string const & hdf5_path,
  std::string const & dataset_name,
  arma::mat const & mat,
  opt_t<int> n_cols,
  opt_t<int> n_rows,
  opt_t<int> j0)
{
#if USE_MPI
  par_save_mat(hdf5_path, dataset_name, mat, n_cols, n_rows, j0);
#else
  mat.save(arma::hdf5_name(hdf5_path, dataset_name));
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

  opt_t<std::string> output_file;
  if (args.count("output_file") != 0) {
    output_file = args["output_file"].as<std::string>();
  }

  opt_t<std::string> horizon_file;
  if (args.count("horizon_file") != 0) {
    horizon_file = args["horizon_file"].as<std::string>();
  }

  opt_t<std::string> sun_pos_file;
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

#if USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &mpi_rank);  
#endif

  illum_context context {path.c_str(), shape_index};

  if (task == "visibility") {

    arma::sp_umat A, V;

    timed("- assembling A", [&] () { context.make_A(A, offset); });

    timed("- computing V", [&] () { compute_V(A, V); });

    timed("- writing HDF5 files", [&] () {
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

    /*
     * Load or create a matrix containing the horizons.
     */ 
    arma::mat horizons;
    opt_t<int> j0, j1, nhoriz;
    if (horizon_file) {
      if (!exists_and_is_file(*horizon_file)) {
        std::cerr << "- horizon file " + *horizon_file + "doesn't exist"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      timed(
        "- loading horizon map from " + *horizon_file,
        [&] () {
          load_mat(*horizon_file, "horizons", horizons, j0, j1, nhoriz);
        });
    } else {
      timed(
        "- building horizon map",
        [&] () { context.make_horizons(horizons, nphi, theta_eps, offset); });
    }

    arma::mat sun_positions;
    if (sun_pos_file) {
      timed("- loading sun positions", [&] () {
        sun_positions.load(*sun_pos_file, arma::raw_ascii);
        sun_positions = sun_positions.t();
      });
    } else {
      // A little test problem using made-up numbers
      auto d_sun = 227390024000.; // m
      sun_positions.resize(3, 1);
      sun_positions.col(0) = d_sun*normalise(arma::randn<arma::vec>(3));
    }

    arma::mat disk_xy;
    fib_spiral(disk_xy, 100);

    int nsunpos = sun_positions.n_cols;

    arma::mat ratios(horizons.n_cols, nsunpos);
    arma::vec tmp(horizons.n_cols);

    for (int j = 0; j < nsunpos; ++j) {
      auto sun_pos = sun_positions.col(j);

      timed("- computing ratios", [&] () {
        context.compute_visibility_ratios(
          horizons, sun_pos, disk_xy, tmp, constants::SUN_RADIUS, j0, j1);
      });

      ratios.col(j) = tmp;
    }

    timed("- writing ratios to ratios.h5", [&] () {
      save_mat("ratios.h5", "ratios", ratios, nhoriz, nsunpos, j0);
    });
  }

#if USE_MPI
  MPI_Finalize();
#endif
}
