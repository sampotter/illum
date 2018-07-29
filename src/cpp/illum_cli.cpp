#include "common.hpp"

#include <cxxopts.hpp>

#include "arma_util.hpp"
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
  int nphi, gs_steps;
  opt_t<std::string> output_dir, horizon_file, sun_pos_file;
  bool do_radiosity, print_residual, do_thermal, quiet;
  double dt, albedo, thermal_inertia, initial_temperature;

  // TODO: should we use boost units for this eventually?
  std::string sun_unit, mesh_unit;
};

#if USE_MPI
void set_i0_and_i1(int num_faces, opt_t<int> & i0, opt_t<int> & i1) {
  i0 = (static_cast<double>(mpi_rank)/mpi_size)*num_faces;
  i1 = (static_cast<double>(mpi_rank + 1)/mpi_size)*num_faces;
}
#endif

void do_horizons_task(job_params & params, illum_context & context) {
  opt_t<int> i0, i1;
#if USE_MPI
  set_i0_and_i1(context.get_num_faces(), i0, i1);
#endif
  timed("- building horizon map", [&] () {
    context.make_horizons(params.nphi, params.theta_eps, params.offset, i0, i1);
  });
  boost::filesystem::path output_path {"horizons"};
  if (params.output_dir) {
    output_path = *params.output_dir/output_path;
  }
  if (!params.quiet) {
    timed("- saving horizon map to " + output_path.string(), [&] () {
      context.save_horizons(output_path, i0, i1);
    });
  }
}

void do_radiosity_task(job_params & params, illum_context & context) {
  int nfaces = context.get_num_faces();
  opt_t<int> i0, i1, nhoriz;

#if USE_MPI
  set_i0_and_i1(nfaces, i0, i1);
#endif

  /*
   * Load or create a matrix containing the horizons.
   */
  if (params.horizon_file) {
    file_exists_or_die(*params.horizon_file);
    timed("- loading horizon map from " + *params.horizon_file, [&] () {
      // load_mat(*params.horizon_file, horizons, i0, i1);
      context.load_horizons(*params.horizon_file, i0, i1);
    });
  } else {
    timed("- building horizon map", [&] () {
      context.make_horizons(params.nphi, params.theta_eps, params.offset, i0, i1);
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

  arma::vec rad(nfaces), rad_avg(nfaces, arma::fill::zeros);

  arma::vec therm, therm_avg, therm_max;

  if (params.do_thermal) {
    therm.resize(nfaces);

    therm_avg.resize(nfaces);
    therm_avg.zeros();

    therm_max.resize(nfaces);
    therm_max.zeros();
  }

  // TODO: only construct this if we actually need to use it
  thermal_model therm_model {
    nfaces,
    params.thermal_inertia,
    params.initial_temperature
  };

  arma::sp_mat F;

  if (params.do_radiosity) {
    timed("- assembling form factor matrix", [&] () {
      F = params.albedo*context.compute_F();
      std::cout << " [nnz = " << F.n_nonzero << "]";
    });

    assert(F.n_rows == static_cast<arma::uword>(nfaces));
    assert(F.n_cols == static_cast<arma::uword>(nfaces));
  }

  for (int j = 0; j < nsunpos; ++j) {
    std::string const frame_str {
      std::to_string(j + 1) + "/" + std::to_string(nsunpos)
    };

    rad.zeros();

    /**
     * Incorporate heat flux at surface into radiosity.
     */
    if (params.do_thermal) {
      rad += therm_model.get_radiosity();
    }

    timed("- " + frame_str + ": computing direct radiosity", [&] () {
      rad += context.get_direct_radiosity(
        sun_positions.col(j), disk_xy, i0, i1);
      std::cout << " [mean: " << arma::mean(rad) << " W/m^2]";
    });

    if (params.do_radiosity) {
      arma::sp_mat L_t = (arma::speye(nfaces, nfaces) - arma::trimatl(F)).t();
      arma::sp_mat U = -arma::trimatu(F);

      arma::vec E = rad;
      rad = arma_util::forward_solve(L_t, E);
      for (int iter = 0; iter < params.gs_steps; ++iter) {
        timed("  + doing Gauss-Seidel step", [&] () {
          rad = arma_util::forward_solve(L_t, E - U*rad);
          if (params.print_residual) {
            double res = arma::norm((rad.t()*L_t).t() + U*rad - E);
            std::cout << " [res = " << res << "]";
          }
        });
      };
    }

    rad_avg += (rad - rad_avg)/(j + 1);

    if (params.do_thermal) {
      timed("- " + frame_str + ": stepping thermal model", [&] () {
        therm = therm_model.T.row(0).t();
        therm_model.step(params.dt, rad);
        therm_avg += (therm - therm_avg)/(j + 1);
        if (j > 50) {
          therm_max = arma::max(therm_max, therm);
        }
      });
    }
  }

  boost::filesystem::path output_dir_path = params.output_dir ?
    *params.output_dir : ".";

  if (!params.quiet) {
    timed("- saving radiosity", [&] () {
      arma_util::save_mat(rad, output_dir_path/"rad", i0, i1);
    });

    timed("- saving average radiosity", [&] () {
      arma_util::save_mat(rad_avg, output_dir_path/"rad_avg", i0, i1);
    });

    if (params.do_thermal) {
      timed("- saving thermal", [&] () {
        arma_util::save_mat(therm, output_dir_path/"therm", i0, i1);
      });

      timed("- saving average thermal", [&] () {
        arma_util::save_mat(therm_avg, output_dir_path/"therm_avg", i0, i1);
      });

      timed("- saving max thermal", [&] () {
        arma_util::save_mat(therm_max, output_dir_path/"therm_max", i0, i1);
      });
    }
  }
}

int main(int argc, char * argv[]) {
  std::set<std::string> tasks = {
    "horizons",
    "radiosity"
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
    ("task", "Task to do", cxxopts::value<std::string>())
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
    ("r,radiosity",
     "Compute scattered radiosity in addition to direct radiosity",
     cxxopts::value<bool>()->default_value("false"))
    ("gs_steps", "Number of Gauss-Seidel steps used to compute scattered "
     "radiosity", cxxopts::value<int>()->default_value("1"))
    ("print_residual", "Print the residual at each step when computing the "
     "scattered radiosity", cxxopts::value<bool>()->default_value("false"))
    ("t,thermal", "Drive a thermal model",
     cxxopts::value<bool>()->default_value("false"))
    ("q,quiet", "Don't save output", 
     cxxopts::value<bool>()->default_value("false"))
    ("dt", "Time step for thermal model [s]",
     cxxopts::value<double>()->default_value("600"))
    ("a,albedo", "Albedo for model",
     cxxopts::value<double>()->default_value("0.12"))
    ("ti", "Thermal inertia", cxxopts::value<double>()->default_value("70.0"))
    ("T0", "Initial temperate", cxxopts::value<double>()->default_value("233.0"))
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
  params.do_radiosity = args["radiosity"].as<bool>();
  params.gs_steps = args["gs_steps"].as<int>();
  params.print_residual = args["print_residual"].as<bool>();
  params.do_thermal = args["thermal"].as<bool>();
  params.quiet = args["quiet"].as<bool>();
  params.dt = args["dt"].as<double>();
  params.albedo = args["albedo"].as<double>();
  params.thermal_inertia = args["ti"].as<double>();
  params.initial_temperature = args["T0"].as<double>();
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

  if (task == "horizons") do_horizons_task(params, context);
  else if (task == "radiosity") do_radiosity_task(params, context);

#if USE_MPI
  MPI_Finalize();
#endif
}
