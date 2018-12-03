#include "common.hpp"

#include <cxxopts.hpp>
#include <yaml-cpp/yaml.h>

#include "arma_util.hpp"
#include "constants.hpp"
#include "illum.hpp"
#include "kdtree.hpp"
#include "obj_util.hpp"
#include "thermal.hpp"
#include "timer.hpp"

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
  static job_params from_yaml(std::string const & path_str) {
    job_params params;

    YAML::Node config = YAML::LoadFile(path_str.c_str());

    if (config["task"]) {
      params.task = config["task"].as<std::string>();
    }

    if (config["path"]) {
      params.path = config["path"].as<std::string>();
    }

    if (config["fine"]) {
      params.fine = config["fine"].as<std::string>();
    }

    if (config["offset"]) {
      params.offset = config["offset"].as<double>();
    }

    if (config["theta_eps"]) {
      params.theta_eps = config["theta_eps"].as<double>();
    }

    if (config["nphi"]) {
      params.nphi = config["nphi"].as<double>();
    }

    if (config["radiosity"]) {
      params.radiosity = config["radiosity"].as<bool>();
    }

    if (config["gs_steps"]) {
      params.gs_steps = config["gs_steps"].as<int>();
    }

    if (config["print_residual"]) {
      params.print_residual = config["print_residual"].as<bool>();
    }

    if (config["thermal"]) {
      params.thermal = config["thermal"].as<bool>();
    }

    if (config["quiet"]) {
      params.quiet = config["quiet"].as<bool>();
    }

    if (config["dt"]) {
      params.dt = config["dt"].as<double>();
    }

    if (config["albedo"]) {
      params.albedo = config["albedo"].as<double>();
    }

    if (config["depths"]) {
      params.depths = config["depths"].as<std::vector<double>>();
    }

    if (config["ti"]) {
      params.thermal_inertia = config["ti"].as<double>();
    }

    if (config["rhoc"]) {
      if (config["rhoc"].IsSequence()) {
        params.rhoc = config["rhoc"].as<std::vector<double>>();
      } else {
        params.rhoc = config["rhoc"].as<double>();
      }
    }

    if (config["T0"]) {
      try {
        params.T0 = config["T0"].as<double>();
      } catch (...) {
        params.T0 = config["T0"].as<std::string>();
      }
    }

    if (config["output_dir"]) {
      params.output_dir = config["output_dir"].as<std::string>();
    }

    if (config["form_factors_file"]) {
      params.form_factors_file = config["form_factors_file"].as<std::string>();
    }

    if (config["horizon_file"]) {
      params.horizon_file = config["horizon_file"].as<std::string>();
    }

    if (config["horizon_obj_file"]) {
      params.horizon_obj_file = config["horizon_obj_file"].as<std::string>();
    }

    if (config["sun_pos_file"]) {
      params.sun_pos_file = config["sun_pos_file"].as<std::string>();
    }

    if (config["sun_unit"]) {
      params.sun_unit = config["sun_unit"].as<std::string>();
    }

    if (config["mesh_unit"]) {
      params.mesh_unit = config["mesh_unit"].as<std::string>();
    }
    
    if (config["save_full_thermal"]) {
      params.save_full_thermal = config["save_full_thermal"].as<bool>();
    }

    if (config["record_therm_stats"]) {
      params.record_therm_stats = config["record_therm_stats"].as<bool>();
    }

    if (config["repeat"]) {
      params.repeat = config["repeat"].as<int>();
    }

    if (config["print_config"]) {
      params.print_config = config["print_config"].as<bool>();
    }

    return params;
  }

  opt_t<std::string> task;
  opt_t<std::string> path;
  opt_t<std::string> fine;

  opt_t<double> offset;
  opt_t<double> theta_eps;
  opt_t<int> nphi;
  opt_t<int> gs_steps;

  opt_t<std::string> output_dir;
  opt_t<std::string> form_factors_file;
  opt_t<std::string> horizon_file;
  opt_t<std::string> sun_pos_file;
  opt_t<std::string> horizon_obj_file;
  opt_t<std::string> therm_stats_file;

  opt_t<bool> radiosity;
  opt_t<bool> print_residual;
  opt_t<bool> thermal;
  opt_t<bool> quiet;
  opt_t<bool> save_full_thermal;
  opt_t<bool> record_therm_stats;
  opt_t<bool> print_config;

  opt_t<int> repeat;

  opt_t<std::vector<double>> depths;
  opt_t<double> dt;
  opt_t<double> thermal_inertia;
  opt_t<var_t<double, std::vector<double>>> rhoc;

  opt_t<var_t<double, std::string>> albedo;
  opt_t<var_t<double, std::string>> T0;

  // TODO: should we use boost units for this eventually?
  opt_t<std::string> sun_unit;
  opt_t<std::string> mesh_unit;

  void set_albedo(std::string const & s) {
    try {
      albedo = std::stod(s);
    } catch (std::invalid_argument const & e) {
      file_exists_or_die(s);
      albedo = s;
    } catch (std::out_of_range const & e) {
      std::cerr << "albedo value " << s << " is out of range" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  void display() const {
    using std::cout;
    using std::endl;
    cout << "offset: " << *offset << endl
         << "theta_eps: " << *theta_eps << endl
         << "nphi: " << *nphi << endl
         << "gs_steps: " << *gs_steps << endl
         << "output_dir: " << (output_dir ? *output_dir : "N/A") << endl
         << "horizon_file: " << (horizon_file ? *horizon_file : "N/A") << endl
         << "sun_pos_file: " << (sun_pos_file ? *sun_pos_file : "N/A") << endl
         << "horizon_obj_file: " << (
           horizon_obj_file ? *horizon_obj_file : "N/A")
         << endl
         << "therm_stats_file: " << (
           therm_stats_file ? *therm_stats_file : "N/A")
         << endl
         << "radiosity: " << *radiosity << endl
         << "print_residual: " << *print_residual << endl
         << "thermal: " << *thermal << endl
         << "quiet: " << *quiet << endl
         << "save_full_thermal: " << *save_full_thermal << endl
         << "record_therm_stats: " << *record_therm_stats << endl
         << (repeat ?
             ("repeat: " + std::to_string(*repeat)).c_str() :
             "repeat: no")
         << endl
         << "depths: ";
    if (depths) {
      for (decltype(depths->size()) i = 0; i < depths->size() - 1; ++i)
        cout << (*depths)[i] << ", ";
      cout << (*depths)[depths->size() - 1];
    } else {
      cout << "N/A";
    }
    cout << endl
         << "dt: " << *dt << endl
         << "ti: " << *thermal_inertia << endl
         << "rhoc: ";
    if (rhoc) {
      if (auto r = boost::get<double>(&*rhoc)) {
        cout << *r;
      } else {
        auto rs = boost::get<std::vector<double>>(&*rhoc);
        for (decltype(rs->size()) i = 0; i < rs->size() - 1; ++i)
          cout << (*rs)[i] << ", ";
        cout << (*rs)[rs->size() - 1];
      }
    } else {
      cout << "N/A";
    }
    cout << endl
         << "albedo: " << *albedo << endl
         << "T0: " << *T0 << endl
         << "sun_unit: " << *sun_unit << endl
         << "mesh_unit: " << *mesh_unit << endl;
  }
};

void do_form_factors_task(job_params const & params, illum_context & context)
{
  arma::sp_mat F;

  timed("- assembling form factor matrix", [&] () {
    F = context.compute_F(*params.albedo);
  });

  boost::filesystem::path output_dir_path = params.output_dir ?
    *params.output_dir : ".";

  timed("- saving form factor matrix", [&] () {
    arma_util::save_mat(F, output_dir_path/"form_factors");
  });
}

void do_horizons_task(job_params const & params, illum_context & context)
{
  timed("- building horizon map", [&] () {
    context.make_horizons(*params.nphi, *params.theta_eps, *params.offset);
  });

  boost::filesystem::path output_path {"horizons"};

  if (params.output_dir) {
    output_path = *params.output_dir/output_path;
  }

  if (!*params.quiet) {
    timed("- saving horizon map to " + output_path.string(), [&] () {
      context.save_horizons(output_path);
    });
  }
}

void do_radiosity_task(job_params const & params, illum_context & context)
{
  int nfaces = context.get_num_faces();
  opt_t<int> nhoriz;

  /*
   * Load or create a matrix containing the horizons.
   */
  if (params.horizon_file) {
    file_exists_or_die(*params.horizon_file);
    timed("- loading horizon map from " + *params.horizon_file, [&] () {
      context.load_horizons(*params.horizon_file);
    });
  } else {
    timed("- building horizon map", [&] () {
      context.make_horizons(*params.nphi, *params.theta_eps, *params.offset);
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
  if (*params.sun_unit == "m") {
    sun_positions /= 1000.;
  }

  arma::mat disk_xy;
  fib_spiral(disk_xy, 100);

  int nsunpos = sun_positions.n_cols;

  arma::vec rad(nfaces), rad_avg(nfaces, arma::fill::zeros);

  arma::vec therm, therm_avg, therm_max;

  if (params.thermal && *params.thermal) {
    therm.resize(nfaces);

    therm_avg.resize(nfaces);
    therm_avg.zeros();

    therm_max.resize(nfaces);
    therm_max.zeros();
  }

  opt_t<thermal_model> therm_model;

  if (params.thermal && *params.thermal) {
    therm_model = thermal_model {
      nfaces,
      *params.depths,
      *params.thermal_inertia,
      *params.rhoc,
      *params.T0
    };
  }

  if (*params.record_therm_stats) {
    therm_model->record_stats = true;
  }

  arma::sp_mat F;

  if (*params.radiosity) {
    if (params.form_factors_file) {
      timed("- loading form factor matrix", [&] () {
        F = arma_util::load_mat<arma::sp_mat>(*params.form_factors_file);
        std::cout << " [nnz = " << F.n_nonzero << "]";
      });
    } else {
      timed("- assembling form factor matrix", [&] () {
        F = context.compute_F(*params.albedo);
        std::cout << " [nnz = " << F.n_nonzero << "]";
      });
    }

    assert(F.n_rows == static_cast<arma::uword>(nfaces));
    assert(F.n_cols == static_cast<arma::uword>(nfaces));
  }

  int current_frame = 0;

  auto const step = [&] (int j, int rep = 0) {
    std::string frame_str =
      std::to_string(j + 1) + "/" + std::to_string(nsunpos);
    if (rep > 0) {
      frame_str += " [" + std::to_string(rep) + "/" +
        std::to_string(*params.repeat) + "]";
    }

    rad.zeros();

    /**
     * Incorporate heat flux at surface into radiosity.
     */
    if (*params.thermal) {
      rad += therm_model->get_radiosity();
    }

    timed("- " + frame_str + ": computing direct radiosity", [&] () {
      rad += context.get_direct_radiosity(sun_positions.col(j), disk_xy);
      std::cout << " [mean: " << arma::mean(rad) << " W/m^2]";
    });

    if (*params.radiosity) {
      arma::sp_mat L_t = (arma::speye(nfaces, nfaces) - arma::trimatl(F)).t();
      arma::sp_mat U = -arma::trimatu(F);

      arma::vec E = rad;
      rad = arma_util::forward_solve(L_t, E);
      for (int iter = 0; iter < *params.gs_steps; ++iter) {
        timed("  + doing Gauss-Seidel step", [&] () {
          rad = arma_util::forward_solve(L_t, E - U*rad);
          if (*params.print_residual) {
            double res = arma::norm((rad.t()*L_t).t() + U*rad - E);
            std::cout << " [res = " << res << "]";
          }
        });
      };
    }

    rad_avg += (rad - rad_avg)/(current_frame + 1);

    if (*params.thermal) {
      timed("- " + frame_str + ": stepping thermal model", [&] () {
        therm_model->step(*params.dt, rad);
        therm = therm_model->T.row(0).t();
        therm_avg += (therm - therm_avg)/(current_frame + 1);

        // TODO: add max_lag flag to set 50 to something else
        if (current_frame > 50) {
          therm_max = arma::max(therm_max, therm);
        }
      });
    }
  };

  if (params.repeat) {
    for (int rep = 1; rep <= *params.repeat; ++rep) {
      for (int j = 0; j < nsunpos; ++j) {
        step(j, rep);
      }
    }
  } else {
    for (int j = 0; j < nsunpos; ++j) {
      step(j);
    }
  }

  boost::filesystem::path output_dir_path = params.output_dir ?
    *params.output_dir : ".";

  if (!*params.quiet) {
    timed("- saving radiosity", [&] () {
      arma_util::save_mat(rad, output_dir_path/"rad");
    });

    timed("- saving average radiosity", [&] () {
      arma_util::save_mat(rad_avg, output_dir_path/"rad_avg");
    });

    if (*params.thermal) {
      if (*params.save_full_thermal) {
        timed("- saving full thermal model", [&] () {
          arma_util::save_mat(
            therm_model->T, output_dir_path/"therm");
        });
      } else {
        timed("- saving thermal", [&] () {
          arma_util::save_mat(therm, output_dir_path/"therm");
        });
      }

      timed("- saving average thermal", [&] () {
        arma_util::save_mat(therm_avg, output_dir_path/"therm_avg");
      });

      timed("- saving max thermal", [&] () {
        arma_util::save_mat(therm_max, output_dir_path/"therm_max");
      });
    }

    if (*params.record_therm_stats) {
      timed("- saving thermal stats", [&] () {
        auto path = (output_dir_path/"therm_stats.bin").string();
        therm_model->get_stats_matrix().save(path);
      });
    }
  }
}

void do_residuals_task(job_params const & params, illum_context & context)
{
  /**
   * TODO: this is just a stub to make it easy to remember how to get
   * going here
   */

  (void) params;
  
  points_t centroids;
  centroids.reserve(context.get_num_faces());
  for (Object * object: context.objects) {
    centroids.push_back(static_cast<Tri *>(object)->getCentroid());
  }

  kdtree_t kdtree {centroids};

  auto p = centroids[0];
  size_t k = 10;
  size_t I[10];
  num_t D_sq[10];
  size_t valid = kdtree.query(p, k, I, D_sq);

  for (size_t i = 0; i < valid; ++i) {
    std::cout << "i = " << I[i] << ", "
              << "p = " << centroids[I[i]] << ", "
              << "dist_sq = " << D_sq[i] << std::endl;
  }
}

int main(int argc, char * argv[]) {
  std::set<std::string> tasks = {
    "form_factors",
    "horizons",
    "radiosity",
    "residuals"
  };

  auto tasks_to_string = [&] () {
    std::string s;
    for (auto task: tasks) s += "  " + task + '\n';
    return s;
  };

  cxxopts::Options options(
	boost::filesystem::path {std::string(argv[0])}.filename().c_str(),
    // argv[0],
    ("Available tasks:\n\n" + tasks_to_string()).c_str());

  options.add_options()
    (
      "h,help",
      "Display usage"
    )
    (
      "task",
      "Task to do",
      cxxopts::value<std::string>()->default_value("radiosity")
    )
    (
      "config",
      "Path to YAML configuration file",
      cxxopts::value<std::string>()
    )
    (
      "path",
      "Path to input OBJ file (also used coarse mesh for `residuals' task)",
      cxxopts::value<std::string>()
    )
    (
      "fine",
      "Path to fine mesh for `residuals' task",
      cxxopts::value<std::string>()
    )
    (
      "offset",
      "Offset when calculating visibility from a triangle",
      cxxopts::value<double>()->default_value("1e-5")
    )
    (
      "theta_eps",
      "Tolerance for theta in calculating horizons",
      cxxopts::value<double>()->default_value("1e-3")
    )
    (
      "nphi",
      "Number of phi values (linearly spaced in [0, 2pi])",
      cxxopts::value<int>()->default_value("361")
    )
    (
      "radiosity",
      "Compute scattered radiosity in addition to direct radiosity",
      cxxopts::value<bool>()->default_value("false")
    )
    (
      "gs_steps",
      "Number of Gauss-Seidel steps used to compute scattered radiosity",
      cxxopts::value<int>()->default_value("1")
    )
    (
      "print_residual",
      "Print the residual at each step when computing the scattered radiosity",
      cxxopts::value<bool>()->default_value("false")
    )
    (
      "thermal",
      "Drive a thermal model",
      cxxopts::value<bool>()->default_value("false")
    )
    (
      "quiet",
      "Don't save output",
      cxxopts::value<bool>()->default_value("false")
    )
    (
      "dt",
      "Time step for thermal model [s]",
      cxxopts::value<double>()->default_value("600")
    )
    (
      "albedo",
      "Albedo for model",
      cxxopts::value<std::string>()->default_value("0.12")
    )
    (
      "depths",
      "Depths for layers in thermal models [m]",
      cxxopts::value<std::vector<double>>()
    )
    (
      "ti",
      "Thermal inertia [J m^-2 K^-1 s^-1/2]",
      cxxopts::value<double>()->default_value("70.0")
    )
    (
      "rhoc",
      "Heat capacity per volume [J m^-3]",
      cxxopts::value<std::vector<double>>()
    )
    (
      "T0",
      "Initial temperature",
      cxxopts::value<std::string>()->default_value("233.0")
    )
    (
      "output_dir",
      "Output file",
      cxxopts::value<std::string>()
    )
    (
      "form_factors_file",
      "File containing form factor matrix",
      cxxopts::value<std::string>()
    )
    (
      "horizon_file",
      "Horizon file",
      cxxopts::value<std::string>()
    )
    (
      "horizon_obj_file",
      "Path of OBJ file to use to compute horizon maps",
      cxxopts::value<std::string>()
    )
    (
      "sun_pos_file",
      "File containing sun positions",
      cxxopts::value<std::string>()
    )
    (
      "sun_unit",
      "Units used for sun positions",
      cxxopts::value<std::string>()->default_value("m")
    )
    (
      "mesh_unit",
      "Units used by OBJ file vertices",
      cxxopts::value<std::string>()->default_value("km")
    )
    (
      "save_full_thermal",
      "Save all layers of the thermal model as opposed to only the top layer",
      cxxopts::value<bool>()->default_value("false")
    )
    (
      "record_therm_stats",
      "Collect thermal statistics",
      cxxopts::value<bool>()->default_value("false")
    )
    (
      "repeat",
      "Number of times to repeat the sun positions",
      cxxopts::value<int>()
    )
    (
      "print_config",
      "Print configuration before running",
      cxxopts::value<bool>()->default_value("false")
    )
    ;

  auto args = options.parse(argc, argv);
  if (args["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  job_params params;

  /**
   * If a path to a configuration file is passed, we initialize params
   * from this file. Any other options that are passed overwrite these
   * parameters!
   *
   * TODO: in the future, we can probably support more than just YAML
   * configuration files.
   */
  if (args.count("config") != 0) {
    params = job_params::from_yaml(args["config"].as<std::string>());
  }

  /**
   * Parse options which are job parameters.
   */

  auto const show_help_and_die = [&] () {
    std::cout << options.help() << std::endl;
    std::exit(EXIT_FAILURE);
  };

  if (!params.task || args.count("task")) {
    params.task = args["task"].as<std::string>();
  }

  if (tasks.find(*params.task) == tasks.end()) {
    show_help_and_die();
  }

  if (!params.path) {
    if (!args.count("path")) {
      show_help_and_die();
    } else {
      params.path = args["path"].as<std::string>();
    }
  }

  if (!params.fine && args.count("fine")) {
    params.fine = args["fine"].as<std::string>();
  }

  if (!params.offset || args.count("offset")) {
    params.offset = args["offset"].as<double>();
  }

  if (!params.theta_eps || args.count("theta_eps")) {
    params.theta_eps = args["theta_eps"].as<double>();
  }

  if (!params.nphi || args.count("nphi")) {
    params.nphi = args["nphi"].as<int>();
  }

  if (!params.radiosity || args.count("radiosity")) {
    params.radiosity = args["radiosity"].as<bool>();
  }

  if (!params.gs_steps || args.count("gs_steps")) {
    params.gs_steps = args["gs_steps"].as<int>();
  }

  if (!params.print_residual || args.count("print_residual")) {
    params.print_residual = args["print_residual"].as<bool>();
  }

  if (!params.thermal || args.count("thermal")) {
    params.thermal = args["thermal"].as<bool>();
  }

  if (!params.quiet || args.count("quiet")) {
    params.quiet = args["quiet"].as<bool>();
  }

  if (!params.depths && args.count("depths")) {
    params.depths = args["depths"].as<std::vector<double>>();
  }

  if (!params.dt || args.count("dt")) {
    params.dt = args["dt"].as<double>();
  }

  if (!params.albedo || args.count("albedo")) {
    params.set_albedo(args["albedo"].as<std::string>());
  }

  if (!params.thermal_inertia || args.count("thermal_inertia")) {
    params.thermal_inertia = args["ti"].as<double>();
  }

  if (!params.rhoc && args.count("rhoc")) {
    auto rhoc = args["rhoc"].as<std::vector<double>>();
    if (rhoc.size() == 1) {
      params.rhoc = rhoc[0];
    } else {
      params.rhoc = rhoc;
    }
  }

  try {
    params.T0 = std::stod(args["T0"].as<std::string>());
  } catch (...) {
    params.T0 = args["T0"].as<std::string>();
  }

  if (!params.output_dir && args.count("output_dir")) {
    params.output_dir = args["output_dir"].as<std::string>();
  }

  if (!params.form_factors_file && args.count("form_factors_file")) {
    params.form_factors_file = args["form_factors_file"].as<std::string>();
  }

  if (!params.horizon_file && args.count("horizon_file")) {
    params.horizon_file = args["horizon_file"].as<std::string>();
  }

  if (!params.horizon_obj_file && args.count("horizon_obj_file")) {
    params.horizon_obj_file = args["horizon_obj_file"].as<std::string>();
  }

  if (!params.sun_pos_file && args.count("sun_pos_file")) {
    params.sun_pos_file = args["sun_pos_file"].as<std::string>();
  }

  if (!params.sun_unit || args.count("sun_unit")) {
    params.sun_unit = args["sun_unit"].as<std::string>();
  }

  if (!params.mesh_unit || args.count("mesh_unit")) {
    params.mesh_unit = args["mesh_unit"].as<std::string>();
  }

  if (!params.save_full_thermal || args.count("save_full_thermal")) {
    params.save_full_thermal = args["save_full_thermal"].as<bool>();
  }

  if (!params.record_therm_stats || args.count("record_therm_stats")) {
    params.record_therm_stats = args["record_therm_stats"].as<bool>();
  }

  if (!params.print_config || args.count("print_config")) {
    params.print_config = args["print_config"].as<bool>();
  }
  
  if (!params.repeat && args.count("repeat")) {
    params.repeat = args["repeat"].as<int>();
  }

  assert(*params.sun_unit == "m" || *params.sun_unit == "km");
  assert(*params.mesh_unit == "m" || *params.mesh_unit == "km");

  if (*params.mesh_unit == "m") {
    std::cout << "TODO: mesh_unit == m isn't implemented yet" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  /**
   * Error and consistency checking
   */
  if (params.thermal && *params.thermal) {
    if (!params.depths) {
      std::cerr << "When running a thermal model provide layer levels using "
                << "the --depths flag" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    if (!params.rhoc) {
      std::cerr << "When running a thermal model provide volumetric heat "
                << "capacity values using the --rhoc flag" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  if (*params.print_config) {
    params.display();
  }
  
  illum_context context {*params.path};
  if (params.horizon_obj_file) {
    context.set_horizon_obj_file(*params.horizon_obj_file);
  }

  if (*params.task == "form_factors") {
    do_form_factors_task(params, context);
  } else if (*params.task == "horizons") {
    do_horizons_task(params, context);
  } else if (*params.task == "radiosity") {
    do_radiosity_task(params, context);
  } else if (*params.task == "residuals") {
    do_residuals_task(params, context);
  }
}
