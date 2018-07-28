#include <armadillo>
#include <cxxopts.hpp>
#include <fastbvh>
#include <png++/png.hpp>

#include <algorithm>
#include <stdexcept>

#include <boost/optional.hpp>
template <class T> using opt_t = boost::optional<T>;

#if USE_TBB
#  include <tbb/tbb.h>
#endif

// #include "cet.h"
#include "obj_util.hpp"

constexpr double PI = 3.141592653589793;

enum class mode_e {ORTHO};

mode_e string_to_mode(std::string const & mode_string) {
  if (mode_string == "ortho") return mode_e::ORTHO;
  else throw std::invalid_argument("invalid mode: " + mode_string);
}

int main(int argc, char * argv[])
{
  using namespace cxxopts;

  Options options(argv[0], "TODO");
  
  options.add_options()
    ("h,help", "Display usage")
    ("obj_path", "OBJ file path", value<std::string>())
    ("d,data_path", "Path of data to render", value<std::string>())
    ("i,img_path", "Output file path",
     value<std::string>()->default_value("img.png"))
    ("shape_index", "OBJ file shape index", value<int>()->default_value("0"))
    ("W,width", "Image width [px]", value<int>()->default_value("512"))
    ("H,height", "Image height [px]", value<int>()->default_value("512"))
    ("az_cam", "Camera azimuth [deg]", value<double>()->default_value("0"))
    ("el_cam", "Camera elevation [deg]", value<double>()->default_value("0"))
    ("r_cam", "Distance of camera from model centroid [km]",
     value<double>()->default_value("0"))
    ("mode", "Rendering mode", value<std::string>()->default_value("ortho"))
    ;

  options.parse_positional({"obj_path"});

  auto args = options.parse(argc, argv);
  if (args["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  std::string obj_path = args["obj_path"].as<std::string>();

  opt_t<std::string> data_path;
  if (args["data_path"].count() > 0) {
    data_path = args["data_path"].as<std::string>();
  }

  std::string img_path = args["img_path"].as<std::string>();

  int shape_index = args["shape_index"].as<int>();
  int width = args["width"].as<int>();
  int height = args["height"].as<int>();
  double az_cam = args["az_cam"].as<double>();
  double el_cam = args["el_cam"].as<double>();
  double r_cam = args["r_cam"].as<double>();
  mode_e mode = string_to_mode(args["mode"].as<std::string>());
  
  if (args["r_cam"].count() > 0) {
    if (mode == mode_e::ORTHO) {
      std::cerr << "setting r_cam while mode is ortho is invalid" << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      r_cam = args["r_cam"].as<double>();
    }
  }

  arma::vec data;
  data.load(*data_path);
  
  auto objects = obj_util::get_objects(obj_path.c_str(), shape_index);

  Vector3 c_model {0., 0., 0.};
  for (auto obj: objects) {
    c_model = c_model + static_cast<Tri *>(obj)->getCentroid();
  }
  c_model = c_model/objects.size();

  double r_model = 0;
  for (auto obj: objects) {
    double r = length(static_cast<Tri *>(obj)->getCentroid() - c_model);
    r_model = fmax(r_model, r);
  }

  if (mode == mode_e::ORTHO) {
    r_cam = 1.1*r_model;
  }

  BVH bvh {&objects};

  az_cam *= 180*PI;
  el_cam *= 180*PI;

  if (r_cam == 0) {
    std::cerr << "r_cam is uninitialized somehow" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Vector3 p_cam {
    static_cast<float>(r_cam*std::cos(az_cam)*std::sin(el_cam)),
    static_cast<float>(r_cam*std::sin(az_cam)*std::sin(el_cam)),
    static_cast<float>(r_cam*std::cos(el_cam))
  };

  Vector3 n_eye = -normalize(p_cam);

  p_cam = p_cam + c_model;
  
  double el_up = el_cam - 90;
  Vector3 n_up {
    static_cast<float>(std::cos(az_cam)*std::sin(el_up)),
    static_cast<float>(std::sin(az_cam)*std::sin(el_up)),
    static_cast<float>(std::cos(el_up))
  };

  Vector3 n_right = n_eye^n_up;

  arma::mat img_data(width, height);
  
  if (mode == mode_e::ORTHO) {
    double aspect = static_cast<double>(width)/height;

    std::vector<std::pair<int, int>> misses;

    IntersectionInfo info;
    for (int i = 0; i < height; ++i) {
      double y_eye = 2*(1. - static_cast<double>(i)/(height - 1)) - 1;
      y_eye *= r_model;
      for (int j = 0; j < width; ++j) {
        double x_eye = 2*static_cast<double>(j)/(width - 1) - 1;
        x_eye *= aspect*r_model;

        Vector3 p_ray = p_cam + y_eye*n_up + x_eye*n_right;
        Ray ray(p_ray, n_eye);
        if (!bvh.getIntersection(ray, &info, false)) {
          img_data(j, i) = 0;
        } else {
          int index = static_cast<Tri const *>(info.object)->index;
          img_data(j, i) = data(index);
        }
      }
    }

    img_data -= img_data.min();
    img_data /= img_data.max();

    // auto cmap = (uint8_t **) cet_CBD1;

    // auto lerp = [] (double x, uint8_t lo, uint8_t hi) -> uint8_t {
    //   return static_cast<uint8_t>(
    //     (1 - x)*static_cast<double>(lo) + x*static_cast<double>(hi));
    // };

    // auto mapcolor = [&lerp, &cmap] (double x) -> png::rgb_pixel {
    //   x = 255*std::clamp(x, 0., 1.);
    //   auto i = static_cast<uint8_t>(std::floor(x));
    //   auto frac = x - std::floor(x);
    //   return {
    //     lerp(frac, cmap[i][0], cmap[i + 1][0]),
    //     lerp(frac, cmap[i][1], cmap[i + 1][1]),
    //     lerp(frac, cmap[i][2], cmap[i + 1][2])
    //   };
    // };
    
    // auto jet = [] (double x) -> png::rgb_pixel {
    //   double r = std::clamp(x < 0.7 ? 4*x - 1.5 : -4*x + 4.5, 0., 1.);
    //   double g = std::clamp(x < 0.5 ? 4*x - 0.5 : -4*x + 3.5, 0., 1.);
    //   double b = std::clamp(x < 0.3 ? 4*x + 0.5 : -4*x + 2.5, 0., 1.);
    //   return {
    //     static_cast<uint8_t>(std::floor(256*r)),
    //     static_cast<uint8_t>(std::floor(256*g)),
    //     static_cast<uint8_t>(std::floor(256*b))
    //   };
    // };

    auto bw = [] (double x) -> png::rgb_pixel {
      auto byte = static_cast<uint8_t>(256*std::clamp(x, 0., 1.));
      return {byte, byte, byte};
    };

    png::image<png::rgb_pixel> img {
      static_cast<png::uint_32>(width),
      static_cast<png::uint_32>(height)
    };
    
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        img.set_pixel(j, i, bw(img_data(j, i)));
      }
    }

    img.write(img_path);
  }
}