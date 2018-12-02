#include <armadillo>
#include <cxxopts.hpp>
#include <fastbvh>

#include "png++/png.hpp"

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

enum class mode_e {
  EQUAL_AREA_CYLINDRICAL,
  ORTHOGRAPHIC
};

mode_e string_to_mode(std::string const & mode_string) {
  if (mode_string == "orthographic") {
    return mode_e::ORTHOGRAPHIC;
  } else if (mode_string == "equal_area_cylindrical") {
    return mode_e::EQUAL_AREA_CYLINDRICAL;
  } else {
    throw std::invalid_argument("invalid mode: " + mode_string);
  }
}

int main(int argc, char * argv[])
{
  using namespace cxxopts;

  Options options(argv[0], "TODO");
  
  options.add_options()
    (
      "h,help",
      "Display usage"
    )
    (
      "path",
      "path to OBJ file",
      value<std::string>()
    )
    (
      "data",
      "path of data to render",
      value<std::string>()
    )
    (
      "img",
      "filename of saved PNG image",
      value<std::string>()->default_value("img.png")
    )
    (
      "shape",
      "shape index of mesh in OBJ file",
      value<int>()->default_value("0")
    )
    (
      "L,layer",
      "layer (column) of input data to plot",
      value<arma::uword>()->default_value("0")
    )
    (
      "W,width",
      "image width [px]",
      value<int>()->default_value("512")
    )
    (
      "H,height",
      "image height [px]",
      value<int>()->default_value("512")
    )
    (
      "m,minval",
      "minimum value for plot range",
      value<double>()
    )
    (
      "M,maxval",
      "maximum value for plot range",
      value<double>()
    )
    (
      "az",
      "Camera azimuth [deg]",
      value<double>()->default_value("0")
    )
    (
      "el",
      "camera elevation [deg]",
      value<double>()->default_value("0")
    )
    (
      "rad",
      "distance of camera from model centroid [km]",
      value<double>()->default_value("0")
    )
    (
      "mode",
      "rendering mode or map projection",
      value<std::string>()->default_value("orthographic")
    )
    (
      "cmap",
      "color map to use to render",
      value<std::string>()->default_value("grayscale")
    )
    (
      "transpose",
      "transpose data first (use to plot rows instead of columns)",
      value<bool>()->default_value("false")
    )
    ;

  auto args = options.parse(argc, argv);
  if (argc == 1 || args["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  std::string path = args["path"].as<std::string>();

  opt_t<std::string> data_str;
  if (args["data"].count() > 0) {
    data_str = args["data"].as<std::string>();
  }

  std::string img_str = args["img"].as<std::string>();

  int shape = args["shape"].as<int>();
  arma::uword layer = args["layer"].as<arma::uword>();
  int width = args["width"].as<int>();
  int height = args["height"].as<int>();
  double az = args["az"].as<double>();
  double el = args["el"].as<double>();
  double rad = args["rad"].as<double>();
  mode_e mode = string_to_mode(args["mode"].as<std::string>());
  std::string cmap_str = args["cmap"].as<std::string>();
  
  opt_t<double> minval;
  if (args["minval"].count() > 0) {
    minval = args["minval"].as<double>();
  }

  opt_t<double> maxval;
  if (args["maxval"].count() > 0) {
    maxval = args["maxval"].as<double>();
  }

  if (args["rad"].count() > 0) {
    if (mode == mode_e::ORTHOGRAPHIC) {
      std::cerr << "can't set radius in orthographic mode" << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      rad = args["rad"].as<double>();
    }
  }

  arma::mat data_matrix;
  data_matrix.load(*data_str);
  if (args["transpose"].as<bool>()) {
    data_matrix = data_matrix.t();
  }

  arma::vec data = data_matrix.col(layer);
  if (minval) {
    for (arma::uword i = 0; i < data.n_elem; ++i) {
      data(i) = std::max(*minval, data(i));
    }
  }
  if (maxval) {
    for (arma::uword i = 0; i < data.n_elem; ++i) {
      data(i) = std::min(*maxval, data(i));
    }
  }
  
  auto objects = obj_util::get_objects(path.c_str(), shape);

  if (data.n_elem != objects.size()) {
    throw std::runtime_error {"input data and mesh have incompatible sizes"};
  }

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

  if (mode == mode_e::ORTHOGRAPHIC) {
    rad = 1.1*r_model;
  }

  BVH bvh {&objects};

  az *= PI/180;
  el *= PI/180;

  if (rad == 0) {
    std::cerr << "rad is uninitialized somehow" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Vector3 p_cam {
    static_cast<float>(rad*std::cos(az)*std::sin(el)),
    static_cast<float>(rad*std::sin(az)*std::sin(el)),
    static_cast<float>(rad*std::cos(el))
  };

  Vector3 n_eye = -normalize(p_cam);

  p_cam = p_cam + c_model;
  
  double el_up = el - 90;
  Vector3 n_up {
    static_cast<float>(std::cos(az)*std::sin(el_up)),
    static_cast<float>(std::sin(az)*std::sin(el_up)),
    static_cast<float>(std::cos(el_up))
  };

  Vector3 n_right = n_eye^n_up;

  if (mode == mode_e::ORTHOGRAPHIC) {
    struct elt {
      elt(int i, int j, double value): i {i}, j {j}, value {value} {}
      int i, j;
      double value;
    };

    std::vector<elt> elts;

    double aspect = static_cast<double>(width)/height;

    IntersectionInfo info;
    for (int i = 0; i < height; ++i) {
      double y_eye = 2*(1. - static_cast<double>(i)/(height - 1)) - 1;
      y_eye *= rad;
      for (int j = 0; j < width; ++j) {
        double x_eye = 2*static_cast<double>(j)/(width - 1) - 1;
        x_eye *= aspect*rad;

        Vector3 p_ray = p_cam + y_eye*n_up + x_eye*n_right;
        Ray ray(p_ray, n_eye);
        if (bvh.getIntersection(ray, &info, false)) {
          int index = static_cast<Tri const *>(info.object)->index;
          elts.emplace_back(i, j, data(index));
        }
      }
    }

    double lo = minval ? *minval : data.min();
    double hi = maxval ? *maxval : data.max();

    for (auto & elt: elts) {
      elt.value -= lo;
      elt.value /= hi - lo;
    }
    
    auto jet = [] (double x) -> png::rgb_pixel {
      double r = std::clamp(x < 0.7 ? 4*x - 1.5 : -4*x + 4.5, 0., 1.);
      double g = std::clamp(x < 0.5 ? 4*x - 0.5 : -4*x + 3.5, 0., 1.);
      double b = std::clamp(x < 0.3 ? 4*x + 0.5 : -4*x + 2.5, 0., 1.);
      return {
        static_cast<uint8_t>(std::floor(255*r)),
        static_cast<uint8_t>(std::floor(255*g)),
        static_cast<uint8_t>(std::floor(255*b))
      };
    };

    auto bw = [] (double x) -> png::rgb_pixel {
      auto byte = static_cast<uint8_t>(255*std::clamp(x, 0., 1.));
      return {byte, byte, byte};
    };

    auto cmap = cmap_str == "grayscale" ? bw : jet;

    png::image<png::rgb_pixel> img {
      static_cast<png::uint_32>(width),
      static_cast<png::uint_32>(height)
    };
    
    for (auto & elt: elts) {
      img.set_pixel(elt.j, elt.i, cmap(elt.value));
    }

    img.write(img_str);
  }
}
