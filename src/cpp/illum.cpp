#include "illum.hpp"

#include <cassert>

#include <config.hpp>

#include "arma_util.hpp"
#include "sp_inds.hpp"

#include <fastbvh>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#if USE_TBB
#  include <tbb/tbb.h>
#endif

#include <algorithm>
#include <unordered_map>
#include <vector>

std::vector<Object *>
get_objects(
  const char * path,
  int shape_index)
{
  /**
   * Use tinyobjloader to load selected obj file.
   */
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path);
  if (!err.empty()) {
    std::cerr << err << std::endl;
  }
  if (!ret) {
    std::cerr << "Failed to load " << path << std::endl;
    std::exit(EXIT_FAILURE);
  }

  /**
   * Build a vector of objects for use with our bounding volume
   * hierarchy library (fastbvh).
   */
  tinyobj::shape_t & shape = shapes[shape_index];

  std::vector<Object *> objects;

  auto const & mesh = shape.mesh;
  auto const & indices = mesh.indices;
  auto const & vertices = attrib.vertices;
  auto const & normals = attrib.normals;

  for (size_t i = 0; i < indices.size(); i += 3) {
    tinyobj::index_t i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];

    auto const v0 = Vector3 {
      vertices[3*i0.vertex_index],
      vertices[3*i0.vertex_index + 1],
      vertices[3*i0.vertex_index + 2]
    };

    auto const v1 = Vector3 {
      vertices[3*i1.vertex_index],
      vertices[3*i1.vertex_index + 1],
      vertices[3*i1.vertex_index + 2]
    };

    auto const v2 = Vector3 {
      vertices[3*i2.vertex_index],
      vertices[3*i2.vertex_index + 1],
      vertices[3*i2.vertex_index + 2]
    };

    // If the face normal index isn't set, then we can compute the
    // face normal by taking the cross product of the triangle
    // edges. To ensure a consistent orientation, we can take the dot
    // product between the candidate face normal and one of the vertex
    // normals.

    Vector3 n = normalize((v1 - v0)^(v2 - v0));
    Vector3 n0 {
      normals[3*i0.vertex_index],
      normals[3*i0.vertex_index + 1],
      normals[3*i0.vertex_index + 2]
    };
    if (n*n0 < 0) {
      n = -n;
    }

    // auto const n = normalize(Vector3 {
    //   N[3*i0.vertex_index] + N[3*i1.vertex_index] + N[3*i2.vertex_index],
    //   N[3*i0.vertex_index + 1] + N[3*i1.vertex_index + 1] + N[3*i2.vertex_index + 1],
    //   N[3*i0.vertex_index + 2] + N[3*i1.vertex_index + 2] + N[3*i2.vertex_index + 2]
    // });

    // objects.push_back(new Tri(v0, v1, v2, n, i/3));
    objects.push_back(new Tri(v0/100., v1/100., v2/100., n, i/3));
  }

  return objects;
}

struct illum_context::impl
{
  impl(std::vector<Object *> objects):
    objects {objects},
    bvh_objects(objects.begin(), objects.end()),
    bvh {&bvh_objects},
    num_faces {objects.size()}
  {}

  arma::sp_mat compute_F(double offset = 1e-5);

  void make_horizons(
    int nphi = 361,
    double theta_eps = 1e-3,
    double offset = 1e-5,
    opt_t<int> j0 = opt_t <int> {},
    opt_t<int> j1 = opt_t <int> {});

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
    double sun_radius,
    opt_t<int> const & j0,
    opt_t<int> const & j1);

  int get_num_faces() const {
    return num_faces;
  }

  std::vector<Object *> objects;
  std::vector<Object *> bvh_objects;
  BVH bvh;
  size_t num_faces;
  arma::mat horizons;
};

illum_context::illum_context(char const * path, int shape_index):
  pimpl {new illum_context::impl {get_objects(path, shape_index)}}
{}

illum_context::~illum_context() = default;

arma::sp_mat
illum_context::compute_F(double offset)
{
  return pimpl->compute_F(offset);
}

void
illum_context::make_horizons(
  int nphi,
  double theta_eps,
  double offset,
  opt_t<int> j0,
  opt_t<int> j1)
{
  pimpl->make_horizons(nphi, theta_eps, offset, j0, j1);
}

void
illum_context::save_horizons(
  boost::filesystem::path const & path,
  opt_t<int> i0,
  opt_t<int> i1) const
{
  pimpl->save_horizons(path, i0, i1);
}

void
illum_context::load_horizons(
  boost::filesystem::path const & path,
  opt_t<int> i0,
  opt_t<int> i1)
{
  pimpl->load_horizons(path, i0, i1);
}

arma::vec
illum_context::get_direct_radiosity(
  arma::vec const & sun_position,
  arma::mat const & disk_xy,
  double sun_radius,
  opt_t<int> j0,
  opt_t<int> j1)
{
  return pimpl->get_direct_radiosity(
    sun_position,
    disk_xy,
    sun_radius,
    j0,
    j1);
}

int
illum_context::get_num_faces() const {
  return pimpl->get_num_faces();
}

#if USE_TBB
struct join_horiz_reducer
{
  sp_inds<arma::uword> inds;

  join_horiz_reducer() {}
  join_horiz_reducer(join_horiz_reducer &, tbb::split) {}

  using blocked_range_type = tbb::blocked_range<
    typename std::vector<sp_inds<arma::uword>>::iterator>;

  void operator()(blocked_range_type const & r) {
    for (auto it = r.begin(); it != r.end(); ++it) {
      inds.append(*it);
    }
  }

  void join(join_horiz_reducer const & other) {
    inds.append(other.inds);
  }
};
#endif

arma::sp_mat
illum_context::impl::compute_F(double offset) {
  IntersectionInfo info;

  std::vector<arma::uword> inds;
  std::vector<double> values;

  for (arma::uword j = 0; j < num_faces; ++j) {
    auto tri_j = static_cast<Tri const *>(objects[j]);
    auto p_j = tri_j->getCentroid();
    auto n_j = tri_j->getNormal(info);
    auto q_j = p_j + offset*n_j; // offset centroid = ray origin

    std::vector<arma::uword> hits;

    // First, determine which faces are visible from the jth face

    for (arma::uword i = 0; i < num_faces; ++i) {
      if (i == j) {
        continue;
      }

      auto tri_i = static_cast<Tri const *>(objects[i]);
      auto p_i = tri_i->getCentroid();
      auto r_i = tri_i->getBoundingRadius();

      // Check if triangles are approximately "facing" each other
      if ((p_i - p_j)*n_j <= -r_i) {
        continue;
      }

      // Shoot a ray between the faces if they are (something *must*
      // be hit)
      Ray ray(q_j, normalize(p_i - q_j));
      if (!bvh.getIntersection(ray, &info, false)) {
        // TODO: should probably emit a warning here!
        assert(info.object == nullptr);
        continue;
      }

      // Get the index of the triangle that was hit by the ray
      arma::uword hit_index = static_cast<Tri const *>(info.object)->index;
      if (hit_index == j) {
        continue;
      }

      // Search for the triangle that was hit and insert it (in sorted
      // order) if it's a new visible triangle
      auto lb = std::lower_bound(hits.begin(), hits.end(), hit_index);
      if (lb == hits.end()) {
        hits.insert(lb, hit_index);
      } else {
        if (*lb != hit_index) {
          hits.insert(lb, hit_index);
        }
      }
    }

    // Now, using the visibility information, compute the form factors
    // to the jth face

    auto T_j = static_cast<Tri const *>(objects[j]);
    double A_j = length((T_j->v1 - T_j->v0)^(T_j->v2 - T_j->v0))/2;

    for (auto i: hits) {
      auto T_i = static_cast<Tri const *>(objects[i]);
      auto p_i = T_i->getCentroid();
      auto n_i = T_i->getNormal(info);

      auto p_ij = p_j - p_i;
      double r_ij = length(p_ij);
      auto n_ij = p_ij/r_ij;

      double mu_ij = fmax(0, n_i*n_ij);
      double mu_ji = fmax(0, -n_j*n_ij);

      double F_ij = mu_ij*mu_ji*A_j/(arma::datum::pi*r_ij*r_ij);

      if (std::isnan(F_ij)) {
        std::cout << "NAN" << std::endl;
      }

      if (F_ij == 0) {
        continue;
      }

      inds.push_back(i);
      inds.push_back(j);
      values.push_back(F_ij);
    }
  }

  arma::umat locations(inds);
  locations.reshape(2, values.size());

  return arma::sp_mat(locations, arma::vec(values));

// #if USE_TBB

//   std::vector<sp_inds<arma::uword>> cols(num_faces);

//   auto const build_col = [this, offset, &cols] (size_t j) {
//     IntersectionInfo info;

//     auto tri_j = static_cast<Tri const *>(objects[j]);
//     auto p_j = tri_j->getCentroid();
//     auto n_j = tri_j->getNormal(info);
//     auto q_j = p_j + offset*n_j; // offset centroid = ray origin

//     auto & col = cols[j];
//     auto & rowind = col.rowind;
//     auto & colptr = col.colptr;

//     for (size_t i = 0; i < num_faces; ++i) {
//       auto tri_i = static_cast<Tri const *>(objects[i]);
//       auto p_i = tri_i->getCentroid();
//       auto r_i = tri_i->getBoundingRadius();

//       // Check if triangles are approximately "facing" each other
//       if ((p_i - p_j)*n_j <= -r_i) {
//         continue;
//       }

//       // Shoot a ray between the faces if they are (something *must*
//       // be hit)
//       Ray ray(q_j, normalize(p_i - q_j));
//       if (!bvh.getIntersection(ray, &info, false)) {
//         // TODO: should probably emit a warning here!
//         assert(info.object == nullptr);
//         continue;
//       }

//       // Get the index of the triangle that was hit by the ray
//       auto hit_index = static_cast<Tri const *>(info.object)->index;

//       // Search for the triangle that was hit and insert it (in sorted
//       // order) if it's a new visible triangle
//       auto lb = std::lower_bound(rowind.begin(), rowind.end(), hit_index);
//       if (lb == rowind.end() || *lb != static_cast<size_t>(tri_i->index)) {
//         rowind.insert(lb, hit_index);
//       }
//     }

//     colptr.push_back(0);
//     colptr.push_back(rowind.size());
//   };

//   tbb::parallel_for(size_t(0), num_faces, build_col);

//   join_horiz_reducer reducer;

//   tbb::parallel_reduce(
//     join_horiz_reducer::blocked_range_type {cols.begin(), cols.end()},
//     reducer);

//   auto & inds = reducer.inds;

//   assert(inds.colptr.size() == num_faces + 1);

//   // Now, compute form factors
//   //
//   // TODO: we're not doing this as efficiently as we could be: we're
//   // recomputing dot products, etc. ... but this is just to get t

//   std::vector<double> form_factors;
//   form_factors.reserve(inds.rowind.size());

//   for (arma::uword j = 0; j < num_faces; ++j) {
//     auto ptr0 = inds.colptr[j];
//     auto ptr1 = inds.colptr[j + 1];

//     auto T_j = static_cast<Tri const *>(objects[j]);
//     auto p_j = T_j->getCentroid();
//     auto n_j = T_j->getNormal(unused);

//     double A_j = length((T_j->v1 - T_j->v0)^(T_j->v2 - T_j->v0))/2;

//     for (arma::uword p = ptr0; p < ptr1; ++p) {
//       arma::uword i = inds.rowind[p];

//       if (i == j) {
//         form_factors.push_back(0);
//         continue;
//       }

//       auto T_i = static_cast<Tri const *>(objects[i]);
//       auto p_i = T_i->getCentroid();
//       auto n_i = T_i->getNormal(unused);

//       auto p_ij = p_j - p_i;
//       double r_ij = length(p_ij);
//       auto n_ij = p_ij/r_ij;

//       double mu_ij = fmax(0, n_i*n_ij);
//       double mu_ji = fmax(0, -n_j*n_ij);

//       double F_ij = mu_ij*mu_ji*A_j/(arma::datum::pi*r_ij*r_ij);

//       form_factors.push_back(F_ij);
//     }
//   }

//   assert(form_factors.size() == inds.rowind.size());

//   std::cout << "[max F = " << *std::max_element(form_factors.begin(), form_factors.end()) << "]" << std::endl;

//   return arma::sp_mat {
//     arma::uvec(inds.rowind),
//     arma::uvec(inds.colptr),
//     arma::vec(form_factors),
//     num_faces,
//     num_faces
//   };

// #else
// #  error "Not implemented yet"
// #endif
}

/**
 * Compute Frenet frame for triangle. The convention is that the
 * (n)ormal is just the normal attached to the triangle, the (t)angent
 * vector is (v1 - v0)/|v1 - v0|, and the bivector is just b = t x n.
 */
arma::mat
get_frenet_frame(Tri const * tri)
{
  arma::mat frame(3, 3);
  
  arma::vec n = {tri->n[0], tri->n[1], tri->n[2]};
  frame.col(2) = arma::normalise(n); // (n)ormal
  
  arma::vec t = {
    tri->v1[0] - tri->v0[0],
    tri->v1[1] - tri->v0[1],
    tri->v1[2] - tri->v0[2]
  };
  frame.col(1) = arma::normalise(t); // (t)angent

  frame.col(0) = arma::cross(frame.col(1), frame.col(2)); // (b)ivector

  return frame;
}

/**
 * ind: index of face for which we're tracing a horizon
 * objects: vector of triangles
 * bvh: bounding volume hierarchy used to accelerated raytracing
 * phis: vector of horizon abscissae (values in [0, 2pi))
 * thetas: theta values over which to search for horizon
 * offset: the normal displacement away from the plane of the triangle
 *
 * returns: vector of horizon angles (thetas) for each abscissa phi
 */
arma::vec
trace_horizon(
  Tri const * tri,
  BVH const & bvh,
  arma::vec const & phis,
  double theta_eps,
  double offset)
{
  IntersectionInfo info;

  auto p = tri->getCentroid();
  auto n = tri->getNormal(info);
  p = p + offset*n;

  arma::mat::fixed<3, 3> F = get_frenet_frame(tri);

  auto const get_ray_d = [&] (double ph, double th) {
    arma::vec v(3);
    v(0) = std::cos(ph)*std::sin(th);
    v(1) = std::sin(ph)*std::sin(th);
    v(2) = std::cos(th);
    v = F*v;
    return Vector3(v(0), v(1), v(2));
  };

  auto const shoot_ray = [&] (double ph, double th) {
    Ray ray(p, get_ray_d(ph, th));
    return bvh.getIntersection(ray, &info, false);
  };

  auto const ray_search = [&] (double ph) {
    double lo = 0, hi = arma::datum::pi, mid = hi/2, prev;
    do {
      prev = mid;
      if (shoot_ray(ph, mid)) {
        hi = mid;
      } else {
        lo = mid;
      }
      mid = (lo + hi)/2;
    } while (std::fabs(mid - prev) > theta_eps);    
    return mid;
  };

  arma::vec horizon(arma::size(phis));
  for (size_t i = 0; i < phis.n_elem; ++i) {
    horizon(i) = ray_search(phis(i));
  }

  return horizon;
}

void
illum_context::impl::make_horizons(
  int nphi,
  double theta_eps,
  double offset,
  opt_t<int> j0,
  opt_t<int> j1)
{
  auto phis = arma::linspace(0, 2*arma::datum::pi, nphi);

  int nhoriz = j1.value_or(num_faces) - j0.value_or(0);
  horizons.set_size(nphi, nhoriz);

  auto const compute_horizon = [&] (int j) {
    auto tri = static_cast<Tri const *>(objects[j]);
    horizons.col(j - j0.value_or(0)) = trace_horizon(tri, bvh, phis, theta_eps, offset);
  };

#if USE_TBB
  tbb::parallel_for(
    size_t(j0.value_or(0)),
    size_t(j1.value_or(num_faces)),
    compute_horizon);
#else
  for (int j = j0.value_or(0); j < j1.value_or(num_faces); ++j) {
    compute_horizon(j);
  }
#endif
}

void
illum_context::impl::save_horizons(
  boost::filesystem::path const & path,
  opt_t<int> i0,
  opt_t<int> i1) const
{
  arma_util::save_mat(horizons, path, i0, i1);
}

void
illum_context::impl::load_horizons(
  boost::filesystem::path const & path,
  opt_t<int> i0,
  opt_t<int> i1)
{
  horizons = arma_util::load_mat(path, i0, i1);
}

arma::vec::fixed<3> get_centroid(Object const * obj) {
  auto p = obj->getCentroid();
  return {p[0], p[1], p[2]};
}

arma::vec
illum_context::impl::get_direct_radiosity(
  arma::vec const & sun_position,
  arma::mat const & disk_XY,
  double sun_radius,
  opt_t<int> const & j0,
  opt_t<int> const & j1)
{
  using namespace arma;

  static const auto TWO_PI = 2*arma::datum::pi;

  int nhoriz = j1.value_or(horizons.n_cols) - j0.value_or(0);
  assert(horizons.n_cols == arma::uword(nhoriz));

  arma::vec direct;
  direct.set_size(nhoriz);

  auto nphi = horizons.n_rows;
  auto delta_phi = TWO_PI/(nphi - 1);

  auto const compute_direct_radiosity = [&] (int obj_ind) {
    auto obj = objects[obj_ind];
    int dir_ind = obj_ind - j0.value_or(0);

    vec::fixed<3> p = get_centroid(obj);

    vec::fixed<3> N = normalise(sun_position - p);
    vec::fixed<3> T = normalise((eye(3, 3) - N*N.t())*randn<vec>(3));
    vec::fixed<3> B = cross
      (T, N);

    mat disk(3, disk_XY.n_rows);
    for (size_t j = 0; j < disk_XY.n_rows; ++j) {
      disk.col(j) = sun_position +
        sun_radius*(disk_XY(j, 0)*T + disk_XY(j, 1)*B);
    }

    auto BTN = get_frenet_frame(static_cast<Tri const *>(obj));

    arma::vec horizon = horizons.col(dir_ind);

    int count = 0;

    for (size_t j = 0; j < disk_XY.n_rows - 1; ++j) {
      vec::fixed<3> n_sun = normalise(disk.col(j) - p);
      vec::fixed<3> dir = BTN.t()*n_sun;

      auto theta = std::acos(dir(2));
      auto phi = std::atan2(dir(1), dir(0));
      if (phi < 0) {
        phi += TWO_PI;
      }

      auto phi_index = static_cast<int>(std::floor(phi/delta_phi));
      auto t = (phi - phi_index*delta_phi)/delta_phi;

      if (theta < horizon(phi_index) + t*delta_phi) {
        ++count;
      }
    }

    // Initially set direct illumination to be the percentage of the
    // "sun points" that are visibile above the horizon
    direct(dir_ind) = static_cast<double>(count)/(nphi - 1);

    // Scale the direct illumination by the cosine between the normal
    // pointing towards the sun and the surface normal
    // (i.e. assume Lambertian---fine for thermal)
    direct(dir_ind) *= fmax(0, arma::dot(N, BTN.col(2)));

    assert(0 <= direct(dir_ind));
    assert(direct(dir_ind) <= 1);
  };

#if USE_TBB
  tbb::parallel_for(
    size_t(j0.value_or(0)),
    size_t(j1.value_or(horizons.n_cols)),
    compute_direct_radiosity);
#else
  for (int j = j0.value_or(0); j < j1.value_or(horizons.n_cols); ++j) {
    compute_direct_radiosity(j);
  }
#endif

  return direct;
}

void fib_spiral(arma::mat & xy, int n)
{
  static auto TWO_PI = 2*arma::datum::pi;

  xy.set_size(n, 2);

  auto a = 0., da = TWO_PI*(arma::datum::gratio - 1)/arma::datum::gratio;
  auto r = 0., dr = 1./(n + 1);

  for (int i = 0; i < n; ++i) {
    xy(i, 0) = r*std::cos(a);
    xy(i, 1) = r*std::sin(a);
    a = std::fmod(a + da, TWO_PI);
    r += dr;
  }
}
