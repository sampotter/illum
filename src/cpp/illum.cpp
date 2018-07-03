#include "illum.hpp"

#include <cassert>

#include <config.hpp>

#include "sp_inds.hpp"

#include <fastbvh>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#if USE_TBB
#  include <tbb/tbb.h>
#endif

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
  tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path);

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

    objects.push_back(new Tri(v0, v1, v2, n, i/3));
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

  void make_A(arma::sp_umat & A, double offset);

  void make_horizons(
    arma::mat & horizons,
    int nphi = 361,
    double theta_eps = 1e-3,
    double offset = 1e-5,
    opt_t<int> j0 = opt_t <int> {},
    opt_t<int> j1 = opt_t <int> {});

  void compute_visibility_ratios(
    arma::mat const & horizons,
    arma::vec const & sun_position,
    arma::mat const & disk_xy,
    arma::vec & ratios,
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
};

illum_context::illum_context(char const * path, int shape_index):
  pimpl {new illum_context::impl {get_objects(path, shape_index)}}
{}

illum_context::~illum_context() = default;

void
illum_context::make_A(arma::sp_umat & A, double offset)
{
  pimpl->make_A(A, offset);
}

void
illum_context::make_horizons(
  arma::mat & horizons,
  int nphi,
  double theta_eps,
  double offset,
  opt_t<int> j0,
  opt_t<int> j1)
{
  pimpl->make_horizons(horizons, nphi, theta_eps, offset, j0, j1);
}

void
illum_context::compute_visibility_ratios(
  arma::mat const & horizons,
  arma::vec const & sun_position,
  arma::mat const & disk_xy,
  arma::vec & ratios,
  double sun_radius,
  opt_t<int> j0,
  opt_t<int> j1)
{
  pimpl->compute_visibility_ratios(
    horizons,
    sun_position,
    disk_xy,
    ratios,
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

// TODO: this is going to be really inefficient---build from
// std::vectors instead
void
illum_context::impl::make_A(arma::sp_umat & A, double offset)
{
  IntersectionInfo unused;

#if USE_TBB

  std::vector<sp_inds<arma::uword>> cols(num_faces);

  auto const build_col = [this, offset, &cols] (size_t j) {
    IntersectionInfo info;

    auto tri_j = static_cast<Tri const *>(objects[j]);
    auto p_j = tri_j->getCentroid();
    auto n_j = tri_j->getNormal(info);
    auto q_j = p_j + offset*n_j; // offset centroid = ray origin

    auto & col = cols[j];
    auto & rowind = col.rowind;
    auto & colptr = col.colptr;

    for (size_t i = 0; i < num_faces; ++i) {
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
      bvh.getIntersection(ray, &info, false);

      // Get the index of the triangle that was hit by the ray
      auto hit_index = static_cast<Tri const *>(info.object)->index;

      // Search for the triangle that was hit and insert it (in sorted
      // order) if it's a new visible triangle
      auto lb = std::lower_bound(rowind.begin(), rowind.end(), hit_index);
      if (lb == rowind.end() || *lb != static_cast<size_t>(tri_i->index)) {
        rowind.insert(lb, hit_index);
      }
    }

    colptr.push_back(0);
    colptr.push_back(rowind.size());
  };

  tbb::parallel_for(size_t(0), num_faces, build_col);

  join_horiz_reducer reducer;

  tbb::parallel_reduce(
    join_horiz_reducer::blocked_range_type {cols.begin(), cols.end()},
    reducer);

  auto & inds = reducer.inds;

  std::cout << inds.rowind.size() << ", " << inds.colptr.size() << std::endl;

  assert(inds.colptr.size() == num_faces + 1);

  A = arma::sp_umat {
    arma::uvec(inds.rowind),
    arma::uvec(inds.colptr),
    arma::uvec(inds.rowind.size(), arma::fill::ones),
    num_faces,
    num_faces
  };

#else

  std::vector<arma::uword> rowind, colptr;

  colptr.push_back(0);

  for (size_t j = 0; j < num_faces; ++j) {
    auto tri_j = static_cast<Tri const *>(objects[j]);
    auto p_j = tri_j->getCentroid();
    auto n_j = tri_j->getNormal(unused);
    auto q_j = p_j + offset*n_j; // offset centroid = ray origin
    
    for (size_t i = 0; i < num_faces; ++i) {
      auto tri_i = static_cast<Tri const *>(objects[i]);
      auto p_i = tri_i->getCentroid();
      auto r_i = tri_i->getBoundingRadius();

      // Check if triangles are approximately "facing" each other
      if ((p_i - p_j)*n_j <= -r_i) {
        continue;
      }

      // Shoot a ray between the faces if they are (something *must*
      // be hit)
      IntersectionInfo info;
      Ray ray(q_j, normalize(p_i - q_j));
      bvh.getIntersection(ray, &info, false);

      // Get the index of the triangle that was hit by the ray
      auto hit_index = static_cast<Tri const *>(info.object)->index;

      // Search for the triangle that was hit and insert it (in sorted
      // order) if it's a new visible triangle
      auto lb = std::lower_bound(rowind.begin(), rowind.end(), hit_index);
      if (lb == rowind.end() || *lb != static_cast<size_t>(tri_i->index)) {
        rowind.insert(lb, hit_index);
      }
    }

    colptr.push_back(rowind.size());
  }

  A = arma::sp_umat {
    arma::uvec(rowind),
    arma::uvec(colptr),
    arma::uvec(rowind.size(), arma::fill::ones),
    num_faces,
    num_faces
  };
#endif
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

  auto F = get_frenet_frame(tri);

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
  arma::mat & horizons,
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
compute_V(
  arma::sp_umat const & A,
  arma::sp_umat & V)
{
  V = A.t()%A;
}

void
illum_context::impl::compute_visibility_ratios(
  arma::mat const & horizons,
  arma::vec const & sun_position,
  arma::mat const & disk_XY,
  arma::vec & ratios,
  double sun_radius,
  opt_t<int> const & j0,
  opt_t<int> const & j1)
{
  using namespace arma;

  static const auto TWO_PI = 2*arma::datum::pi;

  int nhoriz = j1.value_or(horizons.n_cols) - j0.value_or(0);
  assert(horizons.n_cols == nhoriz);
  ratios.set_size(nhoriz);

  auto nphi = horizons.n_rows;
  auto delta_phi = TWO_PI/(nphi - 1);

  auto const compute_ratio = [&] (int obj_ind) {
    auto obj = objects[obj_ind];

    int ratio_ind = obj_ind - j0.value_or(0);

    vec::fixed<3> p;
    {
      auto centroid = obj->getCentroid();
      p(0) = centroid[0];
      p(1) = centroid[1];
      p(2) = centroid[2];
    }

    auto d = sun_position - p;

    auto N = normalise(d);
    vec::fixed<3> T = normalise((eye(3, 3) - N*N.t())*randn<vec>(3));
    vec::fixed<3> B = cross(T, N);

    mat disk(3, disk_XY.n_rows);
    for (size_t j = 0; j < disk_XY.n_rows; ++j) {
      disk.col(j) = sun_radius*(disk_XY(j, 0)*T + disk_XY(j, 1)*B);
    }

    auto btn = get_frenet_frame(static_cast<Tri const *>(obj));

    arma::vec horizon = horizons.col(ratio_ind);

    int count = 0;

    for (size_t j = 0; j < disk_XY.n_rows - 1; ++j) {
      vec::fixed<3> dir = btn.t()*normalise(d + disk.col(j));

      // TODO: not necessary---could store the horizons in the correct
      // format in the first place
      auto phi = std::atan2(dir(1), dir(0));
      if (phi < 0) phi += TWO_PI;

      auto theta = std::acos(dir(2));

      auto phi_index = static_cast<int>(std::floor(phi/delta_phi));
      auto t = (phi - phi_index*delta_phi)/delta_phi;

      if (theta < (1 - t)*horizon(phi_index) + t*horizon(phi_index + 1)) {
        ++count;
      }
    }

    ratios(ratio_ind) = static_cast<double>(count)/(nphi - 1);
  };

#if USE_TBB
  tbb::parallel_for(
    size_t(j0.value_or(0)),
    size_t(j1.value_or(horizons.n_cols)),
    compute_ratio);
#else
  for (int j = j0.value_or(0); j < j1.value_or(horizons.n_cols); ++j) {
    compute_ratio(j);
  }
#endif
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
