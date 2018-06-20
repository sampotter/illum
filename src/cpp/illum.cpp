#include "illum.hpp"

#include <fastbvh>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <unordered_map>
#include <vector>

std::vector<Object *> get_objects(const char * path, int shape_index) {
  /**
   * Use tinyobjloader to load selected obj file.
   */
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path);

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

  for (int i = 0; i < indices.size(); i += 3) {
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
    bvh {&bvh_objects}
  {}

  void make_A(arma::sp_umat & A);
  void prune_A(arma::sp_umat const & A, arma::sp_umat & pruned, double offset = 1e-5);
  void make_horizons(arma::mat & horizons, int nphi = 361, double theta_eps = 1e-3,
                     double offset = 1e-5);

  std::vector<Object *> objects;
  std::vector<Object *> bvh_objects;
  BVH bvh;
};

illum_context::illum_context(char const * path, int shape_index):
  pimpl {std::make_unique<illum_context::impl>(get_objects(path, shape_index))}
{}

illum_context::~illum_context() = default;

void
illum_context::make_A(arma::sp_umat & A)
{
  pimpl->make_A(A);
}

void
illum_context::prune_A(arma::sp_umat const & A, arma::sp_umat & pruned,
                       double offset)
{
  pimpl->prune_A(A, pruned, offset);
}

void
illum_context::make_horizons(arma::mat & horizons, int nphi, double theta_eps,
                             double offset)
{
  pimpl->make_horizons(horizons, nphi, theta_eps, offset);
}

void
illum_context::impl::make_A(arma::sp_umat & A)
{
  int nfaces = objects.size();

  auto const get_bounding_radius = [] (Tri const * tri) {
    auto p = tri->getCentroid();
    return std::max(
      length(tri->v0 - p),
      std::max(
        length(tri->v1 - p),
        length(tri->v2 - p)));
  };

  arma::vec R(nfaces);
  for (int i = 0; i < nfaces; ++i) {
    auto tri_i = static_cast<Tri const *>(objects[i]);
    R(i) = get_bounding_radius(tri_i);
  }

  auto const P = [&] (int i) {
    return objects[i]->getCentroid();
  };

  auto const N = [&] (int i) {
    IntersectionInfo unused;
    return objects[i]->getNormal(unused);
  };

  // TODO: this is going to be really inefficient---build from
  // std::vectors instead
  A = arma::sp_umat(nfaces, nfaces);
  for (int j = 0; j < nfaces; ++j) {
    for (int i = 0; i < nfaces; ++i) {
      if ((P(i) - P(j))*N(j) > -R(i)) {
        A(i, j) = 1;
      }
    }
  }
}

void
illum_context::impl::prune_A(arma::sp_umat const & A, arma::sp_umat & pruned,
                             double offset)
{
  int nfaces = objects.size();
  assert(nfaces == A.n_rows && nfaces == A.n_cols);

  IntersectionInfo info;

  std::vector<arma::uword> rowind_vec, colptr_vec;
  colptr_vec.reserve(nfaces + 1);

  int col = 0;
  colptr_vec.push_back(col);

  for (int j = 0; j < nfaces; ++j) {
    auto * obj_j = objects[j];
    auto n_j = obj_j->getNormal(info);
    auto p_j = obj_j->getCentroid();
    p_j = p_j + offset*n_j; // perturb to avoid self-intersection

    std::vector<arma::uword> hits;

    auto cast_ray_and_insert_hit = [&] (Vector3 const & normal) {
      if (bvh.getIntersection(Ray(p_j, normal), &info, false)) {
        auto index = static_cast<Tri const *>(info.object)->index;
        auto lb = std::lower_bound(hits.begin(), hits.end(), index);
        if (lb == hits.end() || *lb != index) {
          hits.insert(lb, index);
        }
      }
    };

    for (int i = 0; i < nfaces; ++i) {
      if (!A(i, j)) continue;

      // Shoot one ray from centroid to centroid:
      auto p_i = objects[i]->getCentroid();
      cast_ray_and_insert_hit(normalize(p_i - p_j));
    }

    // std::cout << hits.size() << "/" << A.col(i).n_nonzero << std::endl;
    col += hits.size();
    colptr_vec.push_back(col);

    rowind_vec.insert(rowind_vec.end(), hits.begin(), hits.end());
  }

  arma::uvec rowind(rowind_vec), colptr(colptr_vec);
  arma::uvec values(rowind.n_elem, arma::fill::ones);
  pruned = arma::sp_umat(rowind, colptr, values, nfaces, nfaces);
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
 * 
 * TODO: add a perturbation parameter (amount to lift observer away
 * from ground to get higher vantage point)
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
  for (int i = 0; i < phis.n_elem; ++i) {
    horizon(i) = ray_search(phis(i));
  }

  return horizon;
}

void
illum_context::impl::make_horizons(arma::mat & horizons, int nphi,
                                   double theta_eps, double offset)
{
  auto phis = arma::linspace(0, 2*arma::datum::pi, nphi);
  horizons.set_size(nphi, objects.size());
  // TODO: maybe can replace this w/ foreach
  // TODO: openmp
  for (int ind = 0; ind < objects.size(); ++ind) {
    auto tri = static_cast<Tri const *>(objects[ind]);
    horizons.col(ind) = trace_horizon(tri, bvh, phis, theta_eps, offset);
  }
}

void compute_V(arma::sp_umat const & A, arma::sp_umat & V)
{
  V = A.t()%A;
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