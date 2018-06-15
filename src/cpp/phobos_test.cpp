#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// #include <omp.h>

#include <armadillo>
#include <fastbvh>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

float eps = 0;

std::vector<Object *> get_objects(tinyobj::attrib_t const & attrib,
								  tinyobj::shape_t const & shape) {
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

void
build_A_zero(std::vector<Object *> const & objects, arma::sp_umat & A) {
  assert(A.is_empty());
  auto nfaces = objects.size();
  A.set_size(nfaces, nfaces);
  IntersectionInfo unused;
  for (int j = 0; j < nfaces; ++j) {
    auto obj_j = objects[j];
    auto p_j = obj_j->getCentroid();
    auto n_j = obj_j->getNormal(unused);
    auto rhs = p_j*n_j;
    for (int i = 0; i < nfaces; ++i) {
      auto obj_i = objects[i];
      auto p_i = obj_i->getCentroid();
      auto lhs = p_i*n_j;
      if (lhs > rhs) {
        A(i, j) = true;
      }
    }
  }
}

arma::sp_umat compute_A(std::vector<Object *> const & objects) {
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

  arma::sp_umat A(nfaces, nfaces);
  for (int j = 0; j < nfaces; ++j) {
    for (int i = 0; i < nfaces; ++i) {
      if ((P(i) - P(j))*N(j) > -R(i)) {
        A(i, j) = 1;
      }
    }
  }

  return A;
}

arma::sp_umat compute_V(arma::sp_umat const & A) {
  return A.t()%A;
}

arma::sp_umat
fix_visibility(BVH const & bvh, std::vector<Object *> const & objects,
               arma::sp_umat const & V) {
  IntersectionInfo info;

  std::vector<arma::uword> rowind_vec, colptr_vec;
  colptr_vec.reserve(V.n_cols);

  int col = 0;
  colptr_vec.push_back(col);

  for (int i = 0; i < objects.size(); ++i) {
    auto * obj_i = objects[i];
    auto n_i = obj_i->getNormal(info);
    auto p_i = obj_i->getCentroid();
    p_i = p_i + 1e-5*n_i; // perturb to avoid self-intersection

    std::vector<arma::uword> hits;

    auto cast_ray_and_insert_hit = [&] (Vector3 const & normal) {
      if (bvh.getIntersection(Ray(p_i, normal), &info, false)) {
        auto index = static_cast<Tri const *>(info.object)->index;
        auto lb = std::lower_bound(hits.begin(), hits.end(), index);
        if (lb == hits.end() || *lb != index) {
          hits.insert(lb, index);
        }
      }
    };

    for (int j = 0; j < objects.size(); ++j) {
      // TODO: not accessing this in the most cache efficient way
      if (!V(i, j)) {
        continue;
      }

      auto * tri_j = static_cast<Tri const *>(objects[j]);
      auto p_j = tri_j->getCentroid();

      // Shoot one ray from centroid to centroid:
      cast_ray_and_insert_hit(normalize(p_j - p_i));
    }

    // std::cout << hits.size() << "/" << V.col(i).n_nonzero << std::endl;

    col += hits.size();
    colptr_vec.push_back(col);

    rowind_vec.insert(rowind_vec.end(), hits.begin(), hits.end());
  }

  arma::uvec rowind(rowind_vec), colptr(colptr_vec);
  arma::uvec values(rowind.n_elem, arma::fill::ones);
  return arma::sp_umat(rowind, colptr, values, V.n_rows, V.n_cols);
}

/**
 * Compute Frenet frame for triangle. The convention is that the
 * (n)ormal is just the normal attached to the triangle, the (t)angent
 * vector is (v1 - v0)/|v1 - v0|, and the bivector is just b = t x n.
 */
arma::mat
get_frenet_frame(Tri const * tri) {
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
trace_horizon(Tri const * tri,
              BVH const & bvh,
              arma::vec const & phis,
              double const theta_eps = 1e-3)
{
  IntersectionInfo info;

  auto p = tri->getCentroid();
  auto n = tri->getNormal(info);
  p = p + 1e-5*n;

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

arma::mat
build_horizon_map(std::vector<Object *> const & objects,
                   BVH const & bvh, int nphi)
{
  auto phis = arma::linspace(0, 2*arma::datum::pi, nphi);
  arma::mat H(nphi, objects.size());
  // TODO: maybe can replace this w/ foreach
  // TODO: openmp
  for (int ind = 0; ind < objects.size(); ++ind) {
    // std::cout << ind << std::endl;
    Tri const * tri = static_cast<Tri const *>(objects[ind]);
    H.col(ind) = trace_horizon(tri, bvh, phis);
  }
  return H;
}

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
void
write_csc_inds(arma::SpMat<T> const & S, const char * path) {
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

int main(int argc, char * argv[])
{
  // std::string filename = "../../../../data/SHAPE0.OBJ";
  std::string filename = "../../../../data/SHAPE0_dec5000.obj";
  if (argc >= 2) filename = argv[1];

  int shape_index = 0;
  if (argc >= 3) shape_index = std::atoi(argv[2]);

  /**
   * Use tinyobjloader to load selected obj file.
   */
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string err;
  bool ret = tinyobj::LoadObj(
    &attrib, &shapes, &materials, &err, filename.c_str());

  /**
   * Build a vector of objects for use with our bounding volume
   * hierarchy library (fastbvh).
   */
  tinyobj::shape_t & shape = shapes[shape_index];
  std::vector<Object *> objects = get_objects(attrib, shape);
  std::vector<Object *> bvh_objects(objects.begin(), objects.end());

  auto nfaces = objects.size();

  /**
   * Build the BVH.
   */
  BVH bvh(&bvh_objects);

  //////////////////////////////////////////////////////////////////////////////
  // visibility sandbox
  //

  std::cout << "computing A" << std::endl;
  auto A_before = compute_A(objects);
  write_csc_inds(A_before, "A_before.h5");
  // write_coo(A_before, "A_before.coo");

  std::cout << "fixing A" << std::endl;
  auto A_after = fix_visibility(bvh, objects, compute_V(A_before));
  write_csc_inds(A_after, "A_after.h5");
  // write_coo(A_after, "A_after.coo");

  //////////////////////////////////////////////////////////////////////////////
  // horizon sandbox

  // int nphi = 361;
  // auto H = build_horizon_map(objects, bvh, nphi);
  // H.save(arma::hdf5_name("horizons.h5", "horizons"));
}
