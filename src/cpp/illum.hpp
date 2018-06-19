#ifndef __ILLUM_HPP__
#define __ILLUM_HPP__

#include <unordered_map>

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
prune_A(BVH const & bvh, std::vector<Object *> const & objects,
        arma::sp_umat const & A, double offset=1e-5)
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
  return arma::sp_umat(rowind, colptr, values, nfaces, nfaces);
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
              double theta_eps = 1e-3,
              double offset = 1e-5)
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

arma::mat
build_horizon_map(std::vector<Object *> const & objects,
                  BVH const & bvh, int nphi, double theta_eps = 1e-3,
                  double offset = 1e-5)
{
  auto phis = arma::linspace(0, 2*arma::datum::pi, nphi);
  arma::mat H(nphi, objects.size());
  // TODO: maybe can replace this w/ foreach
  // TODO: openmp
  for (int ind = 0; ind < objects.size(); ++ind) {
    // std::cout << ind << std::endl;
    Tri const * tri = static_cast<Tri const *>(objects[ind]);
    H.col(ind) = trace_horizon(tri, bvh, phis, theta_eps, offset);
  }
  return H;
}

#endif // __ILLUM_HPP__
