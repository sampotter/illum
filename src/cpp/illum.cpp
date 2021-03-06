#include "illum.hpp"

#include <cassert>

#include <config.hpp>

#include "arma_util.hpp"
#include "constants.hpp"
#include "obj_util.hpp"
#include "sp_inds.hpp"

#include <nanoflann.hpp>
#if USE_TBB
#  include <tbb/tbb.h>
#endif

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <vector>

illum_context::illum_context(boost::filesystem::path const & obj_path):
  objects {obj_util::get_objects(obj_path.c_str(), 0)},
  bvh_objects {objects.begin(), objects.end()},
  bvh {&bvh_objects},
  num_faces {objects.size()}
{}

illum_context::~illum_context() {}

arma::sp_mat
illum_context::compute_F(
  var_t<double, std::string> const & albedo,
  double offset)
{
  struct elt {
    elt(arma::uword i, arma::uword j, double F): i {i}, j {j}, F {F} {}
    arma::uword i, j;
    double F;
  };

  auto compute_form_factors_for_face = [&] (arma::uword j) -> std::vector<elt> {
    IntersectionInfo info;

    std::vector<elt> elts;

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
      if (r_ij < std::numeric_limits<double>::epsilon()) {
        continue;
      }

      auto n_ij = p_ij/r_ij;

      double mu_ij = fmax(0, n_i*n_ij);
      double mu_ji = fmax(0, -n_j*n_ij);

      double F_ij = mu_ij*mu_ji*A_j/(arma::datum::pi*r_ij*r_ij);

      if (std::isnan(F_ij)) {
        throw std::runtime_error("computed a NaN form factor");
      }

      if (F_ij == 0) {
        continue;
      }

      elts.emplace_back(i, j, F_ij);
    }

    return elts;
  };

  /**
   * First, compute form factor elements for all approximately visible
   * pairs.  If we're using TBB, we do this using a parallel reducer,
   * otherwise we do it in a relatively straightforward serial
   * fashion.
   */
  std::vector<elt> elts;
#if USE_TBB
  tbb::concurrent_vector<std::vector<elt>> all_elts(num_faces);

  tbb::parallel_for(size_t(0), num_faces, [&] (size_t j) {
    all_elts[j] = compute_form_factors_for_face(j);
  });

  struct reducer
  {
    using range_t = tbb::blocked_range<decltype(all_elts)::iterator>;

    reducer() {}
    reducer(reducer &, tbb::split) {}

    void operator()(range_t const & r) {
      for (auto it = r.begin(); it != r.end(); ++it) {
        elts.insert(elts.end(), it->begin(), it->end());
      }
    }

    void join(reducer const & other) {
      elts.insert(elts.end(), other.elts.begin(), other.elts.end());
    }

    std::vector<elt> elts;
  };

  reducer r;
  tbb::parallel_reduce(reducer::range_t {all_elts.begin(), all_elts.end()}, r);
  elts = r.elts;
#else
  for (arma::uword j = 0; j < num_faces; ++j) {
    auto new_elts = compute_form_factors_for_face(j);
    elts.insert(elts.end(), new_elts.begin(), new_elts.end());
  }
#endif

  /**
   * Next, we sort the indices into column major order. Although
   * Armadillo can do this, we do this ourselves to take advantage of
   * TBB's parallel sort when it's available to us.
   */
  auto comp = [] (elt const & e1, elt const & e2) -> bool {
    return e1.j == e2.j ? e1.i < e2.i : e1.j < e2.j;
  };
#if USE_TBB
  tbb::parallel_sort(elts.begin(), elts.end(), comp);
#else
  std::sort(elts.begin(), elts.end(), comp);
#endif

  /**
   * Build the matrix of location values in the format Armadillo expects
   * for its constructor.
   */
  arma::umat locs(2, elts.size());
  for (arma::uword k = 0; k < elts.size(); ++k) {
    locs(0, k) = elts[k].i;
    locs(1, k) = elts[k].j;
  }

  /**
   * Build the vector of form factors to pass to the sp_mat
   * constructor.
   */
  arma::vec values(elts.size());
  for (arma::uword k = 0; k < elts.size(); ++k) {
    values(k) = elts[k].F;
  }

  /**
   * Construct and return the sparse form factor matrix---the false
   * here tells Armadillo not to sort the entries into column-major
   * ordering, since we've already done that above.
   */
  arma::sp_mat F {locs, values, num_faces, num_faces, false};

  /**
   * Scale each row of F by the albedo. If the variant that holds the
   * albedo is a double, we assume constant albedo and scale each row
   * by the same number. Otherwise, we load the vector stored at the
   * given path and do diagonal scaling.
   */
  if (double const * rho = boost::get<double>(&albedo)) {
    F *= *rho;
  }
  else if (std::string const * path = boost::get<std::string>(&albedo)) {
    arma::vec Rho;
    Rho.load(*path);
    for (arma::uword i = 0; i < F.n_rows; ++i) {
      F.row(i) *= Rho(i);
    }
  }

  return F;
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
illum_context::make_horizons(int nphi, double theta_eps, double offset)
{
  auto phis = arma::linspace(0, 2*arma::datum::pi, nphi);
  int nhoriz = num_faces;
  horizons.set_size(nphi, nhoriz);

  if (horizon_obj_path) {
    std::vector<Object *> horizon_objects =
      obj_util::get_objects(horizon_obj_path->c_str(), 0);
    std::vector<Object *> horizon_bvh_objects {
      horizon_objects.begin(), horizon_objects.end()};
    BVH horizon_bvh(&horizon_bvh_objects);
    auto const compute_horizon = [&] (int j) {
      auto tri = static_cast<Tri const *>(objects[j]);
      horizons.col(j) = trace_horizon(tri, horizon_bvh, phis, theta_eps, offset);
    };
#if USE_TBB
    tbb::parallel_for(size_t(0), size_t(num_faces), compute_horizon);
#else
    for (int j = 0; j < num_faces; ++j) {
      compute_horizon(j);
    }
#endif
  } else {
    auto const compute_horizon = [&] (int j) {
      auto tri = static_cast<Tri const *>(objects[j]);
      horizons.col(j) = trace_horizon(tri, bvh, phis, theta_eps, offset);
    };
#if USE_TBB
    tbb::parallel_for(size_t(0), size_t(num_faces), compute_horizon);
#else
    for (int j = 0; j < num_faces; ++j) {
      compute_horizon(j);
    }
#endif
  }
}

void
illum_context::save_horizons(boost::filesystem::path const & path) const
{
  arma_util::save_mat(horizons, path);
}

void
illum_context::load_horizons(boost::filesystem::path const & path)
{
  horizons = arma_util::load_mat<arma::mat>(path);
}

arma::vec::fixed<3> get_centroid(Object const * obj) {
  auto p = obj->getCentroid();
  return {p[0], p[1], p[2]};
}

arma::vec
illum_context::get_direct_radiosity(arma::vec const & sun_position,
                                    arma::mat const & disk_XY)
{
  using namespace arma;

  static const auto TWO_PI = 2*arma::datum::pi;

  int nhoriz = horizons.n_cols;
  assert(horizons.n_cols == arma::uword(nhoriz));

  arma::vec direct;
  direct.set_size(nhoriz);

  auto nphi = horizons.n_rows;
  auto delta_phi = TWO_PI/(nphi - 1);

  auto const compute_direct_radiosity = [&] (int obj_ind) {
    auto obj = objects[obj_ind];
    int dir_ind = obj_ind;

    vec::fixed<3> p = get_centroid(obj);

    vec::fixed<3> N = normalise(sun_position - p);
    vec::fixed<3> T = normalise((eye(3, 3) - N*N.t())*randn<vec>(3));
    vec::fixed<3> B = cross(T, N);

    mat disk(3, disk_XY.n_rows);
    for (size_t j = 0; j < disk_XY.n_rows; ++j) {
      disk.col(j) = sun_position +
        constants::SUN_RADIUS_KM*(disk_XY(j, 0)*T + disk_XY(j, 1)*B);
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

    // TODO: Scale by the total power: not at all confident that this
    // is correct, but Erwan seems to think so; will trust this number
    // for now
    direct(dir_ind) *= 1372*std::pow(
      constants::ONE_AU_KM/arma::norm(sun_position), 2);

    assert(0 <= direct(dir_ind));
  };

#if USE_TBB
  tbb::parallel_for(size_t(0), size_t(nhoriz), compute_direct_radiosity);
#else
  for (int j = 0; j < nhoriz; ++j) {
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
