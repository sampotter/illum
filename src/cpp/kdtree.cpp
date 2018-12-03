#include "kdtree.hpp"

#include <nanoflann.hpp>

struct point_cloud_t
{
  points_t const & points;

  point_cloud_t(points_t const & points): points {points} {}

  inline size_t kdtree_get_point_count() const {
    return points.size();
  }

  inline num_t kdtree_get_pt(size_t const i, size_t const dim) const {
    return points[i][dim];
  }

  // This is optional---see nanoflann documentation. When we just
  // return false, a default bounding box calculation is used
  // somewhere internally in nanoflann.
  template <class BBox>
  bool kdtree_get_bbox(BBox & /*bb*/) const {
    return false;
  }
};

using kdtree_adaptor_t = nanoflann::KDTreeSingleIndexAdaptor<
  nanoflann::L2_Simple_Adaptor<num_t, point_cloud_t>,
  point_cloud_t,
  3>;

struct kdtree_impl
{
  point_cloud_t point_cloud;
  kdtree_adaptor_t adaptor;

  kdtree_impl(std::vector<Vector3> const & points):
    point_cloud {points},
    adaptor {3, point_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10)}
  {
    adaptor.buildIndex();
  }

  size_t query(Vector3 const & p, size_t k, size_t * I, num_t * D_sq) const
  {
    return adaptor.knnSearch(&p[0], k, I, D_sq);
  }
};

kdtree_t::kdtree_t(points_t const & points)
{
  m = new kdtree_impl(points);
}

kdtree_t::~kdtree_t()
{
  delete m;
}

size_t
kdtree_t::query(Vector3 const & p, size_t k, size_t * I, num_t * D_sq) const
{
  return m->query(p, k, I, D_sq);
}
