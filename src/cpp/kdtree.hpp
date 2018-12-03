#ifndef __KDTREE_HPP__
#define __KDTREE_HPP__

#include <fastbvh>

using num_t = float;
using points_t = std::vector<Vector3>;

struct kdtree_impl;

struct kdtree_t
{
  kdtree_t(points_t const & points);
  ~kdtree_t();
  size_t query(Vector3 const & p, size_t k, size_t * I, num_t * D_sq) const;
private:
  kdtree_impl * m;
};

#endif // __KDTREE_HPP__
