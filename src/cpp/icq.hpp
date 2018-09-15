#ifndef __ICQ_HPP__
#define __ICQ_HPP__

#include <fastbvh>
#include <type_traits>

template <class T = float>
namespace icq {
  static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
  
  struct mesh {
    mesh(const char * path);
  private:
    std::vector<T> _vertices;
    std::vector<Tri> _tris;
  };
}

#endif // __ICQ_HPP__
