#include "icq.hpp"

#include <ifstream>
#include <string>

template <class T>
icq::mesh<T>::mesh(const char * path)
{
  std::size_t size;
  T x, y, z;

  std::string line;
  std::ifstream f {path};

  { // First get the header, which is just a single line containing
    // the number of quads in each direction on a face of the cube
    // in the ICQ model
    std::getline(f, line);
    std::istringstream ss {line};
    ss >> size;
  }

  // Reserve an appropriate amount of space in _vertices for storing all
  // of the vertices in the ICQ file
  std::size_t nodes_per_face = (size + 1)*(size + 1);
  _vertices.reserve(3*6*nodes_per_face);

  // Read in each line and store the corresponding vertex (x, y, z) in
  // _vertices
  while (std::getline(f, line)) {
    std::istringstream ss {line};
    ss >> x >> y >> z;
    _vertices.push_back(x);
    _vertices.push_back(y);
    _vertices.push_back(z);
  }

  _tris.reserve(2*6*size*size);

  int index = 0;
  for (int face = 0; face < 6; ++face) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        T * p00 = &_vertices[3*(nodes_per_face*face + (size + 1)*i + j)];
        T * p10 = &_vertices[3*(nodes_per_face*face + (size + 1)*(i + 1) + j)];
        T * p01 = &_vertices[3*(nodes_per_face*face + (size + 1)*i + (j + 1))];
        T * p11 = &_vertices[3*(nodes_per_face*face + (size + 1)*(i + 1) + j + 1)];

        /**
         * For now, we decompose each set of four nodes into two
         * triangles like this. There may be a more sophisticated way
         * of doing this, but it should probably work okay for now.
         * 
         * v00 +----+ v01
         *     |n1 /|
         *     |  / |
         *     | /  |
         *     |/ n2|
         * v10 +----+ v11
         */

        Vector3 v00 {p00[0], p00[1], p00[2]};
        Vector3 v10 {p10[0], p10[1], p10[2]};
        Vector3 v01 {p01[0], p01[1], p01[2]};
        Vector3 v11 {p11[0], p11[1], p11[2]};

        n1 = normalize((v10 - v00)^(v01 - v00));
        n2 = normalize((v01 - v11)^(v10 - v11));

        _tris.push_back(Tri(v00, v10, v01, n1, index++));
        _tris.push_back(Tri(v11, v01, v10, n2, index++));
      }
    }
  }
}
