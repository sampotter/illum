#include "obj_util.hpp"

#include <fastbvh>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

std::vector<Object *>
obj_util::get_objects(
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
