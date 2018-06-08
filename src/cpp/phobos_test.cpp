#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

    auto const n = Vector3 {
      normals[3*i0.normal_index],
      normals[3*i0.normal_index + 1],
      normals[3*i0.normal_index + 2]
    };

    objects.push_back(new Tri(v0, v1, v2, n, i));
  }

  return objects;
}

std::vector<int> get_num_vis_per_tri(std::vector<Object *> const & objects) {
  std::vector<int> num_vis_per_tri(objects.size(), 0);

  IntersectionInfo info;

  for (int i = 0; i < objects.size(); ++i) {
    auto const obj_i = objects[i];
    auto const n_i = obj_i->getNormal(info);
    auto const p_i = obj_i->getCentroid();
    auto const d_i = n_i*p_i - eps;

    for (int j = 0; j < objects.size(); ++j) {
      if (i == j) {
        continue;
      }
      
      auto const obj_j = objects[j];
      auto const p_j = obj_j->getCentroid();

      if (n_i*p_j <= d_i) {
        continue;
      }

      auto const n_j = obj_j->getNormal(info);
      auto const n_ij = normalize(p_j - p_i);

      if (n_i*n_ij > 0 && n_j*n_ij < 0) {
        ++num_vis_per_tri[i];
      }
    }
  }

  return num_vis_per_tri;
}

std::vector<std::vector<int>> get_vis(std::vector<Object *> const & objects,
                                      BVH const & bvh) {
  std::vector<std::vector<int>> vis;

  IntersectionInfo info;

  // Traverse all of the faces of the triangle mesh.
  for (int i = 0; i < objects.size(); ++i) {
    auto * obj_i = objects[i];
    
    std::vector<int> indices;

    auto const p_i = obj_i->getCentroid() + 1e-6*obj_i->getNormal(info);

    for (int j = 0; j < i; ++j) {

      auto * obj_j = objects[j];

      auto const p_j = obj_j->getCentroid();
      auto const n_ij = normalize(p_j - p_i);

      // Cast a ray from the ith face to the jth face
      if (bvh.getIntersection(Ray(p_i, n_ij), &info, false)) {
        int index = static_cast<Tri const *>(info.object)->index;
        if (index == i) {
          continue;
        }
        auto const res = std::find(indices.begin(), indices.end(), index);
        if (res == indices.end()) {
          indices.push_back(index);
        }
      }
    }

    vis.push_back(indices);
  }

  return vis;
}

int main(int argc, char * argv[])
{
  std::string filename = "../data/Phobos_Ernst_decimated50k.obj";;
  if (argc >= 2) {
	filename = argv[1];
  }

  int shape_index = 0;
  if (argc >= 3) {
    shape_index = std::atoi(argv[2]);
  }

  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string err;
  bool ret = tinyobj::LoadObj(
    &attrib, &shapes, &materials, &err, filename.c_str());

  tinyobj::shape_t & shape = shapes[shape_index];
  
  std::vector<Object *> objects = get_objects(attrib, shape);

  BVH bvh(&objects);
  
  IntersectionInfo info;

  for (int i = 0; i < objects.size(); ++i) {
    auto obj_i = objects[i];

    auto p_i = obj_i->getCentroid();
    auto n_i = obj_i->getNormal(info);

    for (int j = 0; j < objects.size(); ++j) {
      auto obj_j = objects[j];

      auto p_j = obj_j->getCentroid();
      auto n_j = obj_j->getNormal(info);

      auto n_ij = normalize(p_j - p_i);

      // if (n_i*n_ij > 0 && n_j*n_ij < 0) {
      if (n_i*n_ij > 0) {
        std::cout << " 1";
      } else {
        std::cout << " 0";
      }
    }

    std::cout << std::endl;
  }
}
