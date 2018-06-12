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

    Vector3 n;
    if (i0.normal_index >= 0) {
      n = Vector3 {
        normals[3*i0.normal_index],
        normals[3*i0.normal_index + 1],
        normals[3*i0.normal_index + 2]
      };
    } else {
      n = normalize((v1 - v0)^(v2 - v0));
      Vector3 n0 {
        normals[3*i0.vertex_index],
        normals[3*i0.vertex_index + 1],
        normals[3*i0.vertex_index + 2]
      };
      if (n*n0 < 0) {
        n = -n;
      }
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

double
get_bounding_radius(Tri const * tri) {
  auto p = tri->getCentroid();
  return std::max(
    length(tri->v0 - p),
    std::max(
      length(tri->v1 - p),
      length(tri->v2 - p)));
}

std::vector<double>
get_bounding_radii(std::vector<Object *> const & objects) {
  std::vector<double> R;
  auto const nfaces = objects.size();
  R.reserve(nfaces);
  for (auto const * obj: objects) {
    R.push_back(get_bounding_radius(static_cast<Tri const *>(obj)));
  }
  return R;
}

void
build_A_using_bounding_radii(std::vector<Object *> const & objects,
                             arma::sp_umat & A)
{
  assert(A.is_empty());

  auto nfaces = objects.size();
  A.set_size(nfaces, nfaces);

  auto const R = get_bounding_radii(objects);

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
      if (lhs > rhs - R[i]) {
        A(i, j) = true;
      }
    }
  }
}

void
compute_V(arma::sp_umat const & A, arma::sp_umat & V) {
  V = A.t()%A;
}

std::pair<arma::uvec, arma::uvec>
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
    p_i = p_i + 1e-7*n_i; // perturb to avoid self-intersection

    std::vector<arma::uword> hits;

    for (int j = 0; j < objects.size(); ++j) {
      if (!V(i, j)) {
        continue;
      }
      
      auto * obj_j = objects[j];
      auto p_j = obj_j->getCentroid();
      auto n_ij = normalize(p_j - p_i);

      /**
       * Cast a ray to each "facing" face and store the index of the
       * face that was actually hit in `hits'.
       */
      if (bvh.getIntersection(Ray(p_i, n_ij), &info, false)) {
        auto index = static_cast<Tri const *>(info.object)->index;
        auto lb = std::lower_bound(hits.begin(), hits.end(), index);
        if (lb == hits.end() || *lb != index) {
          hits.insert(lb, index);
        }
      }
    }

    col += hits.size();
    colptr_vec.push_back(col);

    rowind_vec.insert(rowind_vec.end(), hits.begin(), hits.end());
  }

  arma::uvec rowind(rowind_vec), colptr(colptr_vec);
  // arma::uvec values(rowind.n_elem, arma::fill::ones);
  // A = arma::sp_umat(rowind, colptr, values, V.n_rows, V.n_cols);

  return {rowind, colptr};
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

  /**
   * Build the BVH.
   */
  BVH bvh(&objects);

  //////////////////////////////////////////////////////////////////////////////
  // sandbox
  //

  arma::sp_umat A;
  // build_A_zero(objects, A);
  build_A_using_bounding_radii(objects, A);

  arma::sp_umat V;
  compute_V(A, V);

  A.reset();

  arma::uvec rowind, colptr;
  std::tie(rowind, colptr) = fix_visibility(bvh, objects, V);
  
  rowind.save(arma::hdf5_name("out.h5", "rowind", arma::hdf5_opts::append));
  colptr.save(arma::hdf5_name("out.h5", "colptr", arma::hdf5_opts::append));
}
