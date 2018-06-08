import numpy as np
import util

splits = [tuple(int(b) == 1 for b in bin(8 + i)[-3:]) for i in range(8)]

def bool2off(vec):
    return bin2off(map(bool, vec))

def bin2off(vec):
    return sum(b*2**-i/2 for i, b in enumerate(vec))

def split2int(split):
    '''A `split' is a length 3 boolean vector (with components
corresponding to (x, y, z) values) which indicates whether each
component of a vector is on one or another side of an AABB split. This
function converts that boolean vector into a linear index to be used
to index into an 8-way heap.

    '''
    return np.uint8(int(''.join(list(map(str, map(int, split)))), base=2))

def get_offset(inds):
    return np.array([
        bool2off(vec) for vec in list(zip(*[splits[k] for k in inds]))])

def get_octree_children(parent, tri, base_tri, thresh):
    '''Builds a level of an octree heap. The input `X' is an Nx3 numpy
array, where each row corresponds to a point in 3-space. The function
returns a length 8 list where each element is either None (indicating
the absence of any children) or an Mx3 numpy array (M < N) which is a
subarray of `X'.

    '''
    level = [[] for _ in range(8)]

    X = tri.verts
    xext, yext, zext = util.get_extent(X)
    splits = np.column_stack([
        np.array(X[:, 0] > (xext[1] + xext[0])/2, dtype=np.uint8),
        np.array(X[:, 1] > (yext[1] + yext[0])/2, dtype=np.uint8),
        np.array(X[:, 2] > (zext[1] + zext[0])/2, dtype=np.uint8)])
    oct_inds = 4*splits[:, 0] + 2*splits[:, 1] + splits[:, 2]

    for i in range(8):
        sel = np.where(i == oct_inds)[0]
        if len(sel) == 0:
            level[i] = None
        else:
            subext = util.subextent(i, xext, yext, zext)
            faces, face_inds = tri.faces_in_extent(subext, base_tri)
            subtri = Triangulation(
                tri.verts[sel],
                faces,
                tri.normals[sel],
                tri.albedos[sel],
                face_inds=face_inds,
                vert_inds=tri.vert_inds[sel])
            level[i] = OctreeNode(parent, subtri, thresh)

    return level

class Triangulation(object):
    def __init__(self, verts, faces, normals, albedos, face_inds=None,
                 vert_inds=None):
        self._verts = verts
        self._faces = faces
        self._normals = normals
        self._albedos = albedos
        self._face_inds = np.arange(self.num_faces) \
            if face_inds is None else face_inds
        self._vert_inds = np.arange(self.num_verts) \
            if vert_inds is None else vert_inds

    def get_verts(self):
        return self._verts

    verts = property(get_verts)

    def get_faces(self):
        return self._faces

    faces = property(get_faces)

    def get_normals(self):
        return self._normals

    normals = property(get_normals)

    def get_albedos(self):
        return self._albedos

    albedos = property(get_albedos)

    def get_face_inds(self):
        return self._face_inds

    face_inds = property(get_face_inds)

    def get_num_faces(self):
        return self._faces.shape[0]

    num_faces = property(get_num_faces)

    def get_vert_inds(self):
        return self._vert_inds

    vert_inds = property(get_vert_inds)

    def get_num_verts(self):
        return self._verts.shape[0]

    num_verts = property(get_num_verts)

    def face_centroid(self, face_ind):
        face = self.faces[face_ind]
        return np.mean(self.verts[face], 0)

    def face_normal(self, face_ind):
        v0, v1, v2 = self.verts[self.faces[face_ind]]
        n = np.cross(v1 - v0, v2 - v0)
        return n/np.linalg.norm(n)

    def face_tangent(self, face_ind):
        face = self.faces[face_ind]
        t = self.verts[face][1] - self.verts[face][0]
        t /= np.linalg.norm(t)
        return t

    def faces_in_extent(self, extent, base_tri=None):
        if base_tri is None:
            base_tri = self

        faces, face_inds = [], []
        mins, maxs = np.array(extent).T

        for face, face_ind in zip(self.faces, self.face_inds):
            v = base_tri.verts[face]
            if np.any(np.all(np.logical_and(mins <= v, v <= maxs), 1)):
                faces.append(face)
                face_inds.append(face_ind)
        return \
            np.array(faces, dtype=np.int32), \
            np.array(face_inds, dtype=np.int32)

def ray_intersects_octree_node(p, n, node):
    if node is None:
        return False
    else:
        return util.ray_intersects_box(p, n, *node.extent)

class OctreeNode(object):
    def __init__(self, parent, tri, thresh):
        self._parent = parent
        print(tri.num_faces)
        if tri.num_faces <= thresh:
            self._children = None
            self._tri = tri
        else:
            base_tri = self.containing_octree.tri
            self._children = get_octree_children(self, tri, base_tri, thresh)
            self._tri = None
        self._extent = util.get_extent(tri._verts)

    def get_containing_octree(self):
        return self.root.parent

    containing_octree = property(get_containing_octree)

    def get_parent(self):
        return self._parent

    parent = property(get_parent)

    def get_root(self):
        if type(self.parent) is Octree:
            return self
        else:
            return self.parent.get_root()

    root = property(get_root)
        
    def get_tri(self):
        return self._tri

    tri = property(get_tri)

    def get_extent(self):
        return self._extent

    extent = property(get_extent)

    def is_leaf_node(self):
        return self._children is None

    def get_leaves(self):
        if self.is_leaf_node():
            yield self
        else:
            for child in self._children:
                if child is not None:
                    yield from child.leaves

    leaves = property(get_leaves)

    def __getitem__(self, *args):
        if type(args[0]) is int:
            return self._children[args[0]]
        else:
            inds = args[0]
            if len(inds) == 1:
                return self._children[inds[0]]
            return self._children[inds[0]].__getitem__(inds[1:])

    def ray_tri_intersections(self, p, n, tri):
        if util.ray_intersects_box(p, n, *self._extent):
            if self.is_leaf_node():
                for face in self._tri._faces:
                    t = util.ray_tri_intersection(p, n, *tri._verts[face])
                    if t >= 0:
                        yield face
            else:
                for node in self._children:
                    if node is not None:
                        yield from node.ray_tri_intersections(p, n, tri)

    def get_nodes_containing_vertex(self, p):
        if extent_contains_point(p, *self.extent):
            if self.is_leaf_node():
                yield self
            else:
                for child in self._children:
                    if child is not None:
                        yield from child.get_node_containing_vertex(p)

def default_thresh(tri):
    return np.ceil(np.log(tri.num_faces)/np.log(8))

class Octree(object):
    def __init__(self, tri, thresh=None):
        self._tri = tri
        self._thresh = default_thresh(tri) if thresh is None else thresh
        print(self._thresh)
        self._root = OctreeNode(self, tri, thresh=self._thresh)

    def get_tri(self):
        return self._tri

    tri = property(get_tri)

    def get_extent(self):
        return self._root._extent

    extent = property(get_extent)

    def get_leaves(self):
        return self._root.leaves

    leaves = property(get_leaves)

    def get_center(self):
        return np.array([np.mean(ext) for ext in self.extent])

    center = property(get_center)

    def get_corners(self):
        return \
            np.array([ext[0] for ext in octree.extent]), \
            np.array([ext[1] for ext in octree.extent])

    def __getitem__(self, *args):
        return self._root.__getitem__(*args)

    def ray_tri_intersections(self, p, n):
        return self._root.ray_tri_intersections(p, n, self._tri)

    def get_nodes_containing_vertex(self, p):
        return self._root.get_node_containings_vertex(p)
