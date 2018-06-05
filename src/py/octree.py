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

def get_octree_children(tri, l):
    '''Builds a level of an octree heap. The input `X' is an Nx3 numpy
array, where each row corresponds to a point in 3-space. The function
returns a length 8 list where each element is either None (indicating
the absence of any children) or an Mx3 numpy array (M < N) which is a
subarray of `X'.

    '''
    level = [[] for _ in range(8)]

    X = tri._verts

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
            inds = tri._inds[sel]
            level[i] = OctreeNode(tri.incident_faces(inds, sel=sel), l=l)

    return level

class Triangulation(object):
    def __init__(self, verts, faces, normals, albedos, inds=None):
        self._verts = verts
        self._faces = faces
        self._normals = normals
        self._albedos = albedos
        self._inds = np.arange(self.num_faces()) if inds is None else inds

    def num_faces(self):
        return self._faces.shape[0]

    def num_verts(self):
        return self._verts.shape[0]

    def incident_faces(self, inds, sel=None):
        # TODO: this isn't a great way to do this, but we'll use it
        # for now...
        mask = np.zeros((self.num_faces(), 3), dtype=np.bool)
        for ind in inds:
            mask = np.logical_or(mask, self._faces == ind)
        mask = np.where(np.logical_or(mask[:, 0], mask[:, 1], mask[:, 2]))[0]
        if sel is None:
            sel = inds
        return Triangulation(
            self._verts[sel],
            self._faces[mask, :],
            self._normals[sel],
            self._albedos[sel],
            inds=inds)

class OctreeNode(object):
    def __init__(self, tri, l=0):
        if l == 0:
            self._children = None
            self._tri = tri
        else:
            self._children = get_octree_children(tri, l - 1)
            self._tri = None
        self._extent = util.get_extent(tri._verts)

    def is_leaf_node(self):
        return self._children is None

    def leaves(self):
        if self.is_leaf_node():
            yield self
        else:
            for child in self._children:
                if child is not None:
                    yield from child.leaves()

    def __getitem__(self, *args):
        if type(args[0]) is int:
            return self._children[args[0]]
        else:
            inds = args[0]
            if len(inds) == 1:
                return self._children[inds[0]]
            return self._children[inds[0]].__getitem__(inds[1:])

    def ray_tri_intersections(self, p, n, tri):
        if ray_intersects_box(p, n, *self._extent):
            if self.is_leaf_node():
                for face in self._tri._faces:
                    t = ray_tri_intersection(p, n, *tri._verts[face])
                    if t >= 0:
                        yield face
            else:
                for node in self._children:
                    if node is not None:
                        yield from node.ray_tri_intersections(p, n, tri)

def default_lmax(tri):
    nfaces = tri._faces.shape[0]
    return int(np.round(np.log(nfaces)/np.log(8)))

class Octree(object):
    def __init__(self, tri, lmax=None):
        self._tri = tri
        self._lmax = default_lmax(tri) if lmax is None else lmax
        self._root = OctreeNode(tri, l=self._lmax)

    def leaves(self):
        return self._root.leaves()

    def __getitem__(self, *args):
        return self._root.__getitem__(*args)

    def ray_tri_intersections(self, p, n):
        return self._root.ray_tri_intersections(p, n, self._tri)
