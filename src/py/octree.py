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

def get_octree_children(X, inds, l):
    '''Builds a level of an octree heap. The input `X' is an Nx3 numpy
array, where each row corresponds to a point in 3-space. The function
returns a length 8 list where each element is either None (indicating
the absence of any children) or an Mx3 numpy array (M < N) which is a
subarray of `X'.

    '''
    level = [[] for _ in range(8)]

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
            level[i] = OctreeNode(X[sel, :], inds=inds[sel], l=l)

    return level

class Triangulation(object):
    def __init__(self, verts, faces, normals, albedos):
        self._verts = verts
        self._faces = faces
        self._normals = normals
        self._albedos = albedos

class OctreeNode(object):
    def __init__(self, X, inds=None, l=0):
        if inds is None:
            inds = np.arange(X.shape[0], dtype=np.int32)
        if l == 0:
            self._children = None
            self._X = X
            self._inds = inds
        else:
            self._children = get_octree_children(X, inds, l - 1)
            self._X = None
            self._inds = None
            
        self._extent = util.get_extent(X)

    def __getitem__(self, *args):
        print(args)
        if type(args[0]) is int:
            return self._children[args[0]]
        else:
            inds = args[0]
            if len(inds) == 1:
                return self._children[inds[0]]
            return self._children[inds[0]].__getitem__(inds[1:])
