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

_elt_dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('ind', 'i4')])

def build_octree_level(X, inds):
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

    octs = 4*splits[:, 0] + 2*splits[:, 1] + splits[:, 2]

    for i in range(8):
        sel = np.where(i == octs)[0]
        if len(sel) == 0:
            level[i] = (None, None)
        else:
            level[i] = (X[sel, :], inds[sel])

    return level

def build_octree_heap(X0, lmax):
    '''Build the heap for the octree.'''
    assert(lmax > 0)

    nrows = X0.shape[0]

    heap = build_octree_level(X0, np.arange(nrows, dtype=np.int32))

    def build_octree_heap_rec(level, l):
        if l >= lmax:
            return
        for i, (X, inds) in enumerate(level):
            if X is None:
                continue
            level[i] = build_octree_level(X, inds)
            build_octree_heap_rec(level[i], l + 1)

    build_octree_heap_rec(heap, 1)

    return heap

def leaves(h, inds=[], yield_inds=False):
    for i, cell in enumerate(h):
        if type(cell) == list:
            yield from leaves(cell, inds + [i], yield_inds)
        elif type(cell) == np.ndarray:
            if yield_inds:
                yield inds + [i]
            else:
                yield cell

class Octree(object):
    def __init__(self, data, lmax):
        self._data = data
        self._heap = build_octree_heap(self._data, lmax)
        self._lmax = lmax

    def __iter__(self):
        return leaves(self._heap)

    def __getitem__(self, key):
        if type(key) == int:
            key = (key,)
        assert(type(key) == tuple)
        assert(len(key) == self._lmax)
        i = 0
        node = self._heap
        while i < self._lmax and self._heap[key[i]] is not None:
            node = node[key[i]]
            i = i + 1
        return node

    def leaves(self, yield_inds=False):
        return leaves(self._heap, yield_inds=yield_inds)
