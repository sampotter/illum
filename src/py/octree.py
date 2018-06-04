import numpy as np
import util

splits = [tuple(int(b) == 1 for b in bin(8 + i)[-3:]) for i in range(8)]

def bool2off(vec):
    return bin2off(map(bool, vec))

def bin2off(vec):
    return sum(b*2**-i/2 for i, b in enumerate(vec))

def scale(level):
    return 2**-level

def split2int(split):
    return np.uint8(int(''.join(list(map(str, map(int, split)))), base=2))

def get_offset(heap_inds):
    return np.array([
        bool2off(vec) for vec in list(zip(*[splits[k] for k in heap_inds]))])

def build_heap_level(X, Xscaled=None):
    heap_level = [[] for _ in range(8)]

    if Xscaled is None:
        Xsplits = [split2int(row > 0.5) for row in X]
    else:
        Xsplits = [split2int(row > 0.5) for row in Xscaled]

    for i, o in enumerate(Xsplits):
        heap_level[o].append(X[i])

    for o in range(8):
        if len(heap_level[o]) == 0:
            heap_level[o] = None
        else:
            heap_level[o] = np.array(heap_level[o])

    return heap_level

def build_octree_heap(X, lmax):
    heap = build_heap_level(X)

    def build_octree_heap_rec(leaf, inds, level):
        if level >= lmax:
            return
        for o, cell in enumerate(leaf):
            if cell is None:
                continue
            inds_ = inds[:] + [o]
            scaled = (cell - get_offset(inds_))/scale(level)
            assert(scaled.min() >= 0)
            assert(scaled.max() <= 1)
            leaf[o] = build_heap_level(cell, scaled)
            build_octree_heap_rec(leaf[o], inds_, level + 1)

    build_octree_heap_rec(heap, [], 1)

    return heap

# TODO: we will end up storing an octree embedded in a spherical
# shell. When we cast rays, we will need to map them into spherical
# coordinates, or intersect them with the warped cells of the octree

def leaves(h, inds=[], yield_inds=False):
    for o, cell in enumerate(h):
        if type(cell) == list:
            yield from leaves(cell, inds + [o], yield_inds)
        elif type(cell) == np.ndarray:
            if yield_inds:
                yield inds + [o], cell
            else:
                yield cell

class Octree(object):
    def __init__(self, data, lmax):
        self._data01 = util.scale01(data)
        self._heap = build_octree_heap(self._data01, lmax)

    def __iter__(self):
        return leaves(self._heap)

    def leaves(self, yield_inds=False):
        return leaves(self._heap, yield_inds)
