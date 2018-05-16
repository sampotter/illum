import numpy as np
import util

splits = [list(int(b) == 1 for b in bin(8 + i)[-3:]) for i in range(8)]

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

def build_heap_level(X):
    heap_level = [[] for _ in range(8)]
    for i, o in enumerate([split2int(row > 0.5) for row in X]):
        heap_level[o].append(X[i])
    for o in range(8):
        if len(heap_level[o]) == 0:
            heap_level[o] = None
        else:
            heap_level[o] = np.array(heap_level[o])
    return heap_level

def add_heap_level(heap):
    for o in range(8):
        scaled = 
        heap[o] = build_heap_level(heap[o])

class Octree(object):
    def __init__(self, data, levels):
        self._data01 = util.scale01(data)
        self._heap = [[] for _ in range(8)]
