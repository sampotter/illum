import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

from mpl_toolkits.mplot3d import Axes3D

from obj import *
from octree import *
from util import *

path = '../../data/Phobos_Ernst_decimated50k.obj'
v, f, vn, va = readobj(path)
tri = Triangulation(v, f, vn, va)

octree = Octree(tri)

nfaces = f.shape[0]

N = 1000

def face_data(i):
    v0, v1, v2 = v[f[i]]
    n = np.cross(v1 - v0, v2 - v0)
    return np.mean([v0, v1, 2], 0), n/np.linalg.norm(n)

print('default ordering...')

# Vis1 = scipy.sparse.dok_matrix((nfaces, nfaces), dtype=np.float32)
Vis1 = np.zeros((N, N), dtype=np.float32)

for i in range(N):
    print(i)

    pi, ni = face_data(i)

    for j in range(i):
        pj, nj = face_data(j)

        nij = pj - pi
        nij /= np.linalg.norm(nij)

        dotprod = ni@nij

        if dotprod > 0:
            Vis1[i, j] = dotprod
            Vis1[j, i] = dotprod

print ('octree ordering...')

Vis2 = np.zeros((N, N), dtype=np.float32)

I = []
# leaves = octree.leaves
while len(I) < N:
    I.extend(next(leaves).tri.face_inds)
            
for a, i in enumerate(I[:N]):
    print(i)

    pi, ni = face_data(i)

    for b, j in enumerate(I[:a]):
        pj, nj = face_data(j)

        nij = pj - pi
        nij /= np.linalg.norm(nij)

        dotprod = ni@nij

        if dotprod > 0:
            Vis2[a, b] = dotprod
            Vis2[b, a] = dotprod

fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)
im = ax.imshow(Vis1)
fig.colorbar(im, ax=ax)

ax = fig.add_subplot(1, 2, 2)
im = ax.imshow(Vis2)
fig.colorbar(im, ax=ax)

fig.show()
