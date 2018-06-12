#!/usr/bin/env python3

import scipy.sparse

from mayavi import mlab
from obj import *

path = '../../data/num_vis.txt'

f = open(path, 'r')

S_dok = scipy.sparse.dok_matrix((50000, 50000), np.bool)

for line in f:
    i, j = tuple(map(int, line.split(',')))
    S_dok[i, j] = True
    S_dok[j, i] = True

S_csr = S_dok.tocsr()

nvis = np.array(np.sum(S_csr, 1)).flatten()
nvis /= nvis.max()

path = '../../data/Phobos_Ernst_decimated50k.obj'
verts, face, norms, albs = readobj(path)
x, y, z = v.T

mesh = mlab.triangular_mesh(x, y, z, face, representation='wireframe', opacity=0)
mesh.mlab_source.dataset.cell_data.scalars = nvis
mesh.mlab_source.dataset.cell_data.scalars.name = 'nvis'
mesh.mlab_source.update()
mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars='nvis')
surf = mlab.pipeline.surface(mesh2)
