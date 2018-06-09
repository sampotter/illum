#!/usr/bin/env python3

import numpy as np
import scipy.sparse
import trimesh

from mayavi import mlab

num_facing = np.loadtxt('../../data/num_facing.txt')

pairs = np.loadtxt('../../data/pairs.txt')

N = 5000
S_dok = scipy.sparse.dok_matrix((N, N), np.bool)
for i, j in pairs:
    S_dok[i, j] = True
S_csr = S_dok.tocsr()
I_nz = np.where(np.sum(S_csr, 1) > 0)[0] # rows w/ nonzero entries
i_nz = I_nz[1]

faceind = i_nz
facesel = np.array(S_csr[i_nz].todense()).flatten()
facesel[faceind] = -1

tri = trimesh.load('../../data/SHAPE0_dec5000.obj')

mesh = mlab.triangular_mesh(*tri.vertices.T, tri.faces,
                            representation='wireframe', opacity=0)
# mesh.mlab_source.dataset.cell_data.scalars = num_facing
mesh.mlab_source.dataset.cell_data.scalars = facesel
mesh.mlab_source.dataset.cell_data.scalars.name = 'num_facing'
mesh.mlab_source.update()
mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars='num_facing')
surf = mlab.pipeline.surface(mesh2)
