#!/usr/bin/env python3

# import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import trimesh

from itertools import product
from mayavi import mlab

tri = trimesh.load('../../data/SHAPE0_dec5000.obj')

face_normals = np.zeros(tri.faces.shape)
face_centroids = np.zeros(tri.faces.shape)

for i, face in enumerate(tri.faces):
    v0, v1, v2 = tri.vertices[face]
    face_centroids[i, :] = (v0 + v1 + v2)/3
    n = np.cross(v1 - v0, v2 - v0)
    n /= np.linalg.norm(n)
    if np.dot(n, tri.vertex_normals[face[0]]) < 0:
        n *= -1
    face_normals[i, :] = n

nfaces = tri.faces.shape[0]

A_eps = np.zeros((5000, 5000), dtype=np.bool)
A_R = np.zeros((5000, 5000), dtype=np.bool)
N = face_normals
P = face_centroids

T = np.zeros_like(N)
B = np.zeros_like(N)
for i, face in enumerate(tri.faces):
    v0, v1, v2 = tri.vertices[tri.faces[i]]
    T[i, :] = v1 - v0
    T[i, :] /= np.linalg.norm(T[i, :])
    B[i, :] = np.cross(T[i, :], N[i, :])

def tnb(i):
    return np.column_stack([T[i, :], N[i, :], B[i, :]])

def xyz2sph(v):
    return np.array([np.arccos(v[2]/np.linalg.norm(v)), np.arctan2(v[1], v[0])])

# Compute tri "radii"
R = np.zeros(nfaces)
for i, p in enumerate(P):
    v0, v1, v2 = tri.vertices[tri.faces[i]]
    R[i] = max(np.linalg.norm(v0 - p),
               np.linalg.norm(v1 - p),
               np.linalg.norm(v2 - p))

epsilon = R.max()

for face_ind in range(nfaces):
    print(face_ind)
    
    # C = np.zeros(nfaces)

    # D = P - P[face_ind, :]
    # Dnorm = np.sqrt(np.sum(D*D, 1)) # can remove?
    # I = np.where(Dnorm > 0)[0]
    # for i in range(D.shape[1]):
    #     D[I, i] /= Dnorm[I]

    # A[:, face_ind] = D@N[face_ind, :] > -epsilon

    A_eps[:, face_ind] = (P - P[face_ind, :])@N[face_ind, :] > -epsilon
    A_R[:, face_ind] = (P - P[face_ind, :])@N[face_ind, :] > -R

V_eps = np.multiply(A_eps.T, A_eps) # SCALAR product!
V_R = np.multiply(A_R.T, A_R)

# PLOT VISIBILITY FOR ONE FACE

# faceind = 1
# C = V[:, faceind].astype(np.float32)
# C[faceind] = -1
# mesh = mlab.triangular_mesh(*tri.vertices.T, tri.faces,
#                             representation='wireframe', opacity=0)
# mesh.mlab_source.dataset.cell_data.scalars = C
# mesh.mlab_source.dataset.cell_data.scalars.name = 'face_color'
# mesh.mlab_source.update()
# mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars='face_color')
# surf = mlab.pipeline.surface(mesh2)

# PLOT THE VISIBILITY MATRIX

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(V_eps)
# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(V_R)
# plt.show()
