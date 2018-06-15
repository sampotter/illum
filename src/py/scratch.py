#!/usr/bin/env python3

import matplotlib.pyplot as plt

plt.ion()

import numpy as np
import scipy.sparse
import trimesh

from itertools import product
from mayavi import mlab

from sparse import *

tri = trimesh.load('../../data/SHAPE0_dec5000.obj')

F = tri.faces
V = tri.vertices

face_normals = np.zeros(F.shape)
face_centroids = np.zeros(F.shape)

for i, face in enumerate(F):
    v0, v1, v2 = V[face]
    face_centroids[i, :] = (v0 + v1 + v2)/3
    n = np.cross(v1 - v0, v2 - v0)
    n /= np.linalg.norm(n)
    if np.dot(n, tri.vertex_normals[face[0]]) < 0:
        n *= -1
    face_normals[i, :] = n

nfaces = F.shape[0]
nverts = V.shape[0]

N = face_normals
P = face_centroids

T = np.zeros_like(N)
B = np.zeros_like(N)
for i, face in enumerate(F):
    v0, v1, v2 = V[F[i]]
    T[i, :] = v1 - v0
    T[i, :] /= np.linalg.norm(T[i, :])
    B[i, :] = np.cross(T[i, :], N[i, :])

def get_frenet_frame(i):
    return np.column_stack([B[i], T[i], N[i]])

##############################
# Face visibility using radii
#

R = np.zeros(nfaces) # bounding radii
for i, p in enumerate(P):
    v0, v1, v2 = V[F[i]]
    R[i] = max(np.linalg.norm(v0 - p),
               np.linalg.norm(v1 - p),
               np.linalg.norm(v2 - p))
epsilon = R.max()

# A_zero = np.zeros((nfaces, nfaces), dtype=np.bool)
# A_eps = np.zeros((nfaces, nfaces), dtype=np.bool)
A_R = np.zeros((nfaces, nfaces), dtype=np.bool)
for j in range(nfaces):
    print(j)
    # A_zero[:, j] = (P - P[j, :])@N[j, :] > 0
    # A_eps[:, j] = (P - P[j, :])@N[j, :] > -epsilon
    A_R[:, j] = (P - P[j, :])@N[j, :] > -R

# V_zero = np.multiply(A_zero.T, A_zero)
# V_eps = np.multiply(A_eps.T, A_eps)
V_R = np.multiply(A_R.T, A_R)

################################################################################
# HORIZON MAPS

j = 830 # test a face
v_j = np.mean(V[F[j]], 0) # get centroid
BTN_j = get_frenet_frame(j)
I_vis = np.nonzero(V_after[:, j])[0]
F_vis = F[I_vis]
nfaces_vis = len(I_vis)

Phi = np.zeros(F_vis.shape, dtype=np.float32)
Theta = np.zeros(F_vis.shape, dtype=np.float32)

for i, face in enumerate(F_vis):

    D_i = (V[face] - v_j).T
    D_i /= np.sqrt(np.sum(D_i**2, 0))
    D_i = BTN_j.T@D_i

    Phi[i] = np.unwrap(np.arctan2(D_i[1, :], D_i[0, :]))
    Theta[i] = np.arccos(D_i[2, :])

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([-np.pi, np.pi], [np.pi/2, np.pi/2], 'k--', linewidth=1)

facecolors = []
edgecolors = []
patches = []
for i in range(nfaces_vis):
    xy = np.array([
        [Phi[i, 0], Theta[i, 0]],
        [Phi[i, 1], Theta[i, 1]],
        [Phi[i, 2], Theta[i, 2]]])
    facecolors.append((1.0, 0.5, 0.5, 0.5))
    edgecolors.append('k')
    patches.append(Polygon(xy, True))
coll = PatchCollection(patches, facecolors=facecolors, edgecolors=edgecolors)
ax.add_collection(coll)
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([0, np.pi])
ax.invert_yaxis()
plt.show()

################################################################################
# PLOT VISIBILITY FOR ONE FACE

j = 100
C = A_after[j, :].astype(np.float)
C[j] = -1

fig = mlab.gcf()
mlab.clf()

mesh = mlab.triangular_mesh(*V.T, F, representation='wireframe', opacity=0)
mesh.mlab_source.dataset.cell_data.scalars = C
mesh.mlab_source.dataset.cell_data.scalars.name = 'face_color'
mesh.mlab_source.update()
mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars='face_color')
surf = mlab.pipeline.surface(mesh2)

mlab.show()

################################################################################
# PLOT THE VISIBILITY MATRIX

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(V_R)
plt.show()

################################################################################
# Getting data from Armadillo

import h5py

A_before = csc_from_h5_file('../cpp/build/Release/A_before.h5', nfaces, nfaces, np.bool)
A_after = csc_from_h5_file('../cpp/build/Release/A_after.h5', nfaces, nfaces, np.bool)
V_arma = csc_from_h5_file('../cpp/build/Release/V.h5', nfaces, nfaces, np.bool)

fig = plt.figure()

fig.add_subplot(321).imshow(A_R)
fig.add_subplot(322).imshow(V_R)

fig.add_subplot(323).imshow(np.array(A_before.todense()))
fig.add_subplot(324).imshow(np.array(V_arma.todense()))

fig.add_subplot(325).imshow(np.array(A_after.todense()))

fig.show()
