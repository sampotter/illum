#!/usr/bin/env python3

import matplotlib.pyplot as plt

plt.ion()

import h5py
import numba
import numpy as np
import scipy.sparse
import trimesh

from itertools import combinations, product
from mayavi import mlab

import points

from sparse import *

################################################################################
# IMPORT GEOMETRY

# tri = trimesh.load('../../data/SHAPE0_dec5000.obj')
tri = trimesh.load('../../data/SHAPE0.OBJ')
F = tri.faces
V = tri.vertices
nfaces = F.shape[0]
nverts = V.shape[0]

def compute_face_normals_and_centroids():
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
    return face_normals, face_centroids

print('- computing faces normals and centroids')
N, P = compute_face_normals_and_centroids()

def compute_face_tangents_and_bivectors():
    T = np.zeros_like(N)
    B = np.zeros_like(N)
    for i, face in enumerate(F):
        v0, v1, v2 = V[F[i]]
        T[i, :] = v1 - v0
        T[i, :] /= np.linalg.norm(T[i, :])
        B[i, :] = np.cross(T[i, :], N[i, :])
    return T, B

print('- compute face tangents and bivectors')
T, B = compute_face_tangents_and_bivectors()

def get_frenet_frame(i, order='btn'):
    assert(len(order) == 3)
    assert(set(order) == {'t', 'n', 'b'})
    cols = {'t': T[i], 'n': N[i], 'b': B[i]}
    return np.column_stack([cols[order[0]], cols[order[1]], cols[order[2]]])

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
# GETTING DATA FROM ARMADILLO

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

################################################################################
# HORIZON MAPS

j = np.random.randint(nfaces) # test a face
v_j = np.mean(V[F[j]], 0) # get centroid
BTN_j = get_frenet_frame(j)
I_vis = np.nonzero(V_arma[:, j])[0]
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

j = np.random.randint(nfaces)
C = np.array(A_after.getrow(j).todense(), dtype=np.float).flatten()
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
# BUILD LINK MATRIX

def nchoose2(n): return int(n*(n - 1)/2)

adj_list = [np.nonzero(V_arma[:, j])[0] for j in range(nfaces)]

nlinks = sum(nchoose2(len(lst)) for lst in adj_list)

edge_inds = dict()
link_inds = dict()

edge_ind = 0
link_ind = 0
for j, lst in enumerate(adj_list):
    print(j)
    for i, k in combinations(lst, 2):
        # Add link
        i, k = sorted((i, k))
        key = (i, j, k)
        if key not in link_inds:
            link_inds[key] = link_ind
            link_ind += 1
        # Add edges
        key = tuple(sorted((i, j)))
        if key not in edge_inds:
            edge_inds[key] = edge_ind
            edge_ind += 1
        key = tuple(sorted((j, k)))
        if key not in edge_inds:
            edge_inds[key] = edge_ind
            edge_ind += 1

nedges = len(edge_inds)

sparsity = nlinks/(nedges**2)
print('sparsity = %g' % sparsity)

Links = np.zeros((nedges, nedges), dtype=np.bool)
for i, j, k in link_inds.keys():
    key1 = tuple(sorted((i, j)))
    key2 = tuple(sorted((j, k)))
    Links[edge_inds[key1], edge_inds[key2]] = True
    Links[edge_inds[key2], edge_inds[key1]] = True

################################################################################
# PLOT HORIZONS

f = h5py.File('../cpp/build/Release/horizons.h5')
H = f['horizons'][:]
f.close()

Phi = np.linspace(0, 2*np.pi, H.shape[1])

fig = plt.figure()

def plot_horizon(subplot, index):
    ax = fig.add_subplot(subplot)
    ax.plot([0, 2*np.pi], [np.pi/2, np.pi/2], 'k-.')
    ax.plot(Phi, H[index, :])
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, np.pi)
    ax.invert_yaxis()
    return ax

plot_horizon(221, np.random.randint(nfaces))
plot_horizon(222, np.random.randint(nfaces))
plot_horizon(223, np.random.randint(nfaces))
plot_horizon(224, np.random.randint(nfaces))

fig.show()

################################################################################
# COMPUTE COSINES ON PHOBOS

# create a random but plausible location for the sun

def check_visibility(i, p_sun, r_sun):

    p = P[i, :]

    # we want to model the sun as a disk: create the plane the disk lies
    # in first

    n_sun_plane = p_sun - p.reshape(3, 1)
    n_sun_plane /= np.linalg.norm(n_sun_plane)
    n_sun_plane = n_sun_plane.flatten()

    t_sun_plane = np.random.randn(3, 1)
    t_sun_plane = (np.eye(3) - n_sun_plane@n_sun_plane.T)@t_sun_plane
    t_sun_plane = t_sun_plane.flatten()

    n_sun_plane = n_sun_plane.flatten()

    b_sun_plane = np.cross(t_sun_plane, n_sun_plane)

    # sample a uniform random distribution of points on the disk
    # TODO: we can probably do much better than this

    X, Y = points.fibonacci_spiral(50)

    Xt = X.reshape(X.size, 1)*t_sun_plane
    Yt = Y.reshape(Y.size, 1)*b_sun_plane

    disk = p_sun.T + r_sun*(Xt + Yt)

    BTN = get_frenet_frame(i, order='btn')

    dirs = disk - p
    dirs /= np.sqrt(np.sum(dirs**2, 1)).reshape(dirs.shape[0], 1)
    dirs = dirs@BTN

    dirs_Theta = np.arccos(dirs[:, 2])
    dirs_Phi = np.mod(np.arctan2(dirs[:, 1], dirs[:, 0]), 2*np.pi)

    # check which points are above the horizon

    H_Theta = H[i, :]
    nPhi = H_Theta.size
    H_Phi = np.linspace(0, 2*np.pi, nPhi)
    deltaPhi = 2*np.pi/(nPhi - 1)
    I = np.floor(dirs_Phi/deltaPhi).astype(np.uint16)
    t = (dirs_Phi - H_Phi[I])/(H_Phi[I + 1] - H_Phi[I])
    AboveH = dirs_Theta < (1 - t)*H_Theta[I] + t*H_Theta[I + 1]
    Ratio = AboveH.sum()/AboveH.size

    return Ratio, AboveH, dirs_Phi, dirs_Theta


# position the sun randomly

d_sun = 227390024000 # m
diam_sun = 1391400000 # m
r_sun = diam_sun/2 # m

p_sun = np.random.randn(3, 1)
n_sun = p_sun/np.linalg.norm(p_sun)
p_sun = d_sun*n_sun

# plot horizon w/ sun

i = np.random.randint(nfaces)
Ratio, AboveH, dirs_Phi, dirs_Theta = check_visibility(i, p_sun, r_sun)

fig = plt.figure()
ax = plot_horizon(111, i)
ax.scatter(dirs_Phi[AboveH], dirs_Theta[AboveH], 1, 'r')
ax.scatter(dirs_Phi[~AboveH], dirs_Theta[~AboveH], 1, 'b')
fig.show()

# plot ratios on surface

Ratios = np.zeros(nfaces)
for i in range(nfaces):
    print(i)
    Ratio, AboveH, dirs_Phi, dirs_Theta = check_visibility(i, p_sun, r_sun)
    Ratios[i] = Ratio

Sun_normals = p_sun.T - P
Sun_normals /= np.sqrt(np.sum(Sun_normals**2, 1)).reshape(nfaces, 1)
Cosines = np.sum(Sun_normals*N, 1)

C = Ratios*Cosines

fig = mlab.figure(bgcolor=(0, 0, 0))

mesh = mlab.triangular_mesh(*V.T, F, representation='wireframe', opacity=0)
mesh.mlab_source.dataset.cell_data.scalars = C
mesh.mlab_source.dataset.cell_data.scalars.name = 'face_color'
mesh.mlab_source.update()
mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars='face_color')
surf = mlab.pipeline.surface(mesh2, colormap='gray')

mlab.show()
