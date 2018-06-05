import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from obj import *
from octree import *
from util import *

path = '../../data/Phobos_Ernst_decimated50k.obj'

v, f, vn, va = readobj(path)

tri = Triangulation(v, f, vn, va)

octree = Octree(tri)

face_ind = 100

nphi = 20
ntheta = 11

# Phi and Theta parametrize the upper hemisphere from bottom to top

Phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
Theta = np.linspace(0, np.pi/2, ntheta, endpoint=True)

def face_centroid(i):
    return np.mean(v[f[i]], 0)

def face_normal(i):
    v0, v1, v2 = v[f[i]]
    n = np.cross(v1 - v0, v2 - v0)
    return n/np.linalg.norm(n)

n = face_normal(face_ind)

p = face_centroid(face_ind)
p += np.finfo(np.float32).eps*n

if np.any(np.sign(p) != np.sign(n)):
    n = -n

# Get an arbitrary face tangent vector
t = v[f[face_ind][1]] - v[f[face_ind][0]]
t /= np.linalg.norm(t)

# Get bivector
b = np.cross(t, n)

# Build matrix for Frenet frame
F = np.array([t, b, n])

def get_normal(phi, theta):
    return np.array([
        np.cos(phi)*np.sin(theta),
        np.sin(phi)*np.sin(theta),
        np.cos(theta)])

vis = np.zeros((ntheta, nphi), dtype=np.bool)

def ray_escapes(phi, theta):
    return next(
        octree.ray_tri_intersections(p, F.T@get_normal(phi, theta)),
        None) is None

def make_pred(phi):
    return lambda theta: ray_escapes(phi, theta)

# for j, phi in enumerate(Phi):
#     print('j = %d, phi = %g' % (j, phi))
#     # i = find_first_true(Theta, make_pred(phi))
#     # vis[:i, j] = True
#     vis[:, j] = [ray_escapes(phi, theta) for theta in Theta]
# vis = np.flipud(vis)

Ns = np.array([F.T@get_normal(Phi[j], Theta[i])
               for i in range(ntheta)
               for j in range(nphi)])

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(p[0] + Ns[:, 0], p[1] + Ns[:, 1], p[2] + Ns[:, 2],
           facecolors='none', edgecolors='black')

ax.scatter(*(p + F.T@get_normal(Phi[0], Theta[0])), color='red')
ax.scatter(*(p + F.T@get_normal(Phi[0], Theta[-1])), color='green')
ax.scatter(*(p + F.T@get_normal(Phi[int(nphi/4)], Theta[-1])), color='yellow')

ax.quiver(*p, *n, color='red')
ax.quiver(*p, *t, color='green')
ax.quiver(*p, *b, color='yellow')

ax.set_xlim([p[0] - 1, p[0] + 1])
ax.set_ylim([p[1] - 1, p[1] + 1])
ax.set_zlim([p[2] - 1, p[2] + 1])

fig.show()
