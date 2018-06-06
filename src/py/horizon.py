import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from obj import *
from octree import *
from util import *

path = '../../data/Phobos_Ernst_decimated50k.obj'

v, f, vn, va = readobj(path)

tri = Triangulation(v, f, vn, va)

octree = Octree(tri, lmax=2)

oct_ind = (0, 0)
node = octree[oct_ind]

# Pick a random face
face = node.tri.faces[np.random.randint(node.tri.faces.shape[0])]

nphi = 100

# Phi and Theta parametrize the upper hemisphere from bottom to top

Phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
# Theta = np.linspace(0, np.pi/2, ntheta, endpoint=True)
Theta = np.arccos(np.polynomial.legendre.leggauss(int(nphi/2) + 1)[0][int(nphi/4):])

ntheta = len(Theta)

def face_centroid(face):
    return np.mean(v[face], 0)

def face_normal(face):
    v0, v1, v2 = v[face]
    n = np.cross(v1 - v0, v2 - v0)
    return n/np.linalg.norm(n)

def face_tangent(face):
    t = v[face][1] - v[face][0]
    t /= np.linalg.norm(t)
    return t

n = face_normal(face)

p = face_centroid(face)
p += np.finfo(np.float32).eps*n

if np.any(np.sign(p) != np.sign(n)):
    n = -n

t = face_tangent(face)

# Get bivector
b = np.cross(t, n)

# Build matrix for Frenet frame
F = np.array([t, b, n])

def get_normal(phi, theta):
    return np.array([
        np.cos(phi)*np.sin(theta),
        np.sin(phi)*np.sin(theta),
        np.cos(theta)])

Ns = np.array([F.T@get_normal(Phi[j], Theta[i])
               for i in range(ntheta)
               for j in range(nphi)])

vis = np.zeros((ntheta, nphi), dtype=np.bool)

def ray_escapes(phi, theta):
    return next(
        octree.ray_tri_intersections(p, F.T@get_normal(phi, theta)),
        None) is None

def make_pred(phi):
    return lambda theta: ray_escapes(phi, theta)

for j, phi in enumerate(Phi):
    print('j = %d, phi = %g' % (j, phi))
    # i = find_first_true(Theta, make_pred(phi))
    # vis[:i, j] = True
    vis[:, j] = [ray_escapes(phi, theta) for theta in Theta]
vis = np.flipud(vis)

fig = plt.figure()

ax = fig.add_subplot(1, 2, 1, projection='3d')

# ax = Axes3D(fig, ax=axes[0])

ax.scatter(p[0] + Ns[:, 0], p[1] + Ns[:, 1], p[2] + Ns[:, 2],
           facecolors='none', edgecolors='black')

ax.quiver(*p, *n, color='red')
ax.quiver(*p, *t, color='green')
ax.quiver(*p, *b, color='yellow')

faces = node.tri.faces
inds = np.unique(faces)
verts = tri.verts[inds]

reindex = {ind: i for i, ind in enumerate(inds)}

M, N = faces.shape
faces = faces.flatten()
for i in range(faces.size):
    faces[i] = reindex[faces[i]]
faces = faces.reshape(M, N)

ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces)

ax.set_xlim([p[0] - 1, p[0] + 1])
ax.set_ylim([p[1] - 1, p[1] + 1])
ax.set_zlim([p[2] - 1, p[2] + 1])

ax = fig.add_subplot(1, 2, 2)

ax.imshow(vis)

fig.show()
