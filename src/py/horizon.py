import matplotlib.pyplot as plt
import numpy as np

from mayavi.mlab import *
from mpl_toolkits.mplot3d import Axes3D
from numpy.polynomial.legendre import leggauss

from obj import *
from octree import *
from util import *

path = '../../data/Phobos_Ernst_decimated50k.obj'

v, f, vn, va = readobj(path)

tri = Triangulation(v, f, vn, va)

octree = Octree(tri)

oct_ind = (7, 3, 0, 4, 0)
node = octree[oct_ind]

# Pick a random face
face_ind = np.random.choice(node.tri.face_inds)

nphi = 16

# Phi and Theta parametrize the upper hemisphere from bottom to top

Phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
Theta = np.arccos(leggauss(int(nphi/2) + 1)[0][int(nphi/4):])

ntheta = len(Theta)

n = tri.face_normal(face_ind)
p = tri.face_centroid(face_ind)
p += np.finfo(np.float32).eps*n
if np.any(np.sign(p) != np.sign(n)): n = -n
t = tri.face_tangent(face_ind)
b = np.cross(t, n)
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

xlim = (p[0] - 1, p[0] + 1)
ylim = (p[1] - 1, p[1] + 1)
zlim = (p[2] - 1, p[2] + 1)

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)

ax = fig.add_subplot(1, 2, 2)

ax.imshow(vis)

fig.show()

v = tri.verts
f = tri.faces_in_extent((xlim, ylim, zlim))[0]
x, y, z = v.T
triangular_mesh(x, y, z, faces)
