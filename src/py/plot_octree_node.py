from mayavi.mlab import *

from obj import *
from octree import *
from util import *

path = '../../data/Phobos_Ernst_decimated50k.obj'

v, f, vn, va = readobj(path)

tri = Triangulation(v, f, vn, va)

octree = Octree(tri, lmax=2)

node_tri = octree[1, 0].tri
x, y, z = v.T
faces = node_tri.faces

triangular_mesh(x, y, z, faces)
