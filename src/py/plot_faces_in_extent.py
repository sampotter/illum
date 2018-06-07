from mayavi.mlab import *

from obj import *
from octree import *
from util import *

path = '../../data/Phobos_Ernst_decimated50k.obj'

v, f, vn, va = readobj(path)

tri = Triangulation(v, f, vn, va)

extent = get_extent(v)

faces, face_inds = tri.faces_in_extent(subextent(1, *extent))

triangular_mesh(v[:, 0], v[:, 1], v[:, 2], faces)
