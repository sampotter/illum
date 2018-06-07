from mayavi.mlab import *

from obj import *
from octree import *
from util import *

path = '../../data/Phobos_Ernst_decimated50k.obj'

v, f, vn, va = readobj(path)

tri = Triangulation(v, f, vn, va)

x, y, z = v.T
nrows = f.shape[0]

src = pipeline.triangular_mesh_source(x, y, z, f)
src.data.cell_data.scalars = np.linspace(0, 1, nrows)
surf = pipeline.surface(src)
surf.contour.filled_contours = True
