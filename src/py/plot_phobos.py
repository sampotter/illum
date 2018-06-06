# TODO: this is ridiculously slow... switch to Mayavi for this kind of
# thing

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

path = '../../data/Phobos_Ernst_decimated50k.obj'

v, f, vn, va = readobj(path)

tri = Triangulation(v, f, vn, va)

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f)

fig.show()
