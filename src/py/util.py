import numpy as np

def scale01(X):
    return (X - np.min(X, 0))/(np.max(X, 0) - np.min(X, 0))

def get_extent(arr):
    '''Take an Nx3 array of points (a list of 3-vectors) and return the
extent (the interval along each dimension) of the smallest AABB that
contains them.

    '''
    xmin, xmax = arr[:, 0].min(), arr[:, 0].max()
    ymin, ymax = arr[:, 1].min(), arr[:, 1].max()
    zmin, zmax = arr[:, 2].min(), arr[:, 2].max()
    return (xmin, xmax), (ymin, ymax), (zmin, zmax)
