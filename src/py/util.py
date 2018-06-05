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

def ray_intersects_box(p, n, xext, yext, zext):
    '''Takes a 3-vector `p' and a normal `n' and checks if the ray
emanating from `p' in direction of `n' intersects the axis-aligned
bounding box (AABB) defined by the extents `xext', `yext', and
`zext'.

    '''
    assert(p.size == 3 and n.size == 3)
    xmin, xmax = xext
    ymin, ymax = yext
    zmin, zmax = zext

    if n[0] != 0:
        t = (xmax - p[0])/n[0]
        if t >= 0:
            pt = p + t*n
            if ymin <= pt[1] and pt[1] <= ymax or \
               zmin <= pt[2] and pt[2] <= zmax:
                return True
        t = (xmin - p[0])/n[0]
        if t >= 0:
            pt = p + t*n
            if ymin <= pt[1] and pt[1] <= ymax or \
               zmin <= pt[2] and pt[2] <= zmax:
                return True

    if n[1] != 0:
        t = (ymax - p[1])/n[1]
        if t >= 0:
            pt = p + t*n
            if xmin <= pt[0] and pt[0] <= xmax or \
               zmin <= pt[2] and pt[2] <= zmax:
                return True
        t = (ymin - p[1])/n[1]
        if t >= 0:
            pt = p + t*n
            if xmin <= pt[0] and pt[0] <= xmax or \
               zmin <= pt[2] and pt[2] <= zmax:
                return True

    if n[2] != 0:
        t = (zmax - p[2])/n[2]
        if t >= 0:
            pt = p + t*n
            if xmin <= pt[0] and pt[0] <= xmax or \
               ymin <= pt[1] and pt[1] <= ymax:
                return True
        t = (zmin - p[2])/n[2]
        if t >= 0:
            pt = p + t*n
            if xmin <= pt[0] and pt[0] <= xmax or \
               ymin <= pt[1] and pt[1] <= ymax:
                return True

    return False

def ray_tri_intersection(p, n, v0, v1, v2, tol=np.finfo(np.float32).eps):
    '''Fast method to intersect a ray with a triangle, based on the paper
"Fast, Minimum Storage Ray/Triangle Intersection" by Moller and
Trumbore.

    '''
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(n, edge2)
    det = np.dot(edge1, pvec)
    if det > -tol and det < tol:
        return False
    inv_det = 1.0/det
    tvec = p - v0
    u = inv_det*np.dot(tvec, pvec)
    if u < 0 or u > 1:
        return False
    qvec = np.cross(tvec, edge1)
    v = inv_det*np.dot(n, qvec)
    if v < 0 or u + v > 1:
        return False
    t = inv_det*np.dot(edge2, qvec)
    return t
