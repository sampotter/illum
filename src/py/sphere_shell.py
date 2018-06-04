import matplotlib.pyplot as plt
import numpy as np

plt.ion()

def xyz2sph(xyz):
    assert(xyz.shape[1] == 3)
    sph = np.zeros_like(xyz)
    sph[:, 0] = np.sqrt(np.sum(xyz**2, 1))
    sph[:, 1] = np.arccos(xyz[:, 2]/sph[:, 0])
    sph[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return sph

def righthanded(v1, v2, v3):
    return np.dot(np.cross(v1, v2), v3) > 0

def horizon(v, f, vn, tri_ind):
    # order neighboring triangles in a right-handed fashion to find
    # the lower limit for the horizon

    # find the faces which have tri_ind as an incident vertex
    inds = np.where(f == tri_ind)[0]
    fs = f[inds]

    # simplify this to the unique set of neighboring vertices
    nb = np.unique(fs.flatten())
    nb = nb[nb != tri_ind]

    dvinds = np.zeros_like(fs)
    dvinds[fs == tri_ind] = -1
    for j, n in enumerate(nb):
        dvinds[fs == n] = j
    I, J = np.where(dvinds < np.uint16(-1))
    dvinds = dvinds[I, J].reshape(dvinds.shape[0], 2)

    v0 = v[tri_ind]
    dv = v[nb] - v0
    dv /= np.sqrt(np.sum(dv**2, 1))[:, np.newaxis]

    nrows = dv.shape[0]

    order = np.zeros(nrows, dtype=np.uint16)
    mask = np.zeros(nrows, dtype=np.bool)

    vn0 = vn[tri_ind]

    i0, i1 = dvinds[0]
    if righthanded(dv[i0], dv[i1], vn0):
        order[:2] = [i0, i1]
    else:
        order[:2] = [i1, i0]
    mask[0] = True

    for r in range(2, nrows):
        I = dvinds == order[r - 1]
        I = np.where(np.logical_and(
            np.logical_or(I[:, 0], I[:, 1]),
            np.logical_not(mask)))[0][0]
        mask[I] = True
        row = dvinds[I]
        order[r] = np.select(row != order[r - 1], row)

    angle = np.zeros(nrows)
    for j in range(1, nrows):
        n0 = np.cross(dv[order[j - 1]], vn0)
        n1 = np.cross(dv[order[j]], vn0)
        angle[j] = np.arccos(np.dot(n0, n1))
    angle = np.cumsum(angle)

    llim = np.zeros(nrows)
    for j in range(nrows):
        llim[j] = np.arccos(np.dot(dv[order[j]], vn0))

    return angle, llim, nb

def plot_horizon(angle0, llim0, angle1=None, llim1=None):
    cat = np.concatenate
    plt.figure()
    plt.plot([0, 2*np.pi], [np.pi/2, np.pi/2], linewidth=1, color='k')
    plt.plot(cat([angle0, [2*np.pi]]),
             cat([llim0, [llim0[0]]]), '*-b')
    if angle1 is not None and llim1 is not None:
        plt.plot(cat([angle1, [2*np.pi]]),
                 cat([llim1, [llim1[0]]]), '*-g')
    plt.xlim(0, 2*np.pi)
    plt.ylim(0, np.pi)
    plt.show()
    
