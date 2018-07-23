#!/usr/bin/env python3

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('scalar_field_path', type=str)
    parser.add_argument('obj_path', type=str)
    args = parser.parse_args()

import arma
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import trimesh

from mpl_toolkits.basemap import Basemap

if __name__ == '__main__':

    T = trimesh.load(args.obj_path)

    V = T.vertices
    V -= np.mean(V, 0)
    R = np.sqrt(np.sum(V**2, 1))
    for i in range(3): V[:, i] /= R

    Phi = np.arctan2(V[:, 1], V[:, 0])
    Theta = np.arccos(V[:, 2])

    F = arma.fromfile(args.scalar_field_path)
    
    # triang = tri.Triangulation(Phi, Theta, T.faces)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = Basemap(projection='ortho', lon_0=0, lat_0=0, resolution='l')
    MC = m.contourf(np.degrees(Phi) + 180, 90 - np.degrees(Theta), F[:, 0], 10, ax=ax, tri=True, latlon=True)
    m.drawparallels(np.arange(0,81,20))
    m.drawmeridians(np.arange(-180,181,60))
    plt.show()
     # fig.savefig('test.png')
