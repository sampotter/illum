#!/usr/bin/env python3

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('scalar_field_path', type=str)
    parser.add_argument('obj_path', type=str)
    args = parser.parse_args()

import arma
import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import trimesh

from mpl_toolkits.basemap import Basemap

if __name__ == '__main__':

    T = trimesh.load(args.obj_path)

    V = T.vertices
    V -= np.mean(V, 0)

    # explain
    F = T.faces
    V = V[F].reshape(3*F.shape[0], 3)

    R = np.sqrt(np.sum(V**2, 1))
    V /= R.reshape(V.shape[0], 1)*np.ones((1, 3))

    print(V.shape)

    Phi = np.arctan2(V[:, 1], V[:, 0])
    Lon = np.degrees(Phi) + 180

    Theta = np.arccos(V[:, 2])
    Lat = 90 - np.degrees(Theta)

    data = arma.fromfile(args.scalar_field_path)
    data = data.repeat(3)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # m = Basemap(projection='gnom', lon_0=90, lat_0=0,
    #            llcrnrlon=45, llcrnrlat=-45, urcrnrlon=135, urcrnrlat=45,
    #            resolution='l')
    m = Basemap(projection='ortho', lon_0=165, lat_0=45)
    m.pcolor(Lon, Lat, data, tri=True, latlon=True, cmap=cc.m_fire)
    m.drawparallels(np.arange(-90,91,30))
    m.drawmeridians(np.arange(-180,181,30))
    m.colorbar()
    plt.show()
