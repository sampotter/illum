#!/usr/bin/env python3

import argparse
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='TODO')
    p.add_argument('source_mesh_path', type=str)
    p.add_argument('source_field_path', type=str)
    p.add_argument('target_mesh_path', type=str)
    p.add_argument('target_field_path', type=str)
    p.add_argument('--fix', action='store_true')
    p.add_argument('--save_layers', action='store_true')
    args = p.parse_args()

import arma
import numpy as np
import scipy.spatial
import trimesh

def get_centroids(path):
    T = trimesh.load(path)
    V = T.vertices
    F = T.faces
    return np.mean(V[F], 1)

if __name__ == '__main__':

    print('- loading source mesh')
    P = get_centroids(args.source_mesh_path)

    print('- loading target mesh')
    Q = get_centroids(args.target_mesh_path)

    print('- loading source data')
    X = arma.fromfile(args.source_field_path)

    # TODO: there's some weird transposition issue somewhere in the
    # pipeline---this is just a hack to fix it temporarily
    if args.fix:
        m, n = X.shape
        X = X.reshape(n, m).T
    
    num_src_pts = P.shape[0]
    num_tgt_pts = Q.shape[0]

    if num_src_pts == X.shape[0]:
        X = X.T
        transposed = True
    else:
        transposed = False
    if num_src_pts != X.shape[1]:
        raise Exception('mismatched sizes')
    num_layers = X.shape[0]

    Y = np.empty((num_layers, num_tgt_pts), dtype=X.dtype)

    P_kdtree = scipy.spatial.cKDTree(P)

    for i_tgt in range(num_tgt_pts):
        q = Q[i_tgt, :]
        _, i_src = P_kdtree.query(q)
        Y[:, i_tgt] = X[:, i_src]

    # TODO: add some kind of interpolation

    if transposed:
        Y = Y.T

    if args.fix:
        m, n = Y.shape
        Y = Y.T.reshape(m, n)

    if args.save_layers:
        for layer in range(num_layers):
            arma.tofile(
                Y[layer, :].reshape(Y.shape[1], 1),
                args.target_field_path % layer)
    else:
        arma.tofile(Y, args.target_field_path)
