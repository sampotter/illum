#!/usr/bin/env python3

import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='TODO')

    p.add_argument('source_mesh_path', type=str)
    p.add_argument('source_field_path', type=str)
    p.add_argument('target_mesh_path', type=str)
    p.add_argument('target_field_path', type=str)
    p.add_argument('--mode', type=str, default='nearest')
    p.add_argument('--fix', action='store_true')
    p.add_argument('--save_layers', action='store_true')
    p.add_argument('--neighbors', type=int, default=5)

    args = p.parse_args()

    if args.mode not in {'nearest', 'weighted', 'area_weighted'}:
        raise Exception('invalid interpolation mode "%s"' % args.mode)

import arma
import numpy as np
import scipy.spatial
import trimesh

if __name__ == '__main__':

    print('- loading source mesh')

    T_src = trimesh.load(args.source_mesh_path)
    V_src = T_src.vertices
    F_src = T_src.faces
    P_src = np.mean(V_src[F_src], 1)

    print('- loading target mesh')

    T_tgt = trimesh.load(args.target_mesh_path)
    V_tgt = T_tgt.vertices
    F_tgt = T_tgt.faces
    P_tgt = np.mean(V_tgt[F_tgt], 1)

    print('- loading source data')
    X_src = arma.fromfile(args.source_field_path)

    # TODO: there's some weird transposition issue somewhere in the
    # pipeline---this is just a hack to fix it temporarily
    if args.fix:
        m, n = X_src.shape
        X_src = X_src.reshape(n, m).T

    num_src_pts = P_src.shape[0]
    num_tgt_pts = P_tgt.shape[0]

    if num_src_pts == X_src.shape[0]:
        X_src = X_src.T
        transposed = True
    else:
        transposed = False
    if num_src_pts != X_src.shape[1]:
        raise Exception('mismatched sizes')
    num_layers = X_src.shape[0]

    X_tgt = np.empty((num_layers, num_tgt_pts), dtype=X_src.dtype)

    P_src_kdtree = scipy.spatial.cKDTree(P_src)

    if args.mode == 'nearest':

        for i_tgt in range(num_tgt_pts):
            p = P_tgt[i_tgt, :]
            _, i_src = P_src_kdtree.query(p)
            X_tgt[:, i_tgt] = X_src[:, i_src]

    elif args.mode == 'weighted':

        for i_tgt in range(num_tgt_pts):
            p = P_tgt[i_tgt, :]
            D_src, I_src = P_src_kdtree.query(p, args.neighbors)
            Lam = D_src/np.sum(D_src)
            X_tgt[:, i_tgt] = X_src[:, I_src]@Lam

    elif args.mode == 'area_weighted':

        Area_src = np.empty(num_src_pts)
        for i_src in range(num_src_pts):
            v0, v1, v2 = V_src[F_src[i_src, :]]
            Area_src[i_src] = np.linalg.norm(np.cross(v1 - v0, v2 - v0))/2

        Area_tgt = np.empty(num_tgt_pts)
        for i_tgt in range(num_tgt_pts):
            v0, v1, v2 = V_tgt[F_tgt[i_tgt, :]]
            Area_tgt[i_tgt] = np.linalg.norm(np.cross(v1 - v0, v2 - v0))/2

        for i_tgt in range(num_tgt_pts):
            p = P_tgt[i_tgt, :]
            D_src, I_src = P_src_kdtree.query(p, args.neighbors)
            Lam = D_src/np.sum(D_src)
            for layer in range(num_layers):
                X_tgt[layer, i_tgt] = Lam@(
                    X_src[layer, I_src]*Area_src[I_src])/Area_tgt[i_tgt]

    if transposed:
        X_tgt = X_tgt.T

    if args.fix:
        m, n = X_tgt.shape
        X_tgt = X_tgt.T.reshape(m, n)

    if args.save_layers:
        for layer in range(num_layers):
            arma.tofile(
                X_tgt[layer, :].reshape(X_tgt.shape[1], 1),
                args.target_field_path % layer)
    else:
        arma.tofile(X_tgt, args.target_field_path)
