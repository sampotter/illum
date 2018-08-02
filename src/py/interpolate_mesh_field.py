#!/usr/bin/env python3

import argparse
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='TODO')
    p.add_argument('source_mesh_path', type=str)
    p.add_argument('source_field_path', type=str)
    p.add_argument('target_mesh_path', type=str)
    p.add_argument('target_field_path', type=str)
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
    P_src = get_centroids(args.source_mesh_path)

    print('- loading target mesh')
    P_tgt = get_centroids(args.target_mesh_path)

    print('- loading source data')
    data_src = arma.fromfile(args.source_field_path)
    data_tgt = np.full(P_tgt.shape[0], np.nan, dtype=data_src.dtype)

    if data_src.ndim != 1 and data_src.shape[1] != 1:
        raise Exception('input field must be a vector')

    if P_src.shape[0] != data_src.size:
        raise Exception('mismatched sizes')

    print('- upsampling source data to target mesh')
    kd_src = scipy.spatial.KDTree(P_src)
    for i_tgt, p in enumerate(P_tgt):
        i_src = kd_src.query(p)[1]
        data_tgt[i_tgt] = data_src[i_src]

    print('- smoothing data')
    kd_tgt = scipy.spatial.KDTree(P_tgt)
    for i_tgt, p in enumerate(P_tgt):
        stencil = kd_tgt.query(p, 5)[1]
        data_tgt[i_tgt] = np.mean(data_tgt[stencil])

    if np.any(np.isnan(data_tgt)):
        raise Exception('something bad happened: there are NaNs in the output')

    print('- writing upsampled data')
    arma.tofile(data_tgt.reshape(data_tgt.size, 1), args.target_field_path)
