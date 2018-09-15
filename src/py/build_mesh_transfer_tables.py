#!/usr/bin/env python3

import argparse
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='TODO')
    p.add_argument('mesh_list_path', type=str)
    p.add_argument('-n', '--num_neighbors', type=int, default=5)
    p.add_argument('-o', '--output_path', type=str, default='.')
    args = p.parse_args()

import arma
import numpy as np
import os
import scipy.spatial
import trimesh
import yaml

def get_centroids(path):
    T = trimesh.load(path)
    V = T.vertices
    F = T.faces
    return np.mean(V[F], 1)

if __name__ == '__main__':
    with open(args.mesh_list_path) as f:
        lines = f.readlines()
    paths = [line.strip() for line in lines]

    P_src = get_centroids(paths[0])
    kd_src = scipy.spatial.cKDTree(P_src)

    downsample_lut = dict()
    neighbor_lut = dict()

    for i in range(1, len(paths)):
        print(i)

        P_tgt = get_centroids(paths[i])
        kd_tgt = scipy.spatial.cKDTree(P_tgt)

        # downsampling table
        downsample_lut[i] = np.array(
            [kd_src.query(p)[1] for p in P_tgt],
            dtype=np.uint32)

        # nearest neighbors
        neighbor_lut[i] = np.array(
            [kd_tgt.query(p, args.num_neighbors)[1] for p in P_tgt],
            dtype=np.uint32)

        P_src = P_tgt
        kd_src = kd_tgt

    os.makedirs(args.output_path, exist_ok=True)

    for i in range(1, len(paths)):
        path = os.path.join(args.output_path, 'downsample_lut_%d.bin' % i)
        arma.tofile(downsample_lut[i], path)

    for i in range(1, len(paths)):
        path = os.path.join(args.output_path, 'neighbor_lut_%d.bin' % i)
        arma.tofile(neighbor_lut[i], path)

    dump = yaml.dump({
        'num_neighbors': args.num_neighbors,
        'paths': paths,
        'downsample_lut': [
            'downsample_lut_%d.bin' % i for i in range(1, len(paths))],
        'neighbor_lut': [
            'neighbor_lut_%d.bin' % i for i in range(1, len(paths))]})

    with open(os.path.join(args.output_path, 'info.yaml'), 'w') as f:
        print(dump, file=f)

    
