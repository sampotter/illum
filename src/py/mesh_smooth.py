#!/usr/bin/env python3

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('out', type=str)
    args = parser.parse_args()

import arma
import openmesh as om
import numpy as np

mesh = om.read_trimesh(args.path)
X = arma.fromfile(args.data)
if mesh.n_faces() != X.size:
    raise Exception('incompatible mesh and field sizes')

Y = np.empty_like(X)

for face in mesh.faces():
    i = face.idx()
    if i % 100 == 0:
        print(i)
    indices = [i]
    for other_face in mesh.ff(face):
        indices.append(other_face.idx())
    Y[i] = np.mean(X[indices])

arma.tofile(Y, args.out)
