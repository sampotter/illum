#!/usr/bin/env python3

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('scalar_field_path', type=str)
    parser.add_argument('obj_path', type=str)
    parser.add_argument('-W', '--width', type=int, default=512)
    parser.add_argument('-H', '--height', type=int, default=512)
    parser.add_argument('-o', '--output_path', type=str, default='./tmp')
    parser.add_argument('-s', '--start_frame', type=int, default=0)
    parser.add_argument('-e', '--end_frame', type=int, default=-1)
    parser.add_argument('-n', '--normalize', action='store_true')
    args = parser.parse_args()

import glob
import moderngl
import numpy as np
import os
import trimesh

from moderngl.ext.obj import Obj
from PIL import Image
from pyrr import Matrix44

def check_if_using_mpi(paths):
    path = paths[0]
    parts = path.split('/')[-1].split('.')[0].split('_')
    return len(parts) > 2 and all(s.isdigit() for s in parts[-2:])

def main(dipath, objpath, width, height, outpath, start_frame,
         end_frame, normalize):
    paths = glob.glob(dipath)

    using_mpi = check_if_using_mpi(paths)

    nfiles = len(paths)

    i0s = np.zeros(nfiles, dtype=np.int)
    i1s = np.zeros(nfiles, dtype=np.int)

    if using_mpi:
        for k, path in enumerate(paths):
            i0, i1 = map(int, path.split('/')[-1].split('.')[0].split('_')[1:3])
            i0s[k] = i0
            i1s[k] = i1

    assert(min(i0s) == 0)

    with open(paths[0], 'rb') as f:
        header = f.readline()
        assert(header[:4] == b'ARMA')
        nfaces, nsunpos = map(int, f.readline().split())

    if not using_mpi:
        i0s[0] = 0
        i1s[0] = nfaces

    nfaces = max(i1s)
    scalar_field = np.zeros((nfaces, nsunpos), dtype=np.float64)

    for k, path in enumerate(paths):
        i0, i1 = i0s[k], i1s[k]
        with open(path, 'rb') as f:
            f.readline() # skip first line of header
            nrows, ncols = map(int, f.readline().split())
            assert(nrows == i1 - i0)
            assert(ncols == nsunpos)
            tmp = np.fromfile(f, dtype=np.float64)
            scalar_field[i0:i1, :] = tmp.reshape(ncols, nrows).T

    if normalize:
        scalar_field -= scalar_field.min()
        scalar_field /= scalar_field.max()

    tri = trimesh.load(objpath)
    nfaces = tri.faces.shape[0]
    V = tri.vertices[tri.faces]
    V = V.reshape(V.shape[0]*V.shape[1], V.shape[2])
    V = V@np.diag([1, -1, 1])
    
    C = np.zeros(nfaces)
    tmp = np.column_stack([V, C.repeat(3)])
    vertex_data = tmp.astype(np.float32).tobytes()

    centroid = tuple(tri.vertices.mean(0))
    radius = (tri.vertices.max() - tri.vertices.min())/2
    yext = 1.1*radius
    xext = (width/height)*yext
    zext = yext

    vertex_shader_source = open('shader.vert').read()
    fragment_shader_source = open('shader.frag').read()

    # Context creation

    ctx = moderngl.create_standalone_context()

    # Shaders

    prog = ctx.program(vertex_shader=vertex_shader_source,
                       fragment_shader=fragment_shader_source)

    # Vertex Buffer and Vertex Array

    vbo = ctx.buffer(vertex_data, dynamic=True)
    vao = ctx.simple_vertex_array(prog, vbo, *['in_vert'])

    nbytes = np.dtype(np.float32).itemsize

    # Framebuffers

    fbo = ctx.framebuffer(ctx.renderbuffer((width, height)),
                          ctx.depth_renderbuffer((width, height)))

    fbo.use()
    ctx.enable(moderngl.DEPTH_TEST)

    # Matrices and Uniforms

    views = [
        ('near', (xext, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ('far', (-xext, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ('west', (0.0, yext, 0.0), (0.0, 0.0, -1.0)),
        ('east', (0.0, -yext, 0.0), (0.0, 0.0, -1.0)),
        ('north', (0.0, 0.0, -zext), (0.0, 1.0, 0.0)),
        ('south', (0.0, 0.0, zext), (0.0, 1.0, 0.0))
    ]

    for fr in range(start_frame, end_frame):
        fr %= nsunpos
        print('- frame = %d' % fr)

        # buffer the color data for this frame
        vbo.write_chunks(
            scalar_field[:, fr].repeat(3).astype(np.float32),
            3*nbytes,
            4*nbytes,
            3*nfaces)

        for view, eye, up in views:

            perspective = Matrix44.orthogonal_projection(
                -xext, xext, yext, -yext, -zext, zext)
            lookat = Matrix44.look_at(eye, centroid, up)
            mvp = perspective * lookat
            prog['Mvp'].write(mvp.astype('f4').tobytes())

            # Rendering

            ctx.clear(0, 0, 0, 1);
            vao.render()

            # Loading the image using Pillow

            data = fbo.read(components=3, alignment=1)
            img = Image.frombytes('RGB', fbo.size, data, 'raw', 'RGB', 0, -1)
            img.save(os.path.join(outpath, '%d_%s.png' % (fr, view)))

if __name__ == '__main__':

    dipath = args.scalar_field_path
    objpath = args.obj_path
    width = args.width
    height = args.height
    outpath = args.output_path
    start_frame = args.start_frame
    end_frame = args.end_frame
    normalize = args.normalize

    if os.path.exists(outpath):
        if not os.path.isdir(outpath):
            raise "%s exists and isn't a directory" % outpath
    else:
        os.makedirs(outpath)

    main(dipath, objpath, width, height, outpath, start_frame,
         end_frame, normalize)
