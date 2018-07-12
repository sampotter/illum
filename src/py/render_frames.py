#!/usr/bin/env python3

import argparse
import glob
import moderngl
import numpy as np
import os
import trimesh

from moderngl.ext.obj import Obj
from PIL import Image
from pyrr import Matrix44

def main(dipath, objpath, width, height, outpath, normalize):
    paths = glob.glob(dipath)
    nfiles = len(paths)

    i0s = np.zeros(nfiles, dtype=np.int)
    i1s = np.zeros(nfiles, dtype=np.int)

    for k, path in enumerate(paths):
        i0, i1 = map(int, path.split('/')[-1].split('.')[0].split('_')[1:3])
        i0s[k] = i0
        i1s[k] = i1

    assert(min(i0s) == 0)

    with open(paths[0], 'rb') as f:
        header = f.readline()
        assert(header[:4] == b'ARMA')
        _, nsunpos = map(int, f.readline().split())

    nfaces = max(i1s)
    direct_illum = np.zeros((nfaces, nsunpos), dtype=np.float64)

    for k, path in enumerate(paths):
        i0, i1 = i0s[k], i1s[k]
        with open(path, 'rb') as f:
            f.readline() # skip first line of header
            nrows, ncols = map(int, f.readline().split())
            assert(nrows == i1 - i0)
            assert(ncols == nsunpos)
            tmp = np.fromfile(f, dtype=np.float64)
            direct_illum[i0:i1, :] = tmp.reshape(ncols, nrows).T

    if normalize:
        direct_illum -= direct_illum.min()
        direct_illum /= direct_illum.max()

    tri = trimesh.load(objpath)
    nfaces = tri.faces.shape[0]
    V = tri.vertices[tri.faces]
    V = V.reshape(V.shape[0]*V.shape[1], V.shape[2])
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
        ('right', (xext, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ('left', (-xext, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ('front', (0.0, yext, 0.0), (0.0, 0.0, 1.0)),
        ('back', (0.0, -yext, 0.0), (0.0, 0.0, 1.0)),
        ('top', (0.0, 0.0, zext), (0.0, 1.0, 0.0)),
        ('bottom', (0.0, 0.0, -zext), (0.0, 1.0, 0.0))
    ]

    for fr in range(direct_illum.shape[1]):

        print('- frame = %d' % fr)

        # buffer the color data for this frame
        vbo.write_chunks(
            direct_illum[:, fr].repeat(3).astype(np.float32),
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

    parser = argparse.ArgumentParser(description='TODO')

    parser.add_argument('direct_illum_path', type=str)
    parser.add_argument('obj_path', type=str)
    parser.add_argument('-W', '--width', type=int, default=512)
    parser.add_argument('-H', '--height', type=int, default=512)
    parser.add_argument('-o', '--output_path', type=str, default='./tmp')
    parser.add_argument('-n', '--normalize', action='store_true')

    args = parser.parse_args()

    dipath = args.direct_illum_path
    objpath = args.obj_path
    width = args.width
    height = args.height
    outpath = args.output_path
    normalize = args.normalize

    if os.path.exists(outpath):
        if not os.path.isdir(outpath):
            raise "%s exists and isn't a directory" % outpath
    else:
        os.makedirs(outpath)

    main(dipath, objpath, width, height, outpath, normalize)
