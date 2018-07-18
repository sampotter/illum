#!/usr/bin/env python3

import argparse
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='TODO')
    p.add_argument('image_dir', type=str)
    p.add_argument('-o', '--output_dir', type=str, default='.')
    p.add_argument('-f', '--image_format', type=str, default='png')
    args = p.parse_args()

import glob
import os
import PIL.Image
import PIL.ImageDraw

views = ['east', 'west', 'north', 'near', 'far', 'south']

def get_img_view_paths(fr):
    return {
        v: os.path.join(args.image_dir, '%d_%s.%s' % (fr, v, args.image_format))
        for v in views}

def get_size_and_mode(view_paths):
    p = view_paths[views[0]]
    im = PIL.Image.open(p)
    return im.size, im.mode

def make_merged_and_annotated_img(fr):
    view_paths = get_img_view_paths(fr)
    size, mode = get_size_and_mode(view_paths)
    w, h = size
    merged_im = PIL.Image.new(mode, (3*w, 2*h))
    for i, v in enumerate(view_paths.keys()):
        row = i // 3
        col = i % 3
        p = view_paths[v]
        im = PIL.Image.open(p)
        d = PIL.ImageDraw.Draw(im)
        d.text((0, 0), v)
        merged_im.paste(im, (col*w, row*h, (col + 1)*w, (row + 1)*h))
    merged_im_path = '%d.%s' % (fr, args.image_format)
    print('- writing %s' % merged_im_path)
    merged_im.save(os.path.join(args.output_dir, merged_im_path))

if __name__ == '__main__':
    glob_str = os.path.join(args.image_dir, '*.%s' % args.image_format)
    frames = list({
        int(path.split('/')[-1].split('_')[0]) for path in glob.glob(glob_str)})
    for fr in frames:
        make_merged_and_annotated_img(fr)
