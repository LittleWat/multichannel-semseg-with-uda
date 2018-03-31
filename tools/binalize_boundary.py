import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def frame_img(img):
    img = np.array(img)
    res = np.ones([img.shape[0] + 2, img.shape[1] + 2])  # * 255
    res[1:-1, 1:-1] = img
    return Image.fromarray(np.uint8(res))


def binarize(img, thre):
    img = np.array(img)
    mask = np.zeros_like(img)
    mask[np.where(img > thre)] = 1
    return Image.fromarray(np.uint8(mask))


def save_binary_imgs(boundary_dir, thre):
    from util import mkdir_if_not_exist
    outdir = boundary_dir + "_binary"
    mkdir_if_not_exist(outdir)

    print ("Result will be saved in %s" % outdir)

    fn_list = os.listdir(boundary_dir)

    for fn in tqdm(fn_list):
        raw_boundary_img = Image.open(os.path.join(boundary_dir, fn))
        out_img = binarize(raw_boundary_img, thre)
        out_img = frame_img(out_img)
        out_img.save(os.path.join(outdir, fn))

    print ("Finished!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binarize raw boundary imgs')
    parser.add_argument('boundary_dir', type=str, help="Raw boundary directory name")

    parser.add_argument('--thre', type=int, default=50,
                        help='threshold to binalize. Set from 0 to 255')
    args = parser.parse_args()
    save_binary_imgs(boundary_dir=args.boundary_dir, thre=args.thre)
