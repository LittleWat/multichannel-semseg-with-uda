import argparse
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from util import mkdir_if_not_exist


def save_img_by_PIL(np_img, outfn):
    np_img = np.clip(np_img, 0, 255)
    res = Image.fromarray(np.uint8(np_img))
    res.save(outfn)


def edge_detect(rgb_dir, min_canny_thre, max_canny_thre):
    base_dir = os.path.split(rgb_dir)[0]
    edge_dir = os.path.join(base_dir, "edges")
    laplacian_dir = os.path.join(edge_dir, "laplacian")
    canny_dir = os.path.join(edge_dir, "canny")
    sobel_dir = os.path.join(edge_dir, "sobel")

    mkdir_if_not_exist(laplacian_dir)
    mkdir_if_not_exist(canny_dir)
    mkdir_if_not_exist(sobel_dir)

    print ("Result will be saved in %s" % edge_dir)

    fn_list = os.listdir(rgb_dir)

    for fn in tqdm(fn_list):
        img = cv2.imread(os.path.join(rgb_dir, fn))
        imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        imgY = imgYUV[:, :, 0]

        # Laplacian
        out_laplacian_im = cv2.Laplacian(imgY, cv2.CV_64F) + 128
        out_laplacian_fn = os.path.join(laplacian_dir, fn)
        save_img_by_PIL(out_laplacian_im, out_laplacian_fn)

        # Canny
        out_canny_im = cv2.Canny(imgY, cv2.CV_64F, min_canny_thre, max_canny_thre)
        out_canny_fn = os.path.join(canny_dir, fn)
        save_img_by_PIL(out_canny_im, out_canny_fn)

        # Sobel
        dx = cv2.Sobel(imgY, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(imgY, cv2.CV_64F, 0, 1, ksize=3)
        out_sobel_im = np.sqrt(dx ** 2 + dy ** 2)
        out_sobel_fn = os.path.join(sobel_dir, fn)
        save_img_by_PIL(out_sobel_im, out_sobel_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Edge Detection using opencv functions')
    parser.add_argument('rgbdir', type=str, help="Directory contains rgb images")
    parser.add_argument('--min_canny_thre', type=int, default=100)
    parser.add_argument('--max_canny_thre', type=int, default=200)
    args = parser.parse_args()

    edge_detect(args.rgbdir, args.min_canny_thre, args.max_canny_thre)
