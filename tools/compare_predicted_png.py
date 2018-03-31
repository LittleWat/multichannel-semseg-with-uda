"""
Compare predicted visualized png.

Create merged png image.
"""

import os
import sys

from PIL import Image
from tqdm import tqdm

from util import mkdir_if_not_exist


def merge_four_images(files, outimg):
    """
    Create merged_img.

    files : list of 4 image file paths
    merged_img(PIL Image)
        - topleft: 1st
        - bottomleft: 2nd
        - topright: 3rd
        - bottomright: 4th
    """
    assert len(files) == 4

    img = [Image.open(file_) for file_ in files]

    img_size = img[0].size
    merged_img = Image.new('RGB', (img_size[0] * 2, img_size[1] * 2))
    for row in range(2):
        for col in range(2):
            merged_img.paste(img[row * 2 + col], (img_size[0] * row, img_size[1] * col))

    merged_img.save(outimg)


def main(vis_dirs, outdir):
    """Out merged_imgs from 4 directories (one directory is gt directory)."""
    assert len(vis_dirs) == 4

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for i, filename in enumerate(tqdm(os.listdir(vis_dirs[-1]))):
        # if i % 100 == 0:
        #     print(i)

        files = [os.path.join(vis_dir, filename) for vis_dir in vis_dirs]
        outimg = os.path.join(outdir, filename)
        merge_four_images(files, outimg)

    print ("Finished! Result dir is %s" % outdir)


if __name__ == '__main__':
    args = sys.argv

    """num of args need to be 3."""

    # merge_four_images(args[1:], 'sample_merged.png')
    # vis_dirs = ['/data/ugui0/dataset/adaptation/segmentation_test'] + args[1:]
    vis_dirs = ["/data/unagi0/dataset/NYUDv2/gupta/rgb/"]
    pred_base_dir = args[1]
    target_dir_list = ["vis", "depth", "boundary"]
    vis_dirs += [os.path.join(pred_base_dir, x) for x in target_dir_list]
    print (vis_dirs)

    # for i in range(20):
    #     outdir = 'merged_imgs/merged_imgs_{0}'.format(i)
    #     if os.path.exists(outdir):
    #         continue
    #     else:
    #         break
    outdir = os.path.join(pred_base_dir, "merged")
    mkdir_if_not_exist(outdir)

    main(vis_dirs, outdir)
