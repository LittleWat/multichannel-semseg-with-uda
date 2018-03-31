import argparse
import os
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

from util import save_colorized_lbl, mkdir_if_not_exist, exec_eval


def refine_by_bwboundary(segdir, bwbddir, dataset, min_thre, max_thre):
    basedir = os.path.split(segdir)[0]
    out_seg_dir = os.path.join(basedir, "refined_label")
    out_vis_dir = os.path.join(basedir, "refined_vis")
    mkdir_if_not_exist(out_seg_dir)
    mkdir_if_not_exist(out_vis_dir)

    print ("Result will be saved in %s" % out_seg_dir)
    print ("Result will be saved in %s" % out_vis_dir)

    segdir = os.path.join(basedir, "label")
    img_fn_list = os.listdir(segdir)

    for img_fn in tqdm(img_fn_list):

        segimg_fn = os.path.join(segdir, img_fn)
        # bwbdimg_fn = os.path.join(basedir, "bwboundary", img_fn)
        bwbdimg_fn = os.path.join(bwbddir, img_fn)

        segimg = np.array(Image.open(segimg_fn))
        bwbdimg = np.array(Image.open(bwbdimg_fn))

        cnter = Counter(bwbdimg.flatten())
        ok_id_list = [k for k, v in cnter.items() if v < max_thre and v > min_thre and k != 1]

        res = np.copy(segimg)
        for ok_id in ok_id_list:
            ok_idxes = np.where(bwbdimg == ok_id)
            cnter = Counter(segimg[ok_idxes].flatten())
            top_id, n_pixel_of_top_id = cnter.most_common()[0]
            res[ok_idxes] = top_id

        res = Image.fromarray(res)

        out_seg_fn = os.path.join(out_seg_dir, img_fn)
        res.save(out_seg_fn)

        out_vis_fn = os.path.join(out_vis_dir, img_fn)
        save_colorized_lbl(res, out_vis_fn, dataset)

    print ("Finished!!!")

    return out_seg_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refine raw boundary imgs')
    parser.add_argument('segdir', type=str, help="Directory that contains segmentation results")
    parser.add_argument('bwbddir', type=str, help="Directory that contains bwboundary results")
    parser.add_argument('--dataset', type=str, default="nyu", help="dataset (used for colorizing indexed label)")
    parser.add_argument('--min_thre', type=int, default=500,
                        help='the minimum number of pixel in a region')
    parser.add_argument('--max_thre', type=int, default=79333,  # 79333 = 425 * 560 / 3
                        help='the maximum number of pixel in a region')
    args = parser.parse_args()

    out_seg_dir = refine_by_bwboundary(segdir=args.segdir, bwbddir=args.bwbddir, dataset=args.dataset,
                                       min_thre=args.min_thre, max_thre=args.max_thre)

    exec_eval(args.dataset, out_seg_dir)
