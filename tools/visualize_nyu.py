# coding: utf-8
import argparse
import os
import sys

import matplotlib
from PIL import Image

sys.path.append(os.getcwd())
# print (sys.path)
# from transform import Colorize

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm

print (os.getcwd())

from util import mkdir_if_not_exist
import json
import random


def vis_with_legend(indir_list, outdir, label_list, raw_rgb_dir, raw_optional_img_dir=None, gt_dir=None, ext="png",
                    title_names=None, n_sample=10):
    N_CLASS = len(label_list)
    values = np.arange(N_CLASS)

    n_imgs = 1 + len(indir_list)
    if raw_optional_img_dir:
        n_imgs += 1
    if gt_dir:
        n_imgs += 1

    mkdir_if_not_exist(outdir)


    n_row = 1
    n_col = int(round(float(n_imgs) / n_row))

    # img_fn_list = os.listdir(raw_rgb_dir)
    img_fn_list = os.listdir(gt_dir)
    # img_fn_list = os.listdir(indir_list[0])
    img_fn_list = random.sample(img_fn_list, n_sample)

    for one_img_fn in tqdm(img_fn_list):
        fig = plt.figure(figsize=(560 * n_col / 100, 425 * n_row / 100))  # sharex=True, sharey=True)

        ax_list = []
        ax_list.append(fig.add_subplot(n_row, n_col, 1))
        raw_img = Image.open(os.path.join(raw_rgb_dir, one_img_fn))

        ax_list[0].imshow(raw_img)
        ax_list[0].axis("off")
        ax_list[0].set_xticklabels([])
        ax_list[0].set_yticklabels([])

        ax_list[0].set_aspect('equal')
        offset = 1
        plt.axis('tight')

        if raw_optional_img_dir:
            ax_list.append(fig.add_subplot(n_row, n_col, offset + 1))
            raw_img = Image.open(os.path.join(raw_optional_img_dir, one_img_fn))

            ax_list[offset].imshow(raw_img, cmap='gray')
            ax_list[offset].axis("off")
            ax_list[offset].set_xticklabels([])
            ax_list[offset].set_yticklabels([])

            ax_list[offset].set_aspect('equal')
            plt.axis('tight')
            offset += 1

        if gt_dir:
            ax_list.append(fig.add_subplot(n_row, n_col, offset + 1))
            gt_img = Image.open(os.path.join(gt_dir, one_img_fn.replace("leftImg8bit", "gtFine_gtlabels")))
            gt_img = np.array(gt_img, dtype=np.uint8)
            ax_list[offset].imshow(gt_img, vmin=0, vmax=N_CLASS - 1, interpolation='none', cmap="jet")
            ax_list[offset].axis("off")
            ax_list[offset].set_xticklabels([])
            ax_list[offset].set_yticklabels([])

            ax_list[offset].set_aspect('equal')
            plt.axis('tight')
            offset += 1


        if title_names is not None:
            for i, title in enumerate(title_names):
                ax_list[i].set_title(title, fontsize=30)

        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        fig.tight_layout(pad=0)

        outfn = os.path.join(outdir, one_img_fn)
        outfn = os.path.splitext(outfn)[0] + '.%s' % ext

        fig.savefig(outfn, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize labels')
    parser.add_argument('--outdir', type=str, required=True,
                        help='visualized dir')
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="gt dir")
    parser.add_argument("--way", type=str, default="legend", help="legend or colorize",
                        choices=['legend', 'colorize'])
    parser.add_argument("--ext", type=str, default="pdf")
    parser.add_argument("--dataset", type=str, default="nyu")
    parser.add_argument("--title_names", type=str, default=None, nargs='*')

    args = parser.parse_args()

    dataset_dic = {
        "nyu": {
            "json_fn": "./dataset/nyu_info.json",
            "raw_rgb_dir": "/data/unagi0/dataset/NYUDv2/gupta/rgb",
            "raw_optional_img_dir": "/data/unagi0/dataset/NYUDv2/gupta/hha",
            # "gt_dir": "/data/unagi0/dataset/NYUDv2/gupta/gt/semantic40"
            "gt_dir":"/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/vis"
        },
    }

    raw_rgb_dir = dataset_dic[args.dataset]["raw_rgb_dir"]
    raw_optional_img_dir = dataset_dic[args.dataset]["raw_optional_img_dir"]
    gt_dir = dataset_dic[args.dataset]["gt_dir"]

    with open(dataset_dic[args.dataset]["json_fn"], 'r') as f:
        info = json.load(f)
        label_list = np.array(info['label'] + ["background"], dtype=np.str)

    if args.way == "legend":

        vis_with_legend(indir_list=[], outdir=args.outdir, label_list=label_list, raw_rgb_dir=raw_rgb_dir,
                        raw_optional_img_dir=raw_optional_img_dir, gt_dir=gt_dir, ext=args.ext,
                        title_names=args.title_names)

    elif args.way == "colorize":  # TODO
        NotImplementedError()
