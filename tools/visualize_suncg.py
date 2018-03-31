# coding: utf-8
import argparse
import os
import sys
import random

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


# label_list = [
#     "road",
#     "sidewalk",
#     "building",
#     "wall",
#     "fence",
#     "pole",
#     "light",
#     "sign",
#     "vegetation",
#     "terrain",
#     "sky",
#     "person",
#     "rider",
#     "car",
#     "truck",
#     "bus",
#     "train",
#     "motocycle",
#     "bicycle",
#     "background"
# ]
#
# values = np.arange(len(label_list))
# N_CLASS = len(label_list)




def vis_with_legend(indir_list, outdir, label_list, raw_rgb_dir, raw_optional_img_dir=None, gt_dir=None,
                    boundary_dir=None, ext="png", title_names=None, n_sample=10, ):
    N_CLASS = len(label_list)
    values = np.arange(N_CLASS)

    n_imgs = 1 + len(indir_list)
    if raw_optional_img_dir:
        n_imgs += 1
    if gt_dir:
        n_imgs += 1
    if boundary_dir:
        n_imgs += 1
    mkdir_if_not_exist(outdir)

    n_row = 1  # 2
    n_col = int(round(float(n_imgs) / n_row))

    # with open("/data/unagi0/dataset/SUNCG-Seg/data_goodlist_v2.txt") as f:
    with open(
            "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab_only_3ch---suncg-train_rgbhha/normal-drn_d_38-20.tar/data_list.txt") as f:
        # with open(
        #         "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhha_only_6ch---suncg-train_rgbhha/b16-drn_d_38-10.tar/data_list.txt") as f:
        fn_id_list = [x.strip() for x in f.readlines()]
    fn_id_list = random.sample(fn_id_list, n_sample)
    # fn_id_list = ["6f905fac454cea2d4cf5fd4d83a83a69/000000"]

    for one_img_id in tqdm(fn_id_list):
        fig = plt.figure(figsize=(640 * n_col / 100, 480 * n_row / 100))  # sharex=True, sharey=True)

        ax_list = []
        ax_list.append(fig.add_subplot(n_row, n_col, 1))
        raw_img = Image.open(os.path.join(raw_rgb_dir, one_img_id + "_mlt.png"))

        ax_list[0].imshow(raw_img)
        ax_list[0].axis("off")
        ax_list[0].set_xticklabels([])
        ax_list[0].set_yticklabels([])

        ax_list[0].set_aspect('equal')
        offset = 1
        plt.axis('tight')

        if raw_optional_img_dir:
            ax_list.append(fig.add_subplot(n_row, n_col, offset + 1))
            raw_img = Image.open(os.path.join(raw_optional_img_dir, one_img_id + "_hha.png"))

            ax_list[offset].imshow(raw_img, cmap='gray')
            ax_list[offset].axis("off")
            ax_list[offset].set_xticklabels([])
            ax_list[offset].set_yticklabels([])

            ax_list[offset].set_aspect('equal')
            plt.axis('tight')
            offset += 1

        if gt_dir:
            ax_list.append(fig.add_subplot(n_row, n_col, offset + 1))
            gt_img = Image.open(os.path.join(gt_dir, one_img_id + "_category40.png"))
            gt_img = np.array(gt_img, dtype=np.uint8)
            ax_list[offset].imshow(gt_img, vmin=0, vmax=N_CLASS - 1, interpolation='none', cmap="jet")
            ax_list[offset].axis("off")
            ax_list[offset].set_xticklabels([])
            ax_list[offset].set_yticklabels([])

            ax_list[offset].set_aspect('equal')
            plt.axis('tight')
            offset += 1

        if boundary_dir:
            ax_list.append(fig.add_subplot(n_row, n_col, offset + 1))
            boundary_img = Image.open(os.path.join(boundary_dir, one_img_id + "_instance_boundary.png"))
            boundary_img = np.array(boundary_img, dtype=np.uint8)
            ax_list[offset].imshow(boundary_img, vmin=0, vmax=N_CLASS - 1, interpolation='none', cmap="gray")
            ax_list[offset].axis("off")
            ax_list[offset].set_xticklabels([])
            ax_list[offset].set_yticklabels([])

            ax_list[offset].set_aspect('equal')
            plt.axis('tight')
            offset += 1

        if title_names is not None:
            for i, title in enumerate(title_names):
                ax_list[i].set_title(title, fontsize=30)

        # fig.subplots_adjust(wspace=0, hspace=0)
        # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        fig.tight_layout(pad=0)

        # colors = [im.cmap(im.norm(value)) for value in values]
        # patches = [mpatches.Patch(color=colors[i], label=label_list[i]) for i in range(len(values))]
        # # lgd = fig.legend(handles=patches, labels=label_list, bbox_to_anchor=(1.05, 1), borderaxespad=0.,
        # #                  fontsize=7, loc='upper left')  # loc=2
        # if n_col * 2 <= N_CLASS:
        #     n_legend_col = n_col * 2
        # else:
        #     n_legend_col = N_CLASS
        # lgd = plt.legend(patches, label_list, loc='lower center', bbox_to_anchor=(0, 0, 1, 1),
        #                  bbox_transform=plt.gcf().transFigure, ncol=n_legend_col, fontsize=5)

        # fig.tight_layout()
        outfn = os.path.join(outdir, os.path.split(one_img_id)[-2] + "_" + os.path.split(one_img_id)[-1])
        outfn = os.path.splitext(outfn)[0] + '.%s' % ext

        fig.savefig(outfn, transparent=True, bbox_inches='tight', pad_inches=0)
        # fig.savefig(outfn, transparent=True, bbox_inches='tight', pad_inches=0, bbox_extra_artists=(lgd,), dpi=300)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize labels')
    # parser.add_argument('--indir_list', type=str, nargs='*',
    #                     help='result directory that contains predicted labels(pngs)')
    ##
    # ~/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgb_only_3ch---suncg-train_rgb/normal-drn_d_38-20.tar
    ##

    parser.add_argument('--outdir', type=str, required=True,
                        help='visualized dir')
    # parser.add_argument("--raw_rgb_dir", type=str, default="/data/ugui0/dataset/adaptation/segmentation_test",
    #                     help="raw img dir")
    # parser.add_argument("--raw_optional_img_dir", type=str, default=None,
    #                     help="raw img dir2")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="gt dir")
    parser.add_argument("--way", type=str, default="legend", help="legend or colorize",
                        choices=['legend', 'colorize'])
    parser.add_argument("--ext", type=str, default="pdf")
    parser.add_argument("--dataset", type=str, default="suncg")
    parser.add_argument("--title_names", type=str, default=None, nargs='*')

    args = parser.parse_args()

    dataset_dic = {
        "suncg": {
            "json_fn": "./dataset/nyu_info.json",
            "raw_rgb_dir": "/data/unagi0/dataset/SUNCG-Seg/mlt_v2",
            "raw_optional_img_dir": "/data/unagi0/dataset/SUNCG-Seg/hha_v2",
            "gt_dir": "/data/unagi0/dataset/SUNCG-Seg/category_v2",
            "boundary_dir": "/data/unagi0/dataset/SUNCG-Seg/boundary_v2",
            # "gt_dir": "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab_only_3ch---suncg-train_rgbhha/normal-drn_d_38-20.tar/label",
            # "gt_dir": "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhha_only_6ch---suncg-train_rgbhha/b16-drn_d_38-10.tar/label",
        }
    }

    raw_rgb_dir = dataset_dic[args.dataset]["raw_rgb_dir"]
    raw_optional_img_dir = dataset_dic[args.dataset]["raw_optional_img_dir"]
    gt_dir = dataset_dic[args.dataset]["gt_dir"]
    boundary_dir = dataset_dic[args.dataset]["boundary_dir"]

    with open(dataset_dic[args.dataset]["json_fn"], 'r') as f:
        info = json.load(f)
        label_list = np.array(info['label'] + ["background"], dtype=np.str)

    if args.way == "legend":

        vis_with_legend(indir_list=[], outdir=args.outdir, label_list=label_list, raw_rgb_dir=raw_rgb_dir,
                        raw_optional_img_dir=raw_optional_img_dir, gt_dir=gt_dir, boundary_dir=boundary_dir,
                        ext=args.ext, title_names=args.title_names)

    elif args.way == "colorize":  # TODO
        NotImplementedError()
