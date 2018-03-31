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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import numpy as np
import scipy.misc as m
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


def one_vis_with_legend(indir, outdir):
    for one_file in tqdm(os.listdir(indir)):
        fullpath = os.path.join(indir, one_file)
        hard_to_see_img = m.imread(fullpath)
        im = plt.imshow(hard_to_see_img.astype(np.int64), interpolation='none', cmap="jet", vmin=0, vmax=N_CLASS - 1)
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label=label_list[i]) for i in range(len(values))]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        outfn = os.path.join(outdir, one_file)
        plt.savefig(outfn, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()


def vis_with_legend(indir_list, outdir, label_list, raw_rgb_dir, raw_optional_img_dir=None, gt_dir=None, ext="pdf",
                    title_names=None):
    N_CLASS = len(label_list)
    values = np.arange(N_CLASS)

    n_imgs = 1 + len(indir_list)
    if raw_optional_img_dir:
        n_imgs += 1
    if gt_dir:
        n_imgs += 1

    mkdir_if_not_exist(outdir)

    n_row = 2
    n_col = int(round(float(n_imgs) / n_row))

    # img_fn_list = os.listdir(raw_rgb_dir)
    img_fn_list = os.listdir(indir_list[0])

    for one_img_fn in tqdm(img_fn_list):
        fig = plt.figure(figsize=(560 * n_col / 100, 425 * n_row / 100 * 1.2))  # sharex=True, sharey=True)

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

        # ax_list[0].set_aspect('equal')
        for i, (indir, title_name) in enumerate(zip(indir_list, title_names[offset:])):
            # hard_to_see_img = m.imread(os.path.join(indir, one_img_fn))
            hard_to_see_img = Image.open(os.path.join(indir, one_img_fn)).resize(raw_img.size)
            hard_to_see_img = np.array(hard_to_see_img)

            ax_list.append(fig.add_subplot(n_row, n_col, i + offset + 1))

            # hsv =  plt.get_cmap('hsv')
            # colors = hsv(np.linspace(0, 1.0, N_CLASS))
            def discrete_cmap(N, base_cmap=None):
                """Create an N-bin discrete colormap from the specified input map"""

                # Note that if base_cmap is a string or None, you can simply do
                #    return plt.cm.get_cmap(base_cmap, N)
                # The following works for string, None, or a colormap instance:

                base = plt.cm.get_cmap(base_cmap)
                color_list = base(np.linspace(0, 1, N))
                cmap_name = base.name + str(N)
                return base.from_list(cmap_name, color_list, N)

            cmap = "gray" if "boundary" in title_name.lower() else "jet"
            vmax = 255 if "boundary" in title_name.lower() else N_CLASS - 1

            im = ax_list[i + offset].imshow(hard_to_see_img.astype(np.uint8), vmin=0, vmax=vmax,
                                            interpolation='none',
                                            cmap=cmap)
            # cmap=discrete_cmap(N_CLASS, "jet"))



            ax_list[i + offset].axis("off")
            ax_list[i + offset].set_xticklabels([])
            ax_list[i + offset].set_yticklabels([])

            ax_list[i + offset].set_title(indir.replace("outputs/", "").replace("/label", "").replace("/", "\n"),
                                          fontsize=4)
            ax_list[i + offset].set_aspect('equal')
            plt.axis('tight')

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
        outfn = os.path.join(outdir, one_img_fn)
        outfn = os.path.splitext(outfn)[0] + '.%s' % ext

        fig.savefig(outfn, transparent=True, bbox_inches='tight', pad_inches=0)
        # fig.savefig(outfn, transparent=True, bbox_inches='tight', pad_inches=0, bbox_extra_artists=(lgd,), dpi=300)
        plt.close()


# TODO This is not work
def vis_using_Colorize(indir_list, outdir):
    indir = indir_list[0]
    # outdir = os.path.join(os.path.split(indir)[0], "vis_labels")
    mkdir_if_not_exist(outdir)

    for one_file in tqdm(os.listdir(indir)):
        fullpath = os.path.join(indir, one_file)
        hard_to_see_img = m.imread(fullpath)
        # outputs = outputs[0, :19].data.max(0)[1]
        # outputs = outputs.view(1, outputs.size()[0], outputs.size()[1])
        outputs = hard_to_see_img  # TODO this should be fixed
        output = Colorize()(outputs)
        output = np.transpose(output.cpu().numpy(), (1, 2, 0))
        img = Image.fromarray(output, "RGB")
        img = img.resize(hard_to_see_img.shape, Image.NEAREST)

        outfn = os.path.join(outdir, one_file)
        plt.savefig(outfn, transparent=True, bbox_inches='tight', pad_inches=0)
        img.save(outfn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize labels')
    parser.add_argument('--indir_list', type=str, nargs='*',
                        help='result directory that contains predicted labels(pngs)')
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
    parser.add_argument("--dataset", type=str, default="nyu")
    parser.add_argument("--title_names", type=str, default=None, nargs='*')

    args = parser.parse_args()

    dataset_dic = {
        "nyu": {
            "json_fn": "./dataset/nyu_info.json",
            "raw_rgb_dir": "/data/unagi0/dataset/NYUDv2/gupta/rgb",
            "raw_optional_img_dir": "/data/unagi0/dataset/NYUDv2/gupta/hha",
            "gt_dir": "/data/unagi0/dataset/NYUDv2/gupta/gt/semantic40"
        },
        "suncg": {
            "json_fn": "./dataset/nyu_info.json",
            "raw_rgb_dir": "/data/unagi0/dataset/SUNCG-Seg/mlt_v2",
            "raw_optional_img_dir": "/data/unagi0/dataset/SUNCG-Seg/hha_v2",
            "gt_dir": "/data/unagi0/dataset/SUNCG-Seg/category_v2",

        }

    }

    raw_rgb_dir = dataset_dic[args.dataset]["raw_rgb_dir"]
    raw_optional_img_dir = dataset_dic[args.dataset]["raw_optional_img_dir"]
    gt_dir = dataset_dic[args.dataset]["gt_dir"]

    with open(dataset_dic[args.dataset]["json_fn"], 'r') as f:
        info = json.load(f)
        label_list = np.array(info['label'] + ["background"], dtype=np.str)

    if args.way == "legend":

        vis_with_legend(indir_list=args.indir_list, outdir=args.outdir, label_list=label_list, raw_rgb_dir=raw_rgb_dir,
                        raw_optional_img_dir=raw_optional_img_dir, gt_dir=gt_dir, ext=args.ext,
                        title_names=args.title_names)

    elif args.way == "colorize":  # TODO
        vis_using_Colorize(args.indir_lis, args.outdir)
