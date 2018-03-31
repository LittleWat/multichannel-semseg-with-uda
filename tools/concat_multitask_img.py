"""
Compare predicted visualized png.

Create merged png image that is randomly selected with original RGB image and GT.
"""

import argparse
import os
import random

import numpy as np
from PIL import Image

from util import mkdir_if_not_exist

VIS_GT_DIR_DIC = {
    "city": "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes_vis_gt/val",
    "city16": "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes16_vis_gt/val",
    "ir": "/data/unagi0/inf_data/ir_seg_dataset/labels_pretty",
    "nyu": "/data/unagi0/dataset/NYUDv2/gupta/gt/semantic40_pretty",
}

RGB_IMG_DIR_DIC = {
    "city": "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes_val_imgs",
    "city16": "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes_val_imgs",
    "ir": "/data/unagi0/inf_data/ir_seg_dataset/rgb_images",
    "nyu": "/data/unagi0/dataset/NYUDv2/gupta/rgb"
}
EXTRA_IMG_DIR_DIC = {
    "ir": "/data/unagi0/inf_data/ir_seg_dataset/ir_images",
    "nyu": "/data/unagi0/dataset/NYUDv2/gupta/hha"
}

parser = argparse.ArgumentParser(description='Visualize Some Results')
parser.add_argument('dataset', choices=["nyu"])
parser.add_argument('--n_img', type=int, default=5)
parser.add_argument('--pred_dir', type=str,
                    help='result directory that visualized pngs')
parser.add_argument('--outdir', type=str, default="multitask_result")
parser.add_argument("--pick_up", action="store_true",
                    help='whether you sample results randomly')

args = parser.parse_args()

rgb_dir = RGB_IMG_DIR_DIC[args.dataset]
# vis_gt_dir = VIS_GT_DIR_DIC[args.dataset]
extra_img_dir = EXTRA_IMG_DIR_DIC[args.dataset] if args.dataset in EXTRA_IMG_DIR_DIC.keys() else None
print (args.pred_dir)

pred_seg_dir = os.path.join(args.pred_dir, "vis")
pred_depth_dir = os.path.join(args.pred_dir, "depth")

if not args.pick_up:
    # rgbfn_list = os.listdir(rgb_dir)
    # rgbfn_list = os.listdir(args.pred_vis_dirs[2])
    rgbfn_list = os.listdir(pred_seg_dir)

    if args.dataset == "ir":
        rgbfn_list = list(filter(lambda x: "D.png" in x, rgbfn_list))  # Only pick up photos taken at nights
        # rgbfn_list = list(filter(lambda x: "N.png" in x, rgbfn_list))  # Only pick up photos taken at nights
else:
    if args.dataset == "city":
        pickup_id_list = [
            "lindau_000006_000019",
            "frankfurt_000001_021406",
            "frankfurt_000001_041074",
            "frankfurt_000001_002512",
            "frankfurt_000000_009688",
            "frankfurt_000001_040575",
            "munster_000050_000019"
        ]
        rgbfn_list = [x + "_leftImg8bit.png" for x in pickup_id_list]

    elif args.dataset == "ir":
        pickup_id_list = [
            "01324N",
            "01250N",
            "01356N",
            # "01251N", # ?
            "01239N",
            "01230N"
        ]
        # day 01582D.png-01533D.png-01535D.png-01453D.png-01403D.png.pdf
        rgbfn_list = [x + ".png" for x in pickup_id_list]

    elif args.dataset == "nyu":
        ok_list = [
            "img_6294", "img_5471", "img_6219", "img_6408", "img_6233"
        ]
        ng_list = [
            "img_6144", "img_5038", "img_5726", "img_5862", "img_6396"
        ]
        # rgbfn_list = [x + ".png" for x in ok_list]
        rgbfn_list = [x + ".png" for x in ng_list]

    else:
        raise NotImplementedError()

pickup_rgbfn_list = random.sample(rgbfn_list, args.n_img)
print ("pickup filename list")
print (pickup_rgbfn_list)

all_img_list = []
for rgbfn in pickup_rgbfn_list:
    full_rgbfn = os.path.join(rgb_dir, rgbfn)

    one_column_img_list = []
    # RGB
    one_column_img_list.append(Image.open(full_rgbfn))

    # EXTRA IMG (Such as IR, Depth img)
    if extra_img_dir is not None:
        full_extra_img_fn = os.path.join(extra_img_dir, rgbfn)
        one_column_img_list.append(Image.open(full_extra_img_fn))

    # Predicted Depth IMG
    full_pred_depth_fn = os.path.join(pred_depth_dir, rgbfn)
    one_column_img_list.append(Image.open(full_pred_depth_fn))

    # Visualized Prediction Results
    full_pred_seg_fn = os.path.join(pred_seg_dir, rgbfn)
    one_column_img_list.append(Image.open(full_pred_seg_fn))

    all_img_list.append(one_column_img_list)


def concat_imgs(imgs):
    n_row = len(imgs[0])
    n_col = len(imgs)
    w, h = imgs[0][0].size

    merged_img = Image.new('RGB', (w * n_col, h * n_row))
    for col in range(n_col):
        for row in range(n_row):
            merged_img.paste(imgs[col][row], (w * col, h * row))

    return merged_img


res = concat_imgs(all_img_list)
size = np.array(res.size)

if args.dataset in ["ir", "nyu"]:
    res = res.resize(size / 4)
else:
    res = res.resize(size / 8)

mkdir_if_not_exist(args.outdir)
shortened_pickup_rgbfn_list = [x.replace("_leftImg8bit.png", "") for x in pickup_rgbfn_list]
pickup_str = "-".join(shortened_pickup_rgbfn_list) + ".pdf"
outfn = os.path.join(args.outdir, pickup_str)
res.save(outfn)
print ("Successfully saved result to %s" % outfn)

imgcat_str = "imgcat %s" % (outfn)
print (imgcat_str)
import subprocess
subprocess.call(imgcat_str, shell=True)