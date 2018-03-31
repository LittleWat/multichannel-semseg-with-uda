import argparse
import os
import os
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

from util import save_dic_to_json

def get_gt_path_list(dataset):
    assert dataset in ["nyu", "suncg"]

def main():
fn_list = open("/data/unagi0/dataset/SUNCG-Seg/data_goodlist_v2.txt").readlines()

fn_list = [fn.strip() for fn in fn_list]

cls_hist = np.zeros(256)
counter = Counter()

for fn in tqdm(fn_list):
    gt_fn = os.path.join("/data/unagi0/dataset/SUNCG-Seg/category_v2/", fn + "_category40.png")
    gt_im = np.array(Image.open(gt_fn))


print (counter)

mul = lambda x, y: x * y
n_pixel_per_img = reduce(mul, gt_im.size)
correct_n_pixel = n_pixel_per_img * len(fn_list)

got_n_pixel = sum(counter.values())

assert got_n_pixel == correct_n_pixel

print (got_n_pixel)

save_dic_to_json(counter, "suncg_gt_distribution.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

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
            # "gt_dir": "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab_only_3ch---suncg-train_rgbhha/normal-drn_d_38-20.tar/label",
            # "gt_dir": "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhha_only_6ch---suncg-train_rgbhha/b16-drn_d_38-10.tar/label",
        }
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
