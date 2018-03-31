import argparse
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from util import mkdir_if_not_exist

GT_DIR_DIC = {
    "city": '/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes_gt/val',
    "nyu": "/data/unagi0/dataset/NYUDv2/gupta/gt/semantic40",
    "ir": "/data/unagi0/inf_data/ir_seg_dataset/labels"
}

parser = argparse.ArgumentParser(description='GT Coloring')
parser.add_argument('dataset', choices=["gta", "city", "test", "ir", "city16", "nyu"])

args = parser.parse_args()

if args.dataset in ["city16", "synthia"]:
    info_json_fn = "./dataset/synthia2cityscapes_info.json"
elif args.dataset in ["nyu"]:
    info_json_fn = "./dataset/nyu_info.json"
elif args.dataset == "ir":
    info_json_fn = "./dataset/ir_info.json"
else:
    info_json_fn = "./dataset/city_info.json"

    # Save visualized predicted pixel labels(pngs)
with open(info_json_fn) as f:
    info_dic = json.load(f)
palette = np.array(info_dic['palette'], dtype=np.uint8)

gt_dir = GT_DIR_DIC[args.dataset]
vis_outdir = os.path.join(os.path.split(gt_dir)[0], os.path.split(gt_dir)[1] + "_pretty")
print ("OUTDIR is %s" % vis_outdir)
mkdir_if_not_exist(vis_outdir)

gtfn_list = os.listdir(gt_dir)

for gtfn in tqdm(gtfn_list):
    full_gtfn = os.path.join(gt_dir, gtfn)
    img = Image.open(full_gtfn).convert("P")
    img.putpalette(palette.flatten())
    vis_fn = os.path.join(vis_outdir, gtfn)
    img.save(vis_fn)
