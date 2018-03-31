import os
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

from util import save_dic_to_json

fn_list = open("/data/unagi0/dataset/SUNCG-Seg/data_goodlist_v2.txt").readlines()

fn_list = [fn.strip() for fn in fn_list]

cls_hist = np.zeros(256)
counter = Counter()

for fn in tqdm(fn_list):
    gt_fn = os.path.join("/data/unagi0/dataset/SUNCG-Seg/category_v2/", fn + "_category40.png")
    gt_im = Image.open(gt_fn)
    counter += Counter(np.array(gt_im).flatten().astype(long))

print (counter)

mul = lambda x, y: x * y
n_pixel_per_img = reduce(mul, gt_im.size)
correct_n_pixel = n_pixel_per_img * len(fn_list)

got_n_pixel = sum(counter.values())

assert got_n_pixel == correct_n_pixel

print (got_n_pixel)

save_dic_to_json(counter, "suncg_gt_distribution.json")
