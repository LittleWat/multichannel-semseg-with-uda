# coding: utf-8

import glob
import os
from copy import deepcopy

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import load_labels


def mkdir_if_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


semantic_labels = load_labels("semantic_labels.json")
label_list = [x.split("_")[0] for x in semantic_labels]
unique_label_list = sorted(list(set(label_list)))
tmp = unique_label_list[0]
unique_label_list = unique_label_list[1:]
unique_label_list.append(tmp)

idx2lblid = {x: i for i, x in enumerate(unique_label_list)}
idx2lblid['<UNK>'] = 255

print (idx2lblid)

lblid_list = [idx2lblid[x] for x in label_list]
n_semantic = len(lblid_list)

# area_list = ["area_1", "area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6"]
# area_list = ["area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6"]
area_list = ["area_1"]


def process_per_area(area):
    gt_dir = "/data/unagi0/dataset/2D-3D-SemanticsData/XYZ/%s/data/semantic/" % area
    out_dir = "/data/unagi0/dataset/2D-3D-SemanticsData/XYZ/%s/data/cls_lbl/" % area
    mkdir_if_not_exist(out_dir)
    gt_fn_list = glob.glob(os.path.join(gt_dir, "*.png"))

    for gt_fn in tqdm(gt_fn_list):
        gt_im = deepcopy(np.asarray(Image.open(gt_fn)))
        # gt_im = cv2.imread(gt_fn)
        # gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)
        try:
            # print (np.array(gt_im)[:, :, 0].max(), np.array(gt_im)[:, :, 1].max(), np.array(gt_im)[:, :, 2].max())

            lbl_im = gt_im[:, :, 0] * 65536 + gt_im[:, :, 1] * 256 + gt_im[:, :, 2]
            out_lbl_arr = np.array([lblid_list[x] if x < n_semantic else 255 for x in lbl_im.flatten()]).reshape(
                lbl_im.shape)
            out_lbl_im = Image.fromarray(np.uint8(out_lbl_arr))
            # print (Counter(out_lbl_arr.flatten()))
            out_fn = os.path.join(out_dir, os.path.split(gt_fn)[-1])
            out_lbl_im.save(out_fn)
            # cv2.imwrite(out_fn, np.uint8(out_lbl_arr))
        except IndexError:
            print ("OOOOOOOOOOOOOOOOOOPS!!!")
            continue


from multiprocessing import Pool

p = Pool(len(area_list))  # 最大プロセス数:10
# result = p.map(process_per_area, area_list)
result = p.map(process_per_area, area_list)
p.map_async(process_per_area(), area_list).get(9999999)
