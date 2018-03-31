import argparse
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from util import save_colorized_lbl, mkdir_if_not_exist


def swap_labels(np_original_gt_im, class_convert_mat):
    # print (collections.Counter(np_original_gt_im.flatten()))
    np_processed_gt_im = np.zeros(np_original_gt_im.shape)
    for swap in class_convert_mat:
        ind_swap = np.where(np_original_gt_im == swap[0])
        np_processed_gt_im[ind_swap] = swap[1]
    processed_gt_im = Image.fromarray(np.uint8(np_processed_gt_im))
    return processed_gt_im


def convert_citylabelTo16label():
    with open('./synthia2cityscapes_info.json', 'r') as f:
        paramdic = json.load(f)

    class_ind = paramdic['city2common']

    city_gt_dir = "/data/ugui0/ksaito/D_A/image_citiscape/www.cityscapes-dataset.com/file-handling/gtFine"
    split_list = ["train", "test", "val"]

    original_suffix = "labelIds"
    processed_suffix = "label16IDs"

    for split in tqdm(split_list):
        base_dir = os.path.join(city_gt_dir, split)
        place_list = os.listdir(base_dir)
        for place in tqdm(place_list):
            target_dir = os.path.join(base_dir, place)
            pngfn_list = os.listdir(target_dir)
            original_pngfn_list = [x for x in pngfn_list if original_suffix in x]

            for pngfn in tqdm(original_pngfn_list):
                gt_fn = os.path.join(target_dir, pngfn)
                original_gt_im = Image.open(gt_fn)

                processed_gt_im = swap_labels(np.array(original_gt_im), class_ind)
                outfn = gt_fn.replace(original_suffix, processed_suffix)
                processed_gt_im.save(outfn, 'PNG')


def convert_synthialabelTo16label():
    with open('./synthia2cityscapes_info.json', 'r') as f:
        paramdic = json.load(f)

    class_ind = np.array(paramdic['synthia2common'])

    synthia_gt_dir = "/data/ugui0/dataset/adaptation/synthia/RAND_CITYSCAPES/GT"

    # original_dir = os.path.join(synthia_gt_dir, "LABELS") # Original dir but this contains strange files
    original_dir = "/data/ugui0/dataset/adaptation/synthia/new_synthia/segmentation_annotation/SYNTHIA/GT/parsed_LABELS"  # Not original. Downloaded from http://crcv.ucf.edu/data/adaptationseg/ICCV_dataset.zip
    processed_dir = os.path.join(synthia_gt_dir, "LABELS16")

    original_pngfn_list = os.listdir(original_dir)

    for pngfn in tqdm(original_pngfn_list):
        gt_fn = os.path.join(original_dir, pngfn)
        original_gt_im = Image.open(gt_fn)
        processed_gt_im = swap_labels(np.array(original_gt_im), class_ind)
        outfn = os.path.join(processed_dir, pngfn)
        processed_gt_im.save(outfn, 'PNG')


def convert_40to13cls(target_dir):
    with open('./dataset/nyu_info.json', 'r') as f:
        paramdic = json.load(f)

    class_ind = np.array(paramdic['40to13cls'])

    base_dir, indir = os.path.split(target_dir)
    out_lbl_dir = os.path.join(base_dir, indir + "-13cls")
    out_vis_dir = os.path.join(base_dir, "vis-13cls")
    original_pngfn_list = os.listdir(target_dir)

    mkdir_if_not_exist(out_lbl_dir)
    mkdir_if_not_exist(out_vis_dir)

    for pngfn in tqdm(original_pngfn_list):
        fullpath = os.path.join(target_dir, pngfn)
        original_im = Image.open(fullpath)
        processed_im = swap_labels(np.array(original_im), class_ind)
        out_lbl_fn = os.path.join(out_lbl_dir, pngfn)
        processed_im.save(out_lbl_fn, 'PNG')

        out_vis_fn = os.path.join(out_vis_dir, pngfn)
        save_colorized_lbl(processed_im, out_vis_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Label Ids')
    parser.add_argument('dataset', type=str, choices=["city", "synthia", "nyu"])
    parser.add_argument('target_dir', type=str, default="")
    args = parser.parse_args()
    if args.dataset == "city":
        convert_citylabelTo16label()
    elif args.dataset == "nyu":
        assert args.target_dir != ""
        convert_40to13cls(args.target_dir)
    else:
        convert_synthialabelTo16label()
