# coding: utf-8
from __future__ import print_function

import argparse
import json
import os
from os.path import join

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm

import glob


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def calc_fw_iu(hist):
    pred_per_class = hist.sum(0)
    gt_per_class = hist.sum(1)

    return np.nansum(
        (gt_per_class * np.diag(hist)) / (pred_per_class + gt_per_class - np.diag(hist))) / gt_per_class.sum()


def calc_pixel_accuracy(hist):
    gt_per_class = hist.sum(1)
    return np.diag(hist).sum() / gt_per_class.sum()


def calc_mean_accuracy(hist):
    gt_per_class = hist.sum(1)
    acc_per_class = np.diag(hist) / gt_per_class
    return np.nanmean(acc_per_class)


def save_colorful_images(prediction, filename, palette, postfix='_color.png'):
    im = Image.fromarray(palette[prediction.squeeze()])
    im.save(filename[:-4] + postfix)


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix will be computed without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Ground truth')
    plt.xlabel('Predicted label')




def calc_all_metrics(class_list, pred_fullpath_list, gt_fullpath_list, consider_background_loss=False,
                     gt2common_mat=None, pred2common_mat=None, background_id=255, out_filename_prefix=""):
    pred_dir, _ = os.path.split(pred_fullpath_list[0])
    gt_dir, _ = os.path.split(gt_fullpath_list[0])
    out_dir = os.path.split(pred_dir)[0].replace("label", "")

    print("pred path: %s" % os.path.abspath(pred_dir))

    n_ignore_pic = 0
    n_class = len(class_list)
    hist = np.zeros((n_class, n_class))

    bg_mapping = np.array([
        [background_id, n_class - 1]
    ])

    for ind, (pred_fullpath, gt_fullpath) in tqdm(enumerate(zip(pred_fullpath_list, gt_fullpath_list))):
        pred = Image.open(pred_fullpath)
        label = Image.open(gt_fullpath)

        pred = pred.resize(label.size)
        pred = np.array(pred)
        label = np.array(label)

        if gt2common_mat is not None:
            label = label_mapping(label, gt2common_mat)
        if pred2common_mat is not None:
            pred = label_mapping(pred, pred2common_mat)
            pred = label_mapping(pred, bg_mapping)

        # print (np.unique(label))
        # print (np.unique(pred))

        if not consider_background_loss:
            not_background_idxes = np.where(label != background_id)
            pred = pred[not_background_idxes]
            label = label[not_background_idxes]
            if len(label) == 0:
                n_ignore_pic += 1
                continue

        hist += fast_hist(label.flatten(), pred.flatten(), n_class)

    print("*** %s *** images were ignored because it has no label" % n_ignore_pic)

    # Get label distribution
    pred_per_class = hist.sum(0)
    gt_per_class = hist.sum(1)

    used_class_id_list = np.where(gt_per_class != 0)[0]
    hist = hist[used_class_id_list][:, used_class_id_list]  # Extract only GT existing (more than 1) classes

    class_list = np.array(class_list)[used_class_id_list]

    iou_list = per_class_iu(hist)
    fwIoU = calc_fw_iu(hist)
    pixAcc = calc_pixel_accuracy(hist)
    mAcc = calc_mean_accuracy(hist)

    result_df = pd.DataFrame({
        'class': class_list,
        'IoU': iou_list,
        "pred_distribution": pred_per_class[used_class_id_list],
        "gt_distribution": gt_per_class[used_class_id_list],
    })
    result_df["IoU"] = result_df["IoU"] * 100  # change to percent ratio

    result_df.set_index("class", inplace=True)
    print("---- info per class -----")
    print(result_df)

    result_ser = pd.Series({
        "pixAcc": pixAcc,
        "mAcc": mAcc,
        "fwIoU": fwIoU,
        "mIoU": iou_list.mean()
    })
    result_ser = result_ser[["pixAcc", "mAcc", "fwIoU", "mIoU"]]
    result_ser *= 100  # change to percent ratio

    print("---- total result -----")
    print(result_ser)

    sep_dir_list = os.path.abspath(pred_dir).split(os.path.sep)
    kind = sep_dir_list[-3]
    model_name = sep_dir_list[-2]

    print("---- For copy and paste -----")
    result_str = "%s, %s, " % (kind, model_name.replace(".tar", ""))

    for val in result_ser.values:
        result_str += str(val) + ", "
    for iou in iou_list:
        result_str += str(iou * 100) + ", "

    result_str += os.path.abspath(pred_dir)
    print(result_str)
    print()

    try:
        # Save confusion matrix
        fig = plt.figure()
        normalized_hist = hist.astype("float") / hist.sum(axis=1)[:, np.newaxis]

        plot_confusion_matrix(normalized_hist, classes=class_list, title='Confusion matrix')
        outfigfn = os.path.join(out_dir, "%s_conf_mat.pdf" % out_filename_prefix)
        fig.savefig(outfigfn, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
        print("Confusion matrix was waved to %s" % outfigfn)

        outdffn = os.path.join(out_dir, "%s_eval_result_df.csv" % out_filename_prefix)
        result_df.to_csv(outdffn)
        print('Info per class was saved at %s !' % outdffn)
        outserfn = os.path.join(out_dir, "%s_eval_result_ser.csv" % out_filename_prefix)
        result_ser.to_csv(outserfn)
        print('Total result is saved at %s !' % outserfn)

    except IOError:
        print("Ooops :(, Saving Process Failed...")
        print("Maybe you have to change the Permission of the output directory: %s"
              % out_dir)


def eval_city(gt_dir, pred_dir, devkit_dir='', dset='cityscapes', add_bg_loss=False, is_label16=False):
    if is_label16:
        with open("./dataset/synthia2cityscapes_info.json", 'r') as fp:
            info = json.load(fp)
    else:
        with open(join(devkit_dir, 'data', dset, 'info.json'), 'r') as fp:
            info = json.load(fp)

    if is_label16:
        name_classes = np.array(info['common_label'], dtype=np.str)
        mapping = np.array(info['city2common'], dtype=np.int)
    else:
        name_classes = np.array(info['label'], dtype=np.str)
        mapping = np.array(info['label2train'], dtype=np.int)  # Not use

    if add_bg_loss:
        num_classes = np.int(info['classes']) + 1
        name_classes = np.array(info['label'] + ["background"], dtype=np.str)

    print(name_classes)
    print("pred path: %s" % os.path.abspath(pred_dir))

    image_path_list = join(devkit_dir, 'data', dset, 'image.txt')
    label_path_list = join(devkit_dir, 'data', dset, 'label.txt')

    gt_imgs = open(label_path_list, 'rb').read().splitlines()
    pred_imgs = open(image_path_list, 'rb').read().splitlines()
    pred_imgs = [os.path.split(x)[-1] for x in pred_imgs]  # frankfurt/frank***.png -> frank***.png

    if is_label16:
        gt_fullpath_list = [os.path.join(gt_dir, fn).replace('labelIds', 'label16IDs') for fn in gt_imgs]
    else:
        gt_fullpath_list = [os.path.join(gt_dir, fn).replace('labelIds', 'gtlabels') for fn in gt_imgs]

    pred_fullpath_list = [os.path.join(pred_dir, fn).replace('gtFine_labelIds', 'leftImg8bit') for fn in pred_imgs]

    calc_all_metrics(name_classes, pred_fullpath_list, gt_fullpath_list, consider_background_loss=add_bg_loss)


def eval_2d3d(pred_dir, add_bg_loss=False):
    with open("./dataset/2d3d_info.json", 'r') as fp:
        info = json.load(fp)

    name_classes = np.array(info['label'], dtype=np.str)

    if add_bg_loss:
        name_classes = np.array(info['label'] + ["background"], dtype=np.str)

    print(name_classes)

    df = pd.read_csv("dataset/Starnford2d3dSemantics/small_test_set.csv")
    gt_fullpath_list = list(df["gt"].values)

    pred_fullpath_list = [os.path.join(pred_dir, os.path.split(x)[-1].replace('semantic', 'rgb'))
                          for x in gt_fullpath_list]

    calc_all_metrics(name_classes, pred_fullpath_list, gt_fullpath_list, consider_background_loss=add_bg_loss)


def eval_sun(pred_dir, add_bg_loss=False):
    with open("./dataset/sun_info.json", 'r') as fp:
        info = json.load(fp)

    name_classes = np.array(info['sun-2d3d-common-label'], dtype=np.str)
    map_2d3d2common = np.array(info['2d3d2common'], dtype=np.int)
    map_sun2common = np.array(info['sun2common'], dtype=np.int)

    if add_bg_loss:
        name_classes = np.array(info['label'] + ["background"], dtype=np.str)

    print(name_classes)

    gt_dir = "/data/unagi0/dataset/SUNRGBD/test13labels"

    gt_fullpath_list = glob.glob(os.path.join(gt_dir, "*"))
    pred_fullpath_list = [os.path.join(pred_dir, os.path.split(x)[-1]) for x in gt_fullpath_list]

    calc_all_metrics(name_classes, pred_fullpath_list, gt_fullpath_list, background_id=255,
                     consider_background_loss=add_bg_loss,
                     gt2common_mat=map_sun2common, pred2common_mat=map_2d3d2common)


def eval_nyu(pred_dir, add_bg_loss=False, use13cls=False, use_nyu_suncg_common_cls=True):
    with open("./dataset/nyu_info.json", 'r') as fp:
        info = json.load(fp)

    name_classes = np.array(info['label'], dtype=np.str)

    if add_bg_loss:
        name_classes = np.array(info['label'] + ["background"], dtype=np.str)

    print(name_classes)

    gt_dir = "/data/unagi0/dataset/NYUDv2/gupta/gt/semantic40"
    pred_fullpath_list = glob.glob(os.path.join(pred_dir, "*"))

    gt_fullpath_list = [os.path.join(gt_dir, os.path.split(x)[-1]) for x in pred_fullpath_list]

    if use13cls:
        mapping = info["40to13cls"]

        correct_cls_list = ["Bed", "Books", "Ceiling", "Chair", "Floor", "Furniture", "Objects", "Picture", "Sofa",
                            "Table", "TV", "Wall", "Window"]
        print("correct class list is ")
        print(correct_cls_list)

        calc_all_metrics(name_classes, pred_fullpath_list, gt_fullpath_list, background_id=255,
                         consider_background_loss=add_bg_loss, gt2common_mat=mapping, pred2common_mat=mapping,
                         out_filename_prefix="13cls")

    elif use_nyu_suncg_common_cls:
        with open("./dataset/suncg_info.json", 'r') as fp:
            suncg_info = json.load(fp)

        ignore_IDs = np.array(suncg_info["ignore_IDs"])

        print ("--- Classes below will be ignored. ---")
        print (name_classes[ignore_IDs])


        gt2common_mat = np.array([[id, 255] for id in ignore_IDs])

        calc_all_metrics(name_classes, pred_fullpath_list, gt_fullpath_list, background_id=255,
                         gt2common_mat=gt2common_mat, consider_background_loss=add_bg_loss, out_filename_prefix="34cls")

    else:
        calc_all_metrics(name_classes, pred_fullpath_list, gt_fullpath_list, background_id=255,
                         consider_background_loss=add_bg_loss, out_filename_prefix="40cls")


def eval_IR(gt_dir, pred_dir, time, split):
    """
    Compute IoU given the predicted colorized images and
    """
    with open(join('dataset', 'ir_info.json'), 'r') as fp:
        ir_info = json.load(fp)
    with open(join('dataset', 'city_info.json'), 'r') as fp:
        city_info = json.load(fp)

    num_classes = np.int(city_info['classes'])
    name_classes = np.array(city_info['label'], dtype=np.str)
    mapping = np.array(ir_info['label2train'], dtype=np.int)
    palette = np.array(city_info['palette'], dtype=np.uint8)
    hist = np.zeros((num_classes, num_classes))
    test_img_id_list = open(join("/data/unagi0/inf_data/ir_seg_dataset", '%s.txt' % split),
                            "rb").read().splitlines()

    if time == "day":
        test_img_id_list = [x for x in test_img_id_list if "D" in x]
    elif time == "night":
        test_img_id_list = [x for x in test_img_id_list if "N" in x]

    print("test image number is %s" % len(test_img_id_list))

    pred_mapping = np.array([
        [14, 13],  # replace "track" with "car"
        [15, 13],  # replace "bus" with "car"
        [12, 11],  # replace "rider" with "person"
        [17, 18]  # replace "motorcycle" with "bicycle"
    ])  # for Adapt
    pred_mapping = np.array(ir_info['label2train'], dtype=np.int)  # for Source Only

    pred_fn_list = [os.path.join(pred_dir, testid + ".png") for testid in test_img_id_list]
    gt_fn_list = [os.path.join(gt_dir, testid + ".png") for testid in test_img_id_list]

    calc_all_metrics(name_classes, pred_fn_list, gt_fn_list, gt2common_mat=mapping,
                     pred2common_mat=pred_mapping,
                     out_filename_prefix="%s_%s" % (split, time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dset', default='city', help='For the challenge use the validation set of cityscapes.',
                        choices=['city', 'ir', "city16", 'gta', '2d3d', "sun", "nyu"])
    parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')
    parser.add_argument('--time', type=str, choices=["day", "night", "all"], default="all",
                        help="only available for ir dataset")
    parser.add_argument('--devkit_dir', default='/data/ugui0/dataset/adaptation/taskcv-2017-public/segmentation',
                        help='base directory of taskcv2017/segmentation')
    parser.add_argument('--add_bg_loss', action="store_true",
                        help='whether you considered background loss when training')
    parser.add_argument('--use13cls', action="store_true",
                        help='nyu or suncg if you convert 40 class to 13 class')
    parser.add_argument('--use40cls', action="store_true",
                        help='whether you use 40cls or not.')

    args = parser.parse_args()

    if args.dset in ["city", "city16"]:
        gt_dir = "/data/ugui0/ksaito/D_A/image_citiscape/www.cityscapes-dataset.com/file-handling/gtFine/val"
        is_label16 = True if args.dset == "city16" else False
        eval_city(gt_dir, args.pred_dir, args.devkit_dir, "cityscapes", add_bg_loss=args.add_bg_loss,
                  is_label16=is_label16)

    elif args.dset == "gta":
        gt_dir = "/data/ugui0/dataset/adaptation/taskcv-2017-public/segmentation/data/"
        eval_city(gt_dir, args.pred_dir, args.devkit_dir, "gta", add_bg_loss=args.add_bg_loss)

    elif args.dset == "ir":
        sep_dir_list = os.path.abspath(args.pred_dir).split(os.path.sep)
        split = sep_dir_list[-3].split("---")[-1].split("-")[-1]
        eval_IR("/data/unagi0/inf_data/ir_seg_dataset/labels", args.pred_dir, args.time, split)

    elif args.dset == "2d3d":
        eval_2d3d(pred_dir=args.pred_dir, add_bg_loss=args.add_bg_loss)

    elif args.dset == "sun":
        eval_sun(pred_dir=args.pred_dir, add_bg_loss=args.add_bg_loss)

    elif args.dset == "nyu":
        eval_nyu(pred_dir=args.pred_dir, add_bg_loss=args.add_bg_loss, use13cls=args.use13cls,
                 use_nyu_suncg_common_cls=not args.use40cls)
