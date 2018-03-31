import collections
import glob
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from torch.utils import data
from torchvision.transforms import Compose
from tqdm import tqdm

from transform import HorizontalFlip, VerticalFlip, ReLabel
from util import emphasize_str

AVAILABLE_DATASET_LIST = ["gta", "city", "test", "ir", "city16", "synthia", "2d3d", "sun", "suncg", "nyu"]


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def default_loader(path):
    return Image.open(path)


class CityDataSet(data.Dataset):
    IMG_SIZE = [2048, 1024]
    PSEUDO_IR_DIR = "/data/unagi0/dataset/cityscapes/pseudo_ir/"

    def __init__(self, root, split="train", img_transform=None, label_transform=None, test=True, input_ch=3,
                 label_type=None, joint_transform=None):
        assert input_ch in [1, 3, 4]
        self.input_ch = input_ch
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform
        self.test = test
        self.use_pseudo_ir = True if "pseudo_ir" in split else False  # train_pseudo_ir
        if self.use_pseudo_ir:
            assert input_ch == 4
            print ("pseudo_ir will be used!")
            split = split.split("_")[0]
            self.split = split

        data_dir = root
        # for split in ["train", "trainval", "val"]:
        imgsets_dir = osp.join(data_dir, "leftImg8bit/%s.txt" % self.split)
        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                img_file = osp.join(data_dir, "leftImg8bit/%s" % name)
                if label_type == "label16":
                    name = name.replace('leftImg8bit', 'gtFine_label16IDs')
                else:
                    name = name.replace('leftImg8bit', 'gtFine_labelTrainIds')
                label_file = osp.join(data_dir, "gtFine/%s" % name)
                self.files[self.split].append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        np3ch = np.array(img)
        if self.input_ch == 1:
            img = ImageOps.grayscale(img)

        elif self.input_ch == 4:
            if self.use_pseudo_ir:
                im_name = os.path.split(img_file)[-1]
                pseudo_ir_name = os.path.join(self.PSEUDO_IR_DIR, self.split, im_name)
                ir_img = np.array(Image.open(pseudo_ir_name))[:, :, np.newaxis]
                extended_np3ch = np.concatenate([np3ch, ir_img], axis=2)
            else:
                extended_np3ch = np.concatenate([np3ch, np3ch[:, :, 0:1]], axis=2)
            img = Image.fromarray(np.uint8(extended_np3ch))

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.joint_transform:
            img, label = self.joint_transform(img, label)

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file

        return img, label


# TODO support joint_transform
class GTADataSet(data.Dataset):
    IMG_SIZE = [1914, 1052]

    def __init__(self, root, split="images", img_transform=None, label_transform=None,
                 test=False, input_ch=3, joint_transform=None):
        # Note; split "train" and "images" are SAME!!!

        assert split in ["images", "test", "train"]

        assert input_ch in [1, 3, 4]
        self.input_ch = input_ch
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root

        imgsets_dir = osp.join(data_dir, "%s.txt" % split)

        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                img_file = osp.join(data_dir, "%s" % name)
                # name = name.replace('leftImg8bit','gtFine_labelTrainIds')
                label_file = osp.join(data_dir, "%s" % name.replace('images', 'labels_gt'))
                self.files[split].append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        np3ch = np.array(img)
        if self.input_ch == 1:
            img = ImageOps.grayscale(img)

        elif self.input_ch == 4:
            extended_np3ch = np.concatenate([np3ch, np3ch[:, :, 0:1]], axis=2)
            img = Image.fromarray(np.uint8(extended_np3ch))

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.joint_transform:
            img, label = self.joint_transform(img, label)

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file

        return img, label


class SynthiaDataSet(data.Dataset):
    IMG_SIZE = [1280, 760]

    def __init__(self, root, split="all", img_transform=None, label_transform=None,
                 test=False, input_ch=3, joint_transform=None):
        # TODO this does not support "split" parameter

        assert input_ch in [1, 3, 4]
        self.input_ch = input_ch
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform
        self.test = test

        rgb_dir = osp.join(root, "RGB")
        gt_dir = osp.join(root, "GT", "LABELS16")

        rgb_fn_list = glob.glob(osp.join(rgb_dir, "*.png"))
        gt_fn_list = glob.glob(osp.join(gt_dir, "*.png"))

        for rgb_fn, gt_fn in zip(rgb_fn_list, gt_fn_list):
            self.files[split].append({
                "rgb": rgb_fn,
                "label": gt_fn
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]
        img_file = datafiles["rgb"]
        img = Image.open(img_file).convert('RGB')
        np3ch = np.array(img)
        if self.input_ch == 1:
            img = ImageOps.grayscale(img)

        elif self.input_ch == 4:
            extended_np3ch = np.concatenate([np3ch, np3ch[:, :, 0:1]], axis=2)
            img = Image.fromarray(np.uint8(extended_np3ch))

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.joint_transform:
            img, label = self.joint_transform(img, label)

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file

        return img, label


def depth_scaling(np_img, depmin=None, depmax=None):
    depmax = depmax if depmax else np_img.max()
    depmin = depmax if depmin else np_img.min()
    processed_np_img = (np_img.astype(np.float64) - depmin) / (depmax - depmin)
    return np.uint8(processed_np_img * 255)


# TODO support joint_transform
class Stanford2D3DSemanticsDataSet(data.Dataset):
    """
    {'<UNK>': 255,
     'beam': 0,
     'board': 1,
     'bookcase': 2,
     'ceiling': 3,
     'chair': 4,
     'clutter': 5,
     'column': 6,
     'door': 7,
     'floor': 8,
     'sofa': 9,
     'table': 10,
     'wall': 11,
     'window': 12}
    """
    ## split info: http://buildingparser.stanford.edu/dataset.html#sample

    TRAIN_SPLIT_AREA_LIST = ["area_1", "area_2", "area_3", "area_4", "area_6"]
    TEST_SPLIT_TRAIN_SPLIT_AREA_LIST = ["area_5a", "area_5b"]
    IMG_SIZE = [1080, 1080]
    SEED = 42

    def check_split_and_input_ch(self, split, input_ch):
        modal = split.split("_")[-1]
        if modal == "rgbd":
            assert input_ch == 4
            print ("4ch is Depth channel")

        elif modal == "d":
            assert input_ch == 1

        if modal == "rgb" and input_ch == 4:
            print (emphasize_str("4ch is R channel"))

    def __init__(self, root, split="train_rgb", img_transform=None, label_transform=None,
                 test=False, input_ch=3, joint_transform=None):
        assert split in ["train_rgb", "train_rgbd", "test_rgb", "test_rgbd", "small_test_rgb", "small_test_rgbd",
                         "train_d", "small_test_d"]  ## "test_rgb", "test_rgbd"
        assert input_ch in [1, 3, 4]
        self.check_split_and_input_ch(split, input_ch)

        self.input_ch = input_ch
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform
        self.test = test

        def get_fn_list_from_area_list(area_list, modal):
            img_fn_list = []
            for train_area in area_list:
                img_dir = osp.join(root, "XYZ", train_area, "data", modal)
                img_fn_list.extend(glob.glob(osp.join(img_dir, "*.png")))

            return img_fn_list

        self.area_list = self.TRAIN_SPLIT_AREA_LIST if "train" in split else self.TEST_SPLIT_TRAIN_SPLIT_AREA_LIST
        rgb_fn_list = sorted(get_fn_list_from_area_list(self.area_list, "rgb"))
        depth_fn_list = sorted(get_fn_list_from_area_list(self.area_list, "depth"))
        gt_fn_list = sorted(get_fn_list_from_area_list(self.area_list, "cls_lbl"))

        len_rgb_fn_list = len(rgb_fn_list)
        assert len_rgb_fn_list == len(depth_fn_list)
        assert len_rgb_fn_list == len(gt_fn_list)

        if "small" in split:
            np.random.seed(self.SEED)
            on_idxes = np.random.permutation(len_rgb_fn_list)[:self.N_SMALL]
            rgb_fn_list = [str(x) for x in list(np.array(rgb_fn_list)[on_idxes])]
            depth_fn_list = [str(x) for x in list(np.array(depth_fn_list)[on_idxes])]
            gt_fn_list = [str(x) for x in list(np.array(gt_fn_list)[on_idxes])]

            import pandas as pd
            df = pd.DataFrame({
                "rgb": rgb_fn_list,
                "depth": depth_fn_list,
                "gt": gt_fn_list
            })
            outfn = osp.join("dataset", "Starnford2d3dSemantics", "small_test_set.csv")
            df.to_csv(outfn, index=False)
            print ("%s was saved" % outfn)

        # Check the order
        def get_id_from_filename(filename):
            return os.path.split(filename)[-1].split("_")[1]

        rand_id = np.random.randint(len(rgb_fn_list))
        assert get_id_from_filename(rgb_fn_list[rand_id]) == get_id_from_filename(depth_fn_list[rand_id])
        assert get_id_from_filename(rgb_fn_list[rand_id]) == get_id_from_filename(gt_fn_list[rand_id])

        for rgb_fn, depth_fn, gt_fn in zip(rgb_fn_list, depth_fn_list, gt_fn_list):
            self.files[split].append({
                "rgb": rgb_fn,
                "depth": depth_fn,
                "label": gt_fn
            })

        print ("N_%s: %s" % (split, len(self.files[split])))

    def __len__(self):
        return len(self.files[self.split])

    def depth_img_preprocess(self, np_depth):
        """
        Replace exception value (65535) with maximum depth value
        Then Scaile to 0-255
        :param np_depth:
        :return: processed_np_depth
        """
        idxes = np.where(np_depth == 65535)
        np_depth[idxes] = -1
        max_dep = np_depth.max()
        np_depth[idxes] = max_dep
        np_depth = depth_scaling(np_depth)
        return np_depth

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        if self.input_ch == 1 and "d" in self.split:
            img_file = datafiles["depth"]
            img = Image.open(img_file)
            np_img = self.depth_img_preprocess(np.array(img))
            img = Image.fromarray(np_img)
            # self.img_transform[-1] = Normalize([.485, .456, .406], [.229, .224, .225])
            # print ("img transform changed")
        else:
            img_file = datafiles["rgb"]
            img = Image.open(img_file)
            np3ch = np.array(img)

            if self.input_ch == 1:
                img = ImageOps.grayscale(img)

            elif self.input_ch == 4:
                if "rgbd" in self.split:
                    depth_file = datafiles["depth"]
                    np_depth = np.array(Image.open(depth_file))
                    np_depth = self.depth_img_preprocess(np_depth)

                    extended_np3ch = np.concatenate([np3ch, np_depth[:, :, np.newaxis]], axis=2)
                    # print ("4ch is Depth channel")
                else:
                    extended_np3ch = np.concatenate([np3ch, np3ch[:, :, 0:1]], axis=2)
                    # print ("4ch is R channel")

                img = Image.fromarray(np.uint8(extended_np3ch))
                # img = Image.fromarray(np.uint32(extended_np3ch))
                # img = extended_np3ch

        # img = np.array(img)
        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")
        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file

        return img, label


# TODO support joint_transform
class SUNRGBDDataSet(data.Dataset):
    """
    {'Bed': 1,
    'Books': 2,
    'Ceiling': 3,
    'Chair': 4,
    'Floor': 5,
    'Furniture': 6,
    'Objects': 7,
    'Picture': 8,
    'Sofa': 9,
    'Table': 10,
    'TV': 11,
    'Wall': 12,
    'Window': 13}
    """
    ## split info: http://buildingparser.stanford.edu/dataset.html#sample

    IMG_SIZE = [730, 530]
    SEED = 42

    def check_split_and_input_ch(self, split, input_ch):
        modal = split.split("_")[-1]
        if modal == "rgbd":
            assert input_ch == 4
            print ("4ch is Depth channel")

        elif modal == "d":
            assert input_ch == 1

        if modal == "rgb" and input_ch == 4:
            print (emphasize_str("4ch is R channel"))

    def __init__(self, root, split="test_rgbd", img_transform=None, label_transform=None,
                 test=False, input_ch=3, joint_transform=None):
        assert split in ["test_d", "test_rgb", "test_rgbd"]
        assert input_ch in [1, 3, 4]
        self.check_split_and_input_ch(split, input_ch)

        self.input_ch = input_ch
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform
        self.test = test

        rgb_dir = osp.join(root, "SUNRGBD-test_images")
        depth_dir = osp.join(root, "sunrgbd_test_depth_new")
        gt_dir = osp.join(root, "test13labels")
        rgb_fn_list = sorted(glob.glob(osp.join(rgb_dir, "*")))
        depth_fn_list = sorted(glob.glob(osp.join(depth_dir, "*")))
        gt_fn_list = sorted(glob.glob(osp.join(gt_dir, "*")))

        len_rgb_fn_list = len(rgb_fn_list)
        assert len_rgb_fn_list == len(depth_fn_list)
        assert len_rgb_fn_list == len(gt_fn_list)

        # Check the order
        def get_id_from_filename(filename):
            return os.path.split(filename)[-1].split("-")[1].replace(".png", "").replace(".jpg", "")

        rand_id = np.random.randint(len(rgb_fn_list))
        assert get_id_from_filename(rgb_fn_list[rand_id]) == get_id_from_filename(depth_fn_list[rand_id])
        assert get_id_from_filename(rgb_fn_list[rand_id]) == get_id_from_filename(gt_fn_list[rand_id])

        for rgb_fn, depth_fn, gt_fn in zip(rgb_fn_list, depth_fn_list, gt_fn_list):
            self.files[split].append({
                "rgb": rgb_fn,
                "depth": depth_fn,
                "label": gt_fn
            })

        print ("N_%s: %s" % (split, len(self.files[split])))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        if self.input_ch == 1 and "d" in self.split:
            img_file = datafiles["depth"]
            img = Image.open(img_file)
            np_img = depth_scaling(np.array(img))
            img = Image.fromarray(np_img)
            # self.img_transform[-1] = Normalize([.485, .456, .406], [.229, .224, .225])
            # print ("img transform changed")
        else:
            img_file = datafiles["rgb"]
            img = Image.open(img_file)
            np3ch = np.array(img)

            if self.input_ch == 1:
                img = ImageOps.grayscale(img)

            elif self.input_ch == 4:
                if "rgbd" in self.split:
                    depth_file = datafiles["depth"]
                    np_depth = np.array(Image.open(depth_file))
                    np_depth = depth_scaling(np_depth)

                    extended_np3ch = np.concatenate([np3ch, np_depth[:, :, np.newaxis]], axis=2)
                    # print ("4ch is Depth channel")
                else:
                    extended_np3ch = np.concatenate([np3ch, np3ch[:, :, 0:1]], axis=2)
                    # print ("4ch is R channel")

                img = Image.fromarray(np.uint8(extended_np3ch))
                # img = Image.fromarray(np.uint32(extended_np3ch))
                # img = extended_np3ch

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, label_file

        return img, label


# TODO support joint_transform
class SUNCGDataSet(data.Dataset):
    """
    40 classes same as NYUDv2
    """
    ## split info: http://buildingparser.stanford.edu/dataset.html#sample

    IMG_SIZE = [640, 480]
    SEED = 42

    def check_split_and_input_ch(self, split, input_ch):
        modal = split.split("_")[-1]
        if modal == "rgbd":
            assert input_ch == 4
            print ("4ch is Depth channel")

        elif modal == "d":
            assert input_ch == 1

        elif modal == "rgbhha":
            assert input_ch == 6

        if modal == "rgb" and input_ch == 4:
            print (emphasize_str("4ch is R channel"))

    def __init__(self, root, split="train_rgbhha", img_transform=None, label_transform=None,
                 test=False, input_ch=3, extra_img_transform=None, joint_transform=None):
        assert split in ["train_rgb", "train_hha", "train_rgbhha", "train_rgbhhab"]
        assert input_ch in [1, 3, 4, 6, 7]
        self.check_split_and_input_ch(split, input_ch)

        self.input_ch = input_ch
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.extra_img_transform = extra_img_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform
        self.test = test

        rgb_dir = osp.join(root, "mlt_v2")  # or opengl_ver2
        depth_dir = osp.join(root, "depth_v2")
        hha_dir = osp.join(root, "hha_v2")
        gt_dir = osp.join(root, "category_v2")
        boundary_dir = osp.join(root, "boundary_v2")

        fn_id_path = osp.join(root, "data_goodlist_v2.txt")
        with open(fn_id_path) as f:
            fn_id_list = [x.strip() for x in f.readlines()]

        print ("SUNCG Dataset Loading filenames...")
        for fn_id in tqdm(fn_id_list):
            self.files[split].append({
                "rgb": osp.join(rgb_dir, fn_id + "_mlt.png"),
                "depth": osp.join(depth_dir, fn_id + "_depth.png"),
                "hha": osp.join(hha_dir, fn_id + "_hha.png"),
                "label": osp.join(gt_dir, fn_id + "_category40.png"),
                "boundary": osp.join(boundary_dir, fn_id + "_instance_boundary.png")
            })

        print ("N_%s: %s" % (split, len(self.files[split])))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        if self.input_ch == 1:
            if "d" in self.split:
                img = Image.open(datafiles["depth"])
                np_img = depth_scaling(np.array(img))
                img = Image.fromarray(np_img)
                if self.img_transform:
                    img = self.img_transform(img)
            else:
                raise NotImplementedError()

        elif self.input_ch == 3:
            if "hha" in self.split:
                hha = Image.open(datafiles["hha"])
                if self.extra_img_transform:
                    img = self.extra_img_transform(hha)
                else:
                    img = self.img_transform(hha)
            # RGB
            else:
                img = Image.open(datafiles["rgb"])

                if self.img_transform:
                    img = self.img_transform(img)

        elif self.input_ch == 4:
            img = Image.open(datafiles["rgb"])
            np3ch = np.array(img)
            if "rgbd" in self.split:
                depth_file = datafiles["depth"]
                np_depth = np.array(Image.open(depth_file))
                np_depth = depth_scaling(np_depth)
                extended_np3ch = np.concatenate([np3ch, np_depth[:, :, np.newaxis]], axis=2)
                # print ("4ch is Depth channel")

            # RGBR
            else:
                extended_np3ch = np.concatenate([np3ch, np3ch[:, :, 0:1]], axis=2)
                # print ("4ch is R channel")
            img = Image.fromarray(np.uint8(extended_np3ch))
            if self.img_transform:
                img = self.img_transform(img)


        # RGBHHA
        elif self.input_ch == 6:
            rgb = Image.open(datafiles["rgb"])
            rgb = self.img_transform(rgb)

            hha = Image.open(datafiles["hha"])
            if self.extra_img_transform:
                hha = self.extra_img_transform(hha)
            else:
                hha = self.img_transform(hha)

            img = torch.cat([rgb, hha])

        # RGBHHAB
        elif self.input_ch == 7:
            rgb = Image.open(datafiles["rgb"])
            rgb = self.img_transform(rgb)

            hha = Image.open(datafiles["hha"])
            if self.extra_img_transform:
                hha = self.extra_img_transform(hha)
            else:
                hha = self.img_transform(hha)

            convert_to_torch_tensor = Compose(
                self.label_transform.transforms[:-1] + [ReLabel(255, 1)])  # Scale, ToTensor (Without Normalize)
            boundary = convert_to_torch_tensor(Image.open(datafiles["boundary"])).unsqueeze(0)
            # boundary = self.label_transform(Image.open(datafiles["boundary"]).convert("P")).unsqueeze(0)

            img = torch.cat([rgb, hha, boundary.float()])


        else:
            raise NotImplementedError()

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, label_file

        return img, label


# TODO support joint_transform
class NYUDv2(data.Dataset):
    """
    40 classes
    """
    ## split info: http://buildingparser.stanford.edu/dataset.html#sample

    IMG_SIZE = [560, 425]
    SEED = 42

    def check_split_and_input_ch(self, split, input_ch):
        modal = split.split("_")[-1]
        if modal == "rgbd":
            assert input_ch == 4
            print ("4ch is Depth channel")

        elif modal == "d":
            assert input_ch == 1

        elif modal == "rgbhha":
            assert input_ch == 6

        if modal == "rgb" and input_ch == 4:
            print (emphasize_str("4ch is R channel"))

    def __init__(self, root, split="all_rgbhha", img_transform=None, label_transform=None,
                 test=False, input_ch=3, extra_img_transform=None, joint_transform=None):

        assert split in ["all_rgb", "all_hha", "all_rgbhha", "trainval_rgb", "trainval_hha", "trainval_rgbhha",
                         "test_rgb", "test_hha", "test_rgbhha"]

        raw_split = split.split("_")[0]
        assert input_ch in [1, 3, 6]
        self.check_split_and_input_ch(split, input_ch)

        self.input_ch = input_ch
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.extra_img_transform = extra_img_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform
        self.test = test

        rgb_dir = osp.join(root, "rgb")
        hha_dir = osp.join(root, "hha")
        gt_dir = osp.join(root, "gt", "semantic40")

        if raw_split == "all":
            rgb_fn_list = sorted(glob.glob(osp.join(rgb_dir, "*")))
            gt_fn_list = sorted(glob.glob(osp.join(gt_dir, "*")))
            hha_fn_list = sorted(glob.glob(osp.join(hha_dir, "*")))
        else:
            with open("/data/unagi0/dataset/NYUDv2/gupta/%s.txt" % raw_split) as f:
                id_list = f.readlines()
            id_list = [x.strip() for x in id_list]
            rgb_fn_list = [osp.join(rgb_dir, "img_%s.png" % x.strip()) for x in id_list]
            gt_fn_list = [osp.join(gt_dir, "img_%s.png" % x.strip()) for x in id_list]
            hha_fn_list = [osp.join(hha_dir, "img_%s.png" % x.strip()) for x in id_list]

        len_rgb_fn_list = len(rgb_fn_list)
        assert len_rgb_fn_list == len(gt_fn_list)
        assert len_rgb_fn_list == len(hha_fn_list)

        rand_id = np.random.randint(len(rgb_fn_list))

        print (rgb_fn_list[rand_id], gt_fn_list[rand_id], hha_fn_list[rand_id])

        for rgb_fn, hha_fn, gt_fn in zip(rgb_fn_list, hha_fn_list, gt_fn_list):
            self.files[split].append({
                "rgb": rgb_fn,
                "hha": hha_fn,
                "label": gt_fn
            })

        print ("N_%s: %s" % (split, len(self.files[split])))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]
        if "hha" not in self.split:
            if self.input_ch == 1 and "d" in self.split:
                img_file = datafiles["depth"]
                img = Image.open(img_file)
                np_img = depth_scaling(np.array(img))
                img = Image.fromarray(np_img)
            else:
                img_file = datafiles["rgb"]
                img = Image.open(img_file)
                np3ch = np.array(img)

                if self.input_ch == 1:
                    img = ImageOps.grayscale(img)

                elif self.input_ch == 4:
                    if "rgbd" in self.split:
                        depth_file = datafiles["depth"]
                        np_depth = np.array(Image.open(depth_file))
                        np_depth = depth_scaling(np_depth)
                        extended_np3ch = np.concatenate([np3ch, np_depth[:, :, np.newaxis]], axis=2)
                        # print ("4ch is Depth channel")
                    else:
                        extended_np3ch = np.concatenate([np3ch, np3ch[:, :, 0:1]], axis=2)
                        # print ("4ch is R channel")

                    img = Image.fromarray(np.uint8(extended_np3ch))

            if self.img_transform:
                img = self.img_transform(img)

        else:
            hha = Image.open(datafiles["hha"])
            if self.extra_img_transform:
                hha = self.extra_img_transform(hha)
            else:
                hha = self.img_transform(hha)

            if self.input_ch == 3:
                img = hha
            elif self.input_ch == 6:
                rgb = Image.open(datafiles["rgb"])
                rgb = self.img_transform(rgb)
                img = torch.cat([rgb, hha])
            else:
                raise NotImplementedError()

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, label_file

        return img, label


class TestDataSet(data.Dataset):
    IMG_SIZE = [1280, 720]

    def __init__(self, root, split="train", img_transform=None, label_transform=None, test=True, input_ch=3,
                 joint_transform=None):
        assert input_ch == 3
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root
        # for split in ["train", "trainval", "val"]:
        imgsets_dir = os.listdir(data_dir)
        for name in imgsets_dir:
            img_file = osp.join(data_dir, "%s" % name)
            self.files[split].append({
                "img": img_file,
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        if self.joint_transform:
            img, img = self.joint_transform(img, img)

        if self.img_transform:
            img = self.img_transform(img)

        if self.test:
            return img, 'hoge', img_file
        else:
            # TODO fix the second img. img, None causes bug, but this statement is strange.
            return img, img


class IRDataSet(data.Dataset):
    IMG_SIZE = [640, 480]

    def __init__(self, root, split="train", img_transform=None, label_transform=None,
                 test=False, input_ch=4, joint_transform=None):
        assert input_ch in [1, 3, 4]
        self.input_ch = input_ch
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform
        self.test = test
        img_dir = osp.join(root, "images")
        lbl_dir = osp.join(root, "labels_converted2GTA")

        # for split in ["train", "trainval", "val"]:
        img_set_fn = osp.join(root, "%s.txt" % split)

        with open(img_set_fn) as img_set_list:
            for name in img_set_list:
                name = name.strip()
                img_file = osp.join(img_dir, "%s.png" % name)
                label_file = osp.join(lbl_dir, "%s.png" % name)
                self.files[split].append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]

        img = Image.open(img_file)  # RGB+FIR(4ch)
        np4ch = np.array(img)

        if self.input_ch == 3:
            img = Image.fromarray(np.uint8(np4ch[:, :, :3]))  # RGB
        elif self.input_ch == 1:
            img = Image.fromarray(np.uint8(np4ch[:, :, -1]))  # FIR

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.joint_transform:
            img, label = self.joint_transform(img, label)

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file
        else:
            return img, label


def get_dataset(dataset_name, split, img_transform, label_transform, test, input_ch=3, joint_transform=None):
    assert dataset_name in AVAILABLE_DATASET_LIST

    name2obj = {
        "gta": GTADataSet,
        "city": CityDataSet,
        "city16": CityDataSet,
        "test": TestDataSet,
        "ir": IRDataSet,
        "synthia": SynthiaDataSet,
        "2d3d": Stanford2D3DSemanticsDataSet,
        "sun": SUNRGBDDataSet,
        "suncg": SUNCGDataSet,
        "nyu": NYUDv2,
    }
    name2root = {
        "gta": "/data/ugui0/dataset/adaptation/taskcv-2017-public/segmentation/data/",
        "city": "/data/ugui0/ksaito/D_A/image_citiscape/www.cityscapes-dataset.com/file-handling/",
        "city16": "/data/ugui0/ksaito/D_A/image_citiscape/www.cityscapes-dataset.com/file-handling/",
        "test": "/data/ugui0/dataset/adaptation/segmentation_test",
        "ir": "/data/unagi0/inf_data/ir_seg_dataset",
        "synthia": "/data/ugui0/dataset/adaptation/synthia/RAND_CITYSCAPES",
        "2d3d": "/data/unagi0/dataset/2D-3D-SemanticsData/",
        "sun": "/data/unagi0/dataset/SUNRGBD/",
        "suncg": "/data/unagi0/dataset/SUNCG-Seg/",
        "nyu": "/data/unagi0/dataset/NYUDv2/gupta/"
    }
    dataset_obj = name2obj[dataset_name]
    root = name2root[dataset_name]

    if dataset_name == "city16":
        return dataset_obj(root=root, split=split, img_transform=img_transform, label_transform=label_transform,
                           test=test, input_ch=input_ch, label_type="label16", joint_transform=joint_transform)

    return dataset_obj(root=root, split=split, img_transform=img_transform, label_transform=label_transform,
                       test=test, input_ch=input_ch, joint_transform=joint_transform)


def check_src_tgt_ok(src_dataset_name, tgt_dataset_name):
    if src_dataset_name == "synthia" and not tgt_dataset_name == "city16":
        raise AssertionError("you must use synthia-city16 pair")
    elif src_dataset_name == "city16" and not tgt_dataset_name == "synthia":
        raise AssertionError("you must use synthia-city16 pair")


def get_n_class(src_dataset_name):
    if src_dataset_name in ["synthia", "city16"]:
        return 16
    elif src_dataset_name in ["gta", "city", "ir", "test"]:
        return 19 + 1
    elif src_dataset_name in ["2d3d"]:
        return 13 + 1
    elif src_dataset_name in ["sun"]:
        return 13 + 1

    elif src_dataset_name in ["suncg", "nyu"]:
        return 40 + 1
    else:
        raise NotImplementedError("You have to define the class of %s dataset" % src_dataset_name)


def get_img_shape(dataset, is_train):
    dataset2img_shape = {
        "gta": GTADataSet.IMG_SIZE,
        "synthia": SynthiaDataSet.IMG_SIZE,
        "city": CityDataSet.IMG_SIZE,
        "city16": CityDataSet.IMG_SIZE,
        "ir": IRDataSet.IMG_SIZE,
        "2d3d": Stanford2D3DSemanticsDataSet.IMG_SIZE,
        "sun": SUNRGBDDataSet.IMG_SIZE,
        "suncg": SUNCGDataSet.IMG_SIZE,
        "nyu": NYUDv2.IMG_SIZE
    }
    if is_train:
        dataset2img_shape["city"] = list(np.array(dataset2img_shape["city"]) / 2)
        dataset2img_shape["city16"] = list(np.array(dataset2img_shape["city16"]) / 2)

    return dataset2img_shape[dataset]
