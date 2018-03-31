import argparse
import json
import os
from pprint import pprint

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from argmyparse import add_additional_params_to_args
from datasets import get_dataset, AVAILABLE_DATASET_LIST
from models.model_util import get_full_model
from transform import get_img_transform, get_lbl_transform
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done, save_colorized_lbl, exec_eval, calc_entropy, \
    set_debugger_org_frc

set_debugger_org_frc()

parser = argparse.ArgumentParser(description='Adapt tester for validation data')
parser.add_argument('tgt_dataset', type=str, choices=AVAILABLE_DATASET_LIST)
parser.add_argument('--split', type=str, default='val', help="'val' or 'test')  is used")
parser.add_argument('trained_checkpoint', type=str, metavar="PTH")
parser.add_argument('--outdir', type=str, default="test_output",
                    help='output directory')
parser.add_argument('--test_img_shape', default=None, nargs=2,
                    help="W H, FOR Valid(2048, 1024) Test(1280, 720)")
parser.add_argument("---saves_prob", action="store_true",
                    help='whether you save probability tensors')
args = parser.parse_args()
args = add_additional_params_to_args(args)

if not os.path.exists(args.trained_checkpoint):
    raise OSError("%s does not exist!" % args.resume)

checkpoint = torch.load(args.trained_checkpoint)
try:
    train_args = checkpoint['args']  # Load args!
except KeyError:
    from easydict import EasyDict as edict

    train_args = edict(json.load(open("train_output/city_only_4ch/param_res152.json", 'r')))

model = get_full_model(train_args.net, train_args.res, train_args.n_class, train_args.input_ch)

try:
    model.load_state_dict(checkpoint['state_dict'])
except:
    model.load_state_dict(checkpoint)

print ("----- train args ------")
pprint(train_args.__dict__, indent=4)
print ("-" * 50)
args.train_img_shape = train_args.train_img_shape
print("=> loaded checkpoint '{}'".format(args.trained_checkpoint))

indir, infn = os.path.split(args.trained_checkpoint)

trained_mode = indir.split(os.path.sep)[-2]
args.mode = "%s---%s-%s" % (trained_mode, args.tgt_dataset, args.split)
model_name = infn.replace(".pth", "")

base_outdir = os.path.join(args.outdir, args.mode, model_name)
mkdir_if_not_exist(base_outdir)

json_fn = os.path.join(base_outdir, "param.json")
check_if_done(json_fn)
args.machine = os.uname()[1]
save_dic_to_json(args.__dict__, json_fn)

train_img_shape = tuple([int(x) for x in args.train_img_shape])
test_img_shape = tuple([int(x) for x in args.test_img_shape])

if "crop_size" in train_args.__dict__.keys() and train_args.crop_size > 0:
    train_img_shape = test_img_shape
    print ("train_img_shape was set to the same as test_img_shape")


if "normalize_way" in train_args.__dict__.keys():
    img_transform = get_img_transform(img_shape=train_img_shape, normalize_way=train_args.normalize_way)
else:
    img_transform = get_img_transform(img_shape=train_img_shape)

if "background_id" in train_args.__dict__.keys():
    label_transform = get_lbl_transform(img_shape=train_img_shape, n_class=train_args.n_class,
                                        background_id=train_args.background_id)
else:
    label_transform = get_lbl_transform(img_shape=train_img_shape, n_class=train_args.n_class)

tgt_dataset = get_dataset(dataset_name=args.tgt_dataset, split=args.split, img_transform=img_transform,
                          label_transform=label_transform, test=True, input_ch=train_args.input_ch)

target_loader = data.DataLoader(tgt_dataset, batch_size=1, pin_memory=True)

if torch.cuda.is_available():
    model.cuda()

model.eval()


def add_subdir_if_necessary(outdir, subdir, tgt_dataset):
    if tgt_dataset == "suncg":
        outdir = os.path.join(outdir, subdir)
    return outdir


data_list_fn = os.path.join(base_outdir, "data_list.txt")
with open(data_list_fn, "w") as f:
    pass

total_ent = 0.
for index, (imgs, labels, paths) in tqdm(enumerate(target_loader)):
    path = paths[0]
    imgs = Variable(imgs)
    if torch.cuda.is_available():
        imgs = imgs.cuda()

    preds = model(imgs)

    total_ent += calc_entropy(preds).data.cpu().numpy()[0]

    subdir = path.split('/')[-2]

    if train_args.net == "psp":
        preds = preds[0]

    if args.saves_prob:
        # Save probability tensors
        prob_outdir = os.path.join(base_outdir, "prob")
        prob_outdir = add_subdir_if_necessary(prob_outdir, subdir, args.tgt_dataset)
        mkdir_if_not_exist(prob_outdir)
        prob_outfn = os.path.join(prob_outdir, path.split('/')[-1].replace('png', 'npy'))
        np.save(prob_outfn, preds[0].data.cpu().numpy())

    # Save predicted pixel labels(pngs)
    if train_args.add_bg_loss:
        pred = preds[0, :train_args.n_class].data.max(0)[1].cpu()
    else:
        pred = preds[0, :train_args.n_class - 1].data.max(0)[1].cpu()

    img = Image.fromarray(np.uint8(pred.numpy()))
    img = img.resize(test_img_shape, Image.NEAREST)
    label_outdir = os.path.join(base_outdir, "label")

    label_outdir = add_subdir_if_necessary(label_outdir, subdir, args.tgt_dataset)

    mkdir_if_not_exist(label_outdir)
    label_fn = os.path.join(label_outdir, path.split('/')[-1])
    img.save(label_fn)

    # Save visualized predicted pixel labels(pngs)
    vis_outdir = os.path.join(base_outdir, "vis")
    vis_outdir = add_subdir_if_necessary(vis_outdir, subdir, args.tgt_dataset)
    mkdir_if_not_exist(vis_outdir)
    vis_fn = os.path.join(vis_outdir, path.split('/')[-1])
    save_colorized_lbl(img, vis_fn, args.tgt_dataset)

exec_eval(args.tgt_dataset, label_outdir)

ave_ent = total_ent / len(target_loader)
print ("average entropy: %s" % ave_ent)

with open(os.path.join(base_outdir, "ave_ent_%s.txt" % ave_ent), "w") as f:
    f.write(str(ave_ent))

