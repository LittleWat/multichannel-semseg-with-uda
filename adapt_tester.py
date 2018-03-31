import os
from pprint import pprint

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from argmyparse import add_additional_params_to_args, get_da_mcd_testing_parser
from datasets import get_dataset
from models.model_util import get_models
from transform import get_img_transform, get_lbl_transform
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done, save_colorized_lbl, exec_eval, calc_entropy

parser = get_da_mcd_testing_parser()
args = parser.parse_args()
args = add_additional_params_to_args(args)
# args = add_img_shape_to_args(args)

indir, infn = os.path.split(args.trained_checkpoint)

trained_mode = indir.split(os.path.sep)[-2]
args.mode = "%s---%s-%s" % (trained_mode, args.tgt_dataset, args.split)

model_name = infn.replace(".pth", "")
if args.use_f2:
    model_name += "-use_f2"

print("=> loading checkpoint '{}'".format(args.trained_checkpoint))
if not os.path.exists(args.trained_checkpoint):
    raise OSError("%s does not exist!" % args.trained_checkpoint)

checkpoint = torch.load(args.trained_checkpoint)
train_args = checkpoint["args"]
args.start_epoch = checkpoint['epoch']
print ("----- train args ------")
pprint(checkpoint["args"].__dict__, indent=4)
print ("-" * 50)
print("=> loaded checkpoint '{}'".format(args.trained_checkpoint))

base_outdir = os.path.join(args.outdir, args.mode, model_name)
mkdir_if_not_exist(base_outdir)

json_fn = os.path.join(base_outdir, "param.json")
check_if_done(json_fn)
args.machine = os.uname()[1]
save_dic_to_json(args.__dict__, json_fn)

train_img_shape = tuple([int(x) for x in train_args.train_img_shape])
test_img_shape = tuple([int(x) for x in args.test_img_shape])

if "normalize_way" in train_args.__dict__.keys():
    img_transform = get_img_transform(img_shape=train_img_shape,
                                      normalize_way=train_args.normalize_way)
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

try:
    G, F1, F2 = get_models(net_name=train_args.net, res=train_args.res, input_ch=train_args.input_ch,
                           n_class=train_args.n_class,
                           method=train_args.method, is_data_parallel=train_args.is_data_parallel)
except AttributeError:
    G, F1, F2 = get_models(net_name=train_args.net, res=train_args.res, input_ch=train_args.input_ch,
                           n_class=train_args.n_class,
                           method="MCD", is_data_parallel=False)

G.load_state_dict(checkpoint['g_state_dict'])
F1.load_state_dict(checkpoint['f1_state_dict'])

if args.use_f2:
    F2.load_state_dict(checkpoint['f2_state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
      .format(args.trained_checkpoint, checkpoint['epoch']))

G.eval()
F1.eval()
F2.eval()

if torch.cuda.is_available():
    G.cuda()
    F1.cuda()
    F2.cuda()

total_ent = 0.
for index, (imgs, _, paths) in tqdm(enumerate(target_loader)):
    path = paths[0]

    imgs = Variable(imgs)
    if torch.cuda.is_available():
        imgs = imgs.cuda()

    feature = G(imgs)
    outputs = F1(feature)

    if args.use_f2:
        outputs += F2(feature)
        outputs /= 2

    total_ent += calc_entropy(outputs).data.cpu().numpy()[0]

    if args.saves_prob:
        # Save probability tensors
        prob_outdir = os.path.join(base_outdir, "prob")
        mkdir_if_not_exist(prob_outdir)
        prob_outfn = os.path.join(prob_outdir, path.split('/')[-1].replace('png', 'npy'))
        np.save(prob_outfn, outputs[0].data.cpu().numpy())

    # Save predicted pixel labels(pngs)
    if train_args.add_bg_loss:
        pred = outputs[0, :args.n_class].data.max(0)[1].cpu()
    else:
        pred = outputs[0, :args.n_class - 1].data.max(0)[1].cpu()

    img = Image.fromarray(np.uint8(pred.numpy()))
    img = img.resize(test_img_shape, Image.NEAREST)
    label_outdir = os.path.join(base_outdir, "label")

    mkdir_if_not_exist(label_outdir)
    label_fn = os.path.join(label_outdir, path.split('/')[-1])
    img.save(label_fn)

    # Save visualized predicted pixel labels(pngs)
    vis_outdir = os.path.join(base_outdir, "vis")
    mkdir_if_not_exist(vis_outdir)
    vis_fn = os.path.join(vis_outdir, path.split('/')[-1])
    save_colorized_lbl(img, vis_fn, args.tgt_dataset)

exec_eval(args.tgt_dataset, label_outdir)

ave_ent = total_ent / len(target_loader)
print ("average entropy: %s" % ave_ent)

with open(os.path.join(base_outdir, "ave_ent_%s.txt" % ave_ent), "w") as f:
    f.write(str(ave_ent))