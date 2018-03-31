import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from datasets import get_dataset
from transform import Scale, Colorize
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done

parser = argparse.ArgumentParser(description='Adapt tester for validation data')
parser.add_argument('tgt_dataset', type=str, choices=["gta", "city", "test", "ir"])
parser.add_argument('trained_model_path', type=str, metavar="PTH")
parser.add_argument('--outdir', type=str, default="test_output",
                    help='output directory')
parser.add_argument('--train_img_shape', default=(1024, 512), nargs=2,
                    help="W H")
parser.add_argument('--test_img_shape', default=(2048, 1024), nargs=2,
                    help="W H, FOR Valid(2048, 1024) Test(1280, 720)")
parser.add_argument('--net', type=str, default="fcn",
                    help="choose from ['fcn', 'psp', 'segnet']")
parser.add_argument('--res', type=str, default='50',
                    help='which resnet 18,50,101,152')
parser.add_argument("--input_ch", type=int, default=3,
                    choices=[1, 3, 4])
parser.add_argument("--n_class", type=int, default=20, help="the number of classes")
parser.add_argument('--split', type=str, default='val', help="'val' or 'test')  is used")

args = parser.parse_args()

indir, infn = os.path.split(args.trained_model_path)

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

img_transform = Compose([
    Scale(train_img_shape, Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),

])
label_transform = Compose([Scale(test_img_shape, Image.BILINEAR), ToTensor()])

tgt_dataset = get_dataset(dataset_name=args.tgt_dataset, split=args.split, img_transform=img_transform,
                          label_transform=label_transform, test=True, input_ch=args.input_ch)

target_loader = data.DataLoader(tgt_dataset, batch_size=1, pin_memory=True)

if args.net == "fcn":
    from models.fcn import ResFCN

    model = torch.nn.DataParallel(ResFCN(args.n_class, args.res, args.input_ch))
else:
    raise NotImplementedError("Only FCN, SegNet, PSPNet are supported!")

model.load_state_dict(torch.load(args.trained_model_path))
model.eval()

if torch.cuda.is_available():
    model.cuda()

for index, (imgs, labels, paths) in tqdm(enumerate(target_loader)):
    path = paths[0]
    imgs = Variable(imgs)
    if torch.cuda.is_available():
        imgs = imgs.cuda()

    preds = model(imgs)

    # Save probability tensors
    prob_outdir = os.path.join(base_outdir, "prob")
    mkdir_if_not_exist(prob_outdir)
    prob_outfn = os.path.join(prob_outdir, path.split('/')[-1].replace('png', 'npy'))
    np.save(prob_outfn, preds[0].data.cpu().numpy())

    # Save predicted pixel labels(pngs)
    pred = preds[0, :args.n_class - 1].data.max(0)[1].cpu().numpy()
    img = Image.fromarray(np.uint8(pred))
    img = img.resize(test_img_shape, Image.NEAREST)
    label_outdir = os.path.join(base_outdir, "label")
    mkdir_if_not_exist(label_outdir)
    label_fn = os.path.join(label_outdir, path.split('/')[-1])
    img.save(label_fn)

    # Save visualized predicted pixel labels(pngs)
    preds = preds[0, :args.n_class - 1].data.max(0)[1]
    preds = preds.view(1, preds.size()[0], preds.size()[1])
    output = Colorize()(preds)
    output = np.transpose(output.cpu().numpy(), (1, 2, 0))
    img = Image.fromarray(output, "RGB")
    img = img.resize(test_img_shape, Image.NEAREST)
    vis_outdir = os.path.join(base_outdir, "vis")
    mkdir_if_not_exist(vis_outdir)
    vis_fn = os.path.join(vis_outdir, path.split('/')[-1])
    img.save(vis_fn)
