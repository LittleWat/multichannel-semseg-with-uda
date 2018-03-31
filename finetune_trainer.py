from __future__ import division

import argparse
import os

import torch
from PIL import Image
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from datasets import get_dataset
from loss import CrossEntropyLoss2d
from transform import ReLabel, ToLabel, Scale, RandomSizedCrop, RandomHorizontalFlip, RandomRotation
from util import check_if_done
from util import mkdir_if_not_exist, save_dic_to_json
from visualize import LinePlotter

parser = argparse.ArgumentParser(description='PyTorch Segmentation Adaptation')
parser.add_argument('src_dataset', type=str, choices=["gta", "city", "ir"])
parser.add_argument('g_path', type=str)
parser.add_argument('--savename', type=str, default="normal", help='save name')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train (default: 40)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 0.001)')
parser.add_argument('--res', type=str, default='50', metavar="ResnetLayerNum",
                    choices=["18", "34", "50", "101", "152"], help='which resnet ["18", "34", "50", "101", "152"]')
parser.add_argument('--train_img_shape', default=(1024, 512), nargs=2, metavar=("W", "H"),
                    help="W H")
parser.add_argument('--net', type=str, default="fcn", choices=['fcn', 'fcnvgg', 'psp', 'segnet'],
                    help="network structure")
parser.add_argument('--base_outdir', type=str, default='train_output',
                    help="base output dir")
parser.add_argument('--batch_size', type=int, default=1,
                    help="batch_size")
parser.add_argument('--augment', action="store_true",
                    help='whether you use data-augmentation or not')
parser.add_argument('--loss_weights_file', type=str, default=None,
                    help='Use this when you control the loss per class')
parser.add_argument("--input_ch", type=int, default=3,
                    choices=[1, 3, 4])
parser.add_argument("--resume", type=str, default=None, metavar="PTH",
                    help="model(pth) path")
parser.add_argument("--n_class", type=int, default=20, help="the number of classes")
args = parser.parse_args()

start_epoch = 0
if args.resume:
    if not os.path.exists(args.resume):
        raise OSError("%s does not exist!" % args.resume)

    indir, infn = os.path.split(args.resume)
    savename, net, res, epoch_with_pth = infn.split("-")
    start_epoch = int(epoch_with_pth.replace(".pth", ""))
    print ("savename is %s (%s was overwritten)" % (savename, args.savename))
    print ("start epoch is %s" % start_epoch)
    args.savename = savename
    args.net = net
    args.res = res

args.outdir = os.path.join(args.base_outdir, "%s_only_%sch" % (args.src_dataset, args.input_ch))
pth_dir = os.path.join(args.outdir, "pth")
tflog_dir = os.path.join(args.outdir, "tflog", args.savename)
mkdir_if_not_exist(pth_dir)
mkdir_if_not_exist(tflog_dir)

json_fn = os.path.join(args.outdir, "param_%s.json" % args.savename)
check_if_done(json_fn)

args.machine = os.uname()[1]
save_dic_to_json(args.__dict__, json_fn)
if args.net == "fcn":
    from models.fcn import ResBase, ResClassifier

    # G = torch.nn.DataParallel(ResBase(args.n_class, layer=args.res, input_ch=args.input_ch))
    G = ResBase(args.n_class, layer=args.res, input_ch=args.input_ch, no_replace=True)
    F1 = torch.nn.DataParallel(ResClassifier(args.n_class))
elif args.net == "fcnvgg":
    from models.vgg_fcn import FCN8sBase, FCN8sClassifier

    # model_g = torch.nn.DataParallel(ResBase(args.n_class, layer=args.res, input_ch=args.input_ch)) # TODO this outputs error
    # TODO implement input_ch
    G = FCN8sBase(args.n_class)
    F1 = torch.nn.DataParallel(FCN8sClassifier(args.n_class))
    F2 = torch.nn.DataParallel(FCN8sClassifier(args.n_class))
elif args.net == "psp":
    # TODO add "input_ch" argument
    from models.pspnet import PSPBase, PSPClassifier

    G = torch.nn.DataParallel(PSPBase())
    F1 = torch.nn.DataParallel(PSPClassifier(num_classes=args.n_class))

elif args.net == "segnet":
    # TODO add "input_ch" argument
    from models.segnet import SegNetBase, SegNetClassifier

    G = torch.nn.DataParallel(SegNetBase())
    F1 = torch.nn.DataParallel(SegNetClassifier(args.n_class))

else:
    raise Exception("Network Error!")

print(G)

G.load_state_dict(torch.load(args.g_path))
epoches = args.epochs
lr = args.lr  # 1e-3 was best

num_class = 20
init_lr = lr
weight_decay = 2e-5
momentum = 0.9
weight = torch.ones(num_class)
if args.loss_weights_file:
    import pandas as pd

    loss_df = pd.read_csv(args.loss_weights_file)
    loss_df.sort_values("class_id", inplace=True)
    weight *= torch.FloatTensor(loss_df.weight.values)
weight[num_class - 1] = 0  # Ignore background loss
print ("loss weight %s" % weight)

max_iters = 92 * epoches

train_img_shape = tuple([int(x) for x in args.train_img_shape])

img_transform_list = [
    Scale(train_img_shape, Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225])
]

if args.augment:
    aug_list = [
        RandomRotation(),
        # RandomVerticalFlip(), # non-realistic
        RandomHorizontalFlip(),
        RandomSizedCrop()
    ]
    img_transform_list = aug_list + img_transform_list

img_transform = Compose(img_transform_list)

label_transform = Compose([
    Scale(train_img_shape, Image.NEAREST),
    ToLabel(),
    ReLabel(255, num_class - 1),
])

src_dataset = get_dataset(dataset_name=args.src_dataset, split="train", img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args.input_ch)

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

criterion = CrossEntropyLoss2d(weight)

optimizer = torch.optim.SGD(F1.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

ploter = LinePlotter()
configure(tflog_dir, flush_secs=5)

G.cuda()
F1.cuda()
F1.train()
for epoch in range(epoches):
    epoch_loss = 0
    for ind, (images, labels) in tqdm(enumerate(train_loader)):

        imgs = Variable(images)
        lbls = Variable(labels)
        if torch.cuda.is_available():
            imgs, lbls = imgs.cuda(), lbls.cuda()

        # update generator and classifiers by source samples
        optimizer.zero_grad()
        preds = G(imgs)
        preds = F1(preds)
        loss = criterion(preds, lbls)
        loss.backward()
        c_loss = loss.data[0]
        epoch_loss += c_loss

        optimizer.step()

        if ind % 100 == 0:
            print("iter [%d] CLoss: %.4f" % (ind, c_loss))

    print("Epoch [%d] Loss: %.4f" % (epoch + 1, epoch_loss))
    ploter.plot("loss", "train", epoch + 1, epoch_loss)
    log_value('loss', epoch_loss, epoch)
    log_value('lr', lr, epoch)

    # lr = adjust_learning_rate(optimizer, lr, weight_decay, epoch, epoches)

    model_fn_g = os.path.join(pth_dir, "%s-%s-%s-g-%d.pth" % (args.savename, args.net, args.res, epoch + 1))
    model_fn_f1 = os.path.join(pth_dir, "%s-%s-%s-f1-%d.pth" % (args.savename, args.net, args.res, epoch + 1))

    torch.save(F.state_dict(), model_fn_f1)
    torch.save(G.state_dict(), model_fn_g)
