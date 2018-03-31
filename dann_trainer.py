from __future__ import division

import argparse
import os

import torch
import tqdm
from PIL import Image
from tensorboard_logger import configure, log_value
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor

from argmyparse import get_da_dann_training_parser, add_additional_params_to_args
from datasets import ConcatDataset, get_dataset, check_src_tgt_ok, get_n_class
from loss import CrossEntropyLoss2d
from models.model_util import fix_batchnorm_when_training, get_optimizer
from transform import ReLabel, ToLabel, Scale, RandomSizedCrop, RandomHorizontalFlip, RandomRotation
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done, save_checkpoint, adjust_learning_rate, \
    emphasize_str
from visualize import LinePlotter

# parser = argparse.ArgumentParser(description='PyTorch Segmentation Adaptation')
# parser.add_argument('src_dataset', type=str, choices=["gta", "city", "test", "ir", "city16", "synthia", "2d3d"])
# parser.add_argument('tgt_dataset', type=str, choices=["gta", "city", "test", "ir", "city16", "synthia", "2d3d"])
# parser.add_argument('--src_split', type=str, default='train',
#                     help="which split('train' or 'trainval' or 'val' or something else) is used ")
# parser.add_argument('--tgt_split', type=str, default='train',
#                     help="which split('train' or 'trainval' or 'val' or something else) is used ")
# parser.add_argument('--savename', type=str, default="dann", help="save name(Do NOT use '-')")
# parser.add_argument('--epochs', type=int, default=40,
#                     help='number of epochs to train (default: 40)')
# parser.add_argument('--lr', type=float, default=1e-3,
#                     help='learning rate (default: 0.001)')
# parser.add_argument('--momentum', type=float, default=0.9,
#                     help='momentum sgd (default: 0.9)')
# parser.add_argument('--weight_decay', type=float, default=2e-5,
#                     help='weight_decay (default: 2e-5)')
# parser.add_argument('--num_k', type=int, default=4,
#                     help='how many steps to repeat the generator update')
# parser.add_argument('--res', type=str, default='50', metavar="ResnetLayerNum",
#                     choices=["18", "34", "50", "101", "152"], help='which resnet 18,50,101,152')
# parser.add_argument('--train_img_shape', default=(1024, 512), nargs=2, metavar=("W", "H"),
#                     help="W H")
# parser.add_argument('--net', type=str, default="fcn", help="network structure",
#                     choices=['fcn', 'fcnvgg', 'psp', 'segnet', "drn_c_26", "drn_c_42", "drn_c_58", "drn_d_22",
#                              "drn_d_38", "drn_d_54", "drn_d_105"])
# parser.add_argument('--opt', type=str, default="sgd", choices=['sgd', 'adam'],
#                     help="network optimizer")
# parser.add_argument('--base_outdir', type=str, default='train_output',
#                     help="base output dir")
# parser.add_argument('--batch_size', '-b', type=int, default=1,
#                     help="batch_size")
# parser.add_argument('--uses_one_classifier', action="store_true",
#                     help="separate f1, f2")
# parser.add_argument('--augment', action="store_true",
#                     help='whether you use data-augmentation or not')
# parser.add_argument('--loss_weights_file', type=str, default=None,
#                     help='Use this when you control the loss per class')
# parser.add_argument("--input_ch", type=int, default=3,
#                     choices=[1, 3, 4])
# parser.add_argument("--resume", type=str, default=None, metavar="PTH.TAR",
#                     help="model(pth) path")
# parser.add_argument("--add_bg_loss", action="store_true",
#                     help='whether you add background loss or not')
# parser.add_argument("--adjust_lr", action="store_true",
#                     help='whether you change lr')
# parser.add_argument("--max_iter", type=int, default=5000)
# parser.add_argument("--fix_bn", action="store_true",
#                     help='whether you fix the paramters of batch normalization layer')

parser = get_da_dann_training_parser()
args = parser.parse_args()
args = add_additional_params_to_args(args)


if args.src_dataset == "2d3d":
    args.train_img_shape = [1080, 1080]
    print ("args.train_img_shape was changed to %s" % args.train_img_shape)

check_src_tgt_ok(args.src_dataset, args.tgt_dataset)
args.n_class = get_n_class(args.src_dataset)

weight = torch.ones(args.n_class)
if args.loss_weights_file:
    import pandas as pd

    loss_df = pd.read_csv(args.loss_weights_file)
    loss_df.sort_values("class_id", inplace=True)
    weight *= torch.FloatTensor(loss_df.weight.values)

if not args.add_bg_loss:
    weight[args.n_class - 1] = 0  # Ignore background loss

# print ("loss weight %s" % weight)
from models.fcn import Discriminator

if args.net == "fcn":
    from models.fcn import ResBase, ResClassifier

    # model_g = torch.nn.DataParallel(ResBase(args.n_class, layer=args.res, input_ch=args.input_ch)) # TODO this outputs error 
    model_g = ResBase(args.n_class, layer=args.res, input_ch=args.input_ch)
    # model_f = torch.nn.DataParallel(ResClassifier(args.n_class))
    # model_f2 = torch.nn.DataParallel(ResClassifier(args.n_class))
    model_f = ResClassifier(args.n_class)
    model_d = Discriminator()

elif args.net == "fcnvgg":
    from models.vgg_fcn import FCN8sBase, FCN8sClassifier

    # model_g = torch.nn.DataParallel(ResBase(args.n_class, layer=args.res, input_ch=args.input_ch)) # TODO this outputs error
    # TODO implement input_ch
    model_g = FCN8sBase(args.n_class)
    model_f = torch.nn.DataParallel(FCN8sClassifier(args.n_class))
    model_d = torch.nn.DataParallel(Discriminator())
elif args.net == "psp":
    # TODO add "input_ch" argument
    from models.pspnet import PSPBase, PSPClassifier

    # model_g = torch.nn.DataParallel(PSPBase(layer=args.res, input_ch=args.input_ch))
    model_g = PSPBase(layer=args.res, input_ch=args.input_ch)
    model_f = torch.nn.DataParallel(PSPClassifier(num_classes=args.n_class))
    model_d = torch.nn.DataParallel(Discriminator())
elif args.net == "segnet":
    # TODO add "input_ch" argument
    from models.segnet import SegNetBase, SegNetClassifier

    model_g = torch.nn.DataParallel(SegNetBase())
    model_f = torch.nn.DataParallel(SegNetClassifier(args.n_class))
    model_d = torch.nn.DataParallel(Discriminator())
elif "drn" in args.net:
    from models.dilated_fcn import DRNSegBase, DRNSegPixelClassifier, DRNSegDomainClassifier

    model_g = DRNSegBase(model_name=args.net, n_class=args.n_class, input_ch=args.input_ch)
    model_f = DRNSegPixelClassifier(n_class=args.n_class)
    model_d = DRNSegDomainClassifier(n_class=args.n_class)


else:
    raise NotImplementedError("Only FCN, SegNet, PSPNet are supported!")

# if args.opt == 'sgd':
#     optimizer_g = torch.optim.SGD(model_g.parameters(), lr=args.lr, momentum=args.momentum,
#                                   weight_decay=args.weight_decay)
#     optimizer_d = torch.optim.SGD(model_d.parameters(), lr=args.lr, momentum=args.momentum,
#                                   weight_decay=args.weight_decay)
#     optimizer_f = torch.optim.SGD(list(model_f.parameters()), lr=args.lr,
#                                   momentum=args.momentum,
#                                   weight_decay=args.weight_decay)
# if args.opt == 'adam':
#     optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr, betas=[0.5, 0.999],
#                                    weight_decay=args.weight_decay)
#     optimizer_f = torch.optim.Adam(list(model_f.parameters()) + list(model_f2.parameters()), lr=args.lr,
#                                    betas=[0.5, 0.999],
#                                    weight_decay=args.weight_decay)

optimizer_g = get_optimizer(model_g.parameters(), args.opt, args.lr, args.momentum, args.weight_decay)
optimizer_d = get_optimizer(model_d.parameters(), args.opt, args.lr, args.momentum, args.weight_decay)
optimizer_f = get_optimizer(model_f.parameters(), args.opt, args.lr, args.momentum, args.weight_decay)

args.start_epoch = 0
if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    if not os.path.exists(args.resume):
        raise OSError("%s does not exist!" % args.resume)

    indir, infn = os.path.split(args.resume)
    savename = infn.split("-")[0]
    print ("savename is %s (%s was overwritten)" % (savename, args.savename))

    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']

    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_f.load_state_dict(checkpoint['f1_state_dict'])

    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_f.load_state_dict(checkpoint['optimizer_f'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))

mode = "%s-%s2%s-%s_%sch" % (args.src_dataset, args.src_split, args.tgt_dataset, args.tgt_split, args.input_ch)
args.outdir = os.path.join(args.base_outdir, mode)

pth_dir = os.path.join(args.outdir, "pth")
model_name = "%s-%s-res%s" % (args.savename, args.net, args.res)
tflog_dir = os.path.join(args.outdir, "tflog", model_name)
mkdir_if_not_exist(pth_dir)
mkdir_if_not_exist(tflog_dir)

json_fn = os.path.join(args.outdir, "param-%s-%s-%s.json" % (args.savename, args.net, args.res))
check_if_done(json_fn)

args.machine = os.uname()[1]
save_dic_to_json(args.__dict__, json_fn)

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
    ReLabel(255, args.n_class - 1),
])

src_dataset = get_dataset(dataset_name=args.src_dataset, split=args.src_split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args.input_ch)

tgt_dataset = get_dataset(dataset_name=args.tgt_dataset, split=args.tgt_split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args.input_ch)

train_loader = torch.utils.data.DataLoader(
    ConcatDataset(
        src_dataset,
        tgt_dataset
    ),
    batch_size=args.batch_size, shuffle=True,
    pin_memory=True)

if torch.cuda.is_available():
    model_g.cuda()
    model_f.cuda()
    model_d.cuda()
    weight = weight.cuda()

criterion = CrossEntropyLoss2d(weight)
criterion_d = nn.CrossEntropyLoss()

ploter = LinePlotter()
configure(tflog_dir, flush_secs=5)

model_g.train()
model_f.train()
model_d.train()
if args.fix_bn:
    print (emphasize_str("BN layers are NOT trained!"))
    fix_batchnorm_when_training(model_g)
    fix_batchnorm_when_training(model_f)
    fix_batchnorm_when_training(model_d)

src_domain_lbl = Variable(torch.ones(args.batch_size).long())
tgt_domain_lbl = Variable(torch.zeros(args.batch_size).long())

for epoch in range(args.start_epoch, args.epochs):
    d_loss_per_epoch = 0
    c_loss_per_epoch = 0

    for ind, (source, target) in tqdm.tqdm(enumerate(train_loader)):
        src_imgs, src_lbls = Variable(source[0]), Variable(source[1])
        tgt_imgs = Variable(target[0])

        if torch.cuda.is_available():
            src_imgs, src_lbls, tgt_imgs = src_imgs.cuda(), src_lbls.cuda(), tgt_imgs.cuda()
            src_domain_lbl, tgt_domain_lbl = src_domain_lbl.cuda(), tgt_domain_lbl.cuda()

        # update generator and classifiers by source samples
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        src_fet = model_g(src_imgs)
        tgt_fet = model_g(tgt_imgs)

        # for k, v in outputs.items():
        #     try:
        #         print ("%s: %s" % (k, v.size()))
        #     except AttributeError:
        #         print ("%s: %s" % (k, v))
        if "drn" in args.net:
            src_domain_pred = model_d(src_fet)
            tgt_domain_pred = model_d(tgt_fet)
        else:
            src_domain_pred = model_d(src_fet["fm4"])
            tgt_domain_pred = model_d(tgt_fet["fm4"])

        loss_d = - criterion_d(src_domain_pred, src_domain_lbl)
        loss_d -= criterion_d(tgt_domain_pred, tgt_domain_lbl)

        src_out = model_f(src_fet)
        loss = criterion(src_out, src_lbls)

        c_loss = loss.data[0]
        loss += loss_d
        loss.backward()
        c_loss_per_epoch += c_loss

        optimizer_g.step()
        optimizer_f.step()

        # update for classifiers
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        src_fet = model_g(src_imgs)
        tgt_fet = model_g(tgt_imgs)
        if "drn" in args.net:
            src_domain_pred = model_d(src_fet)
            tgt_domain_pred = model_d(tgt_fet)
        else:
            src_domain_pred = model_d(src_fet["fm4"])
            tgt_domain_pred = model_d(tgt_fet["fm4"])
        loss_d = criterion_d(src_domain_pred, src_domain_lbl)
        loss_d += criterion_d(tgt_domain_pred, tgt_domain_lbl)
        loss_d.backward()
        optimizer_d.step()
        optimizer_d.zero_grad()

        d_loss = 0
        d_loss += loss_d.data[0] / args.num_k
        d_loss_per_epoch += d_loss
        if ind % 100 == 0:
            print("iter [%d] DLoss: %.6f CLoss: %.4f" % (ind, d_loss, c_loss))

        if ind > args.max_iter:
            break

    print("Epoch [%d] DLoss: %.4f CLoss: %.4f" % (epoch, d_loss_per_epoch, c_loss_per_epoch))
    # ploter.plot("c_loss", "train", epoch + 1, c_loss_per_epoch)
    # ploter.plot("d_loss", "train", epoch + 1, d_loss_per_epoch)
    log_value('c_loss', c_loss_per_epoch, epoch)
    log_value('d_loss', d_loss_per_epoch, epoch)
    log_value('lr', args.lr, epoch)

    if args.adjust_lr:
        args.lr = adjust_learning_rate(optimizer_g, args.lr, args.weight_decay, epoch, args.epochs)
        args.lr = adjust_learning_rate(optimizer_f, args.lr, args.weight_decay, epoch, args.epochs)

    checkpoint_fn = os.path.join(pth_dir, "%s-%s-res%s-%s.pth.tar" % (args.savename, args.net, args.res, epoch + 1))
    save_dic = {
        'epoch': epoch + 1,
        'res': args.res,
        'net': args.net,
        'args': args,
        'g_state_dict': model_g.state_dict(),
        'f1_state_dict': model_f.state_dict(),
        'd_state_dict': model_d.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_f': optimizer_f.state_dict(),
        'optimizer_d': optimizer_d.state_dict(),
    }

    save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)
