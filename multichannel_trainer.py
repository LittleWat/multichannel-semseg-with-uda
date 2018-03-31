from __future__ import division

import os

import torch
from PIL import Image
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from argmyparse import get_src_only_training_parser, add_additional_params_to_args, add_img_shape_to_args
from datasets import get_dataset
from loss import CrossEntropyLoss2d
from models.model_util import get_optimizer, fix_batchnorm_when_training, \
    get_multichannel_model  # check_training
from transform import ReLabel, ToLabel, Scale, RandomSizedCrop, RandomHorizontalFlip, RandomRotation
from util import check_if_done, save_checkpoint, adjust_learning_rate, emphasize_str, get_class_weight_from_file
from util import mkdir_if_not_exist, save_dic_to_json

parser = get_src_only_training_parser()
parser.add_argument('--method', type=str, default="MCD", help="Method Name")
parser.add_argument("--method_detail", type=str, default="MFNet-GateFusion",
                    choices=["MFNet-AddFusion", "MFNet-ConcatConvFusion", "MFNet-GateFusion"])
parser.add_argument("--inch_list", nargs="+", default=[3, 1], type=int)

args = parser.parse_args()
args = add_additional_params_to_args(args)
args = add_img_shape_to_args(args)
args.savename = "multichannel"
detailed_method = args.method + "-" + args.method_detail

if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    if not os.path.exists(args.resume):
        raise OSError("%s does not exist!" % args.resume)

    indir, infn = os.path.split(args.resume)

    old_savename = args.savename
    args.savename = infn.split("-")[0]
    print ("savename is %s (original savename %s was overwritten)" % (args.savename, old_savename))

    checkpoint = torch.load(args.resume)
    args = checkpoint['args']  # Load args!
    [model_g1, model_g2], model_f1 = get_multichannel_model(net_name=args.net, input_ch_list=args.inch_list,
                                                            n_class=args.n_class, method=detailed_method, res=args.res,
                                                            is_data_parallel=args.is_data_parallel)

    optimizer = get_optimizer(list(model_g1.parameters()) + list(model_g2.parameters()) + list(model_f1.parameters()),
                              opt=args.opt, lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    model_g1.load_state_dict(checkpoint['g1_state_dict'])
    model_g2.load_state_dict(checkpoint['g2_state_dict'])
    model_f1.load_state_dict(checkpoint['f1_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}'".format(args.resume))

    json_fn = os.path.join(args.outdir, "param_%s_resume.json" % args.savename)
    check_if_done(json_fn)
    args.machine = os.uname()[1]
    save_dic_to_json(args.__dict__, json_fn)

    start_epoch = checkpoint['epoch']

else:
    [model_g1, model_g2], model_f1 = get_multichannel_model(net_name=args.net, input_ch_list=args.inch_list,
                                                            n_class=args.n_class, method=detailed_method, res=args.res,
                                                            is_data_parallel=args.is_data_parallel)

    optimizer = get_optimizer(list(model_g1.parameters()) + list(model_g2.parameters()) + list(model_f1.parameters()),
                              opt=args.opt, lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)

    args.outdir = os.path.join(args.base_outdir, "%s-%s_only_%sch" % (args.src_dataset, args.split, args.input_ch))
    args.pth_dir = os.path.join(args.outdir, "pth")

    if args.net in ["fcn", "psp"]:
        model_name = "%s-%s-res%s" % (args.savename, args.net, args.res)
    else:
        model_name = "%s-%s" % (args.savename, args.net)

    args.tflog_dir = os.path.join(args.outdir, "tflog", model_name)
    mkdir_if_not_exist(args.pth_dir)
    mkdir_if_not_exist(args.tflog_dir)

    json_fn = os.path.join(args.outdir, "param-%s.json" % model_name)
    check_if_done(json_fn)
    args.machine = os.uname()[1]
    save_dic_to_json(args.__dict__, json_fn)

    start_epoch = 0

train_img_shape = tuple([int(x) for x in args.train_img_shape])

img_transform_list = [
    Scale(train_img_shape, Image.BILINEAR),
    ToTensor(),
    # Normalize([.485, .456, .406], [.229, .224, .225])
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

src_dataset = get_dataset(dataset_name=args.src_dataset, split=args.split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args.input_ch)

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

weight = get_class_weight_from_file(n_class=args.n_class, weight_filename=args.loss_weights_file,
                                    add_bg_loss=args.add_bg_loss)

if torch.cuda.is_available():
    model_g1.cuda()
    model_g2.cuda()
    model_f1.cuda()
    weight = weight.cuda()

criterion = CrossEntropyLoss2d(weight)

configure(args.tflog_dir, flush_secs=5)

model_g1.train()
model_g2.train()
model_f1.train()
if args.fix_bn:
    print (emphasize_str("BN layers are NOT trained!"))
    fix_batchnorm_when_training(model_g1)
    fix_batchnorm_when_training(model_g2)
    fix_batchnorm_when_training(model_f1)

    # check_training(model)

for epoch in range(start_epoch, args.epochs):
    epoch_loss = 0
    for ind, (images, labels) in tqdm(enumerate(train_loader)):

        imgs = Variable(images)
        lbls = Variable(labels)
        if torch.cuda.is_available():
            imgs, lbls = imgs.cuda(), lbls.cuda()

        # update generator and classifiers by source samples
        optimizer.zero_grad()
        fet1 = model_g1(imgs[:, :args.inch_list[0], :, :])
        fet2 = model_g2(imgs[:, args.inch_list[0]:args.inch_list[0] + args.inch_list[1], :, :])

        preds = model_f1(fet1, fet2)
        if args.net == "psp":
            preds = preds[0]

        loss = criterion(preds, lbls)
        loss.backward()
        c_loss = loss.data[0]
        epoch_loss += c_loss

        optimizer.step()

        if ind % 100 == 0:
            print("iter [%d] CLoss: %.4f" % (ind, c_loss))

        if ind > args.max_iter:
            break

    print("Epoch [%d] Loss: %.4f" % (epoch + 1, epoch_loss))
    log_value('loss', epoch_loss, epoch)
    log_value('lr', args.lr, epoch)

    if args.adjust_lr:
        args.lr = adjust_learning_rate(optimizer, args.lr, args.weight_decay, epoch, args.epochs)

    if args.net == "fcn" or args.net == "psp":
        checkpoint_fn = os.path.join(args.pth_dir, "%s-%s-res%s-%s.pth.tar" % (
            args.savename, args.net, args.res, epoch + 1))
    else:
        checkpoint_fn = os.path.join(args.pth_dir, "%s-%s-%s.pth.tar" % (
            args.savename, args.net, epoch + 1))

    args.start_epoch = epoch + 1
    save_dic = {
        'args': args,
        'epoch': epoch + 1,
        'g1_state_dict': model_g1.state_dict(),
        'g2_state_dict': model_g2.state_dict(),
        'f1_state_dict': model_f1.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)
