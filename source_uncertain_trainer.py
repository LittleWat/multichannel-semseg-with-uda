from __future__ import division

import os

import torch
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from argmyparse import get_src_only_training_parser, add_additional_params_to_args
from datasets import get_dataset
from joint_transforms import get_joint_transform
from loss import CrossEntropyLoss2d
from models.model_util import get_optimizer, get_full_model, fix_batchnorm_when_training, \
    get_full_uncertain_model  # check_training
from transform import get_img_transform, \
    get_lbl_transform
from util import check_if_done, save_checkpoint, adjust_learning_rate, emphasize_str, get_class_weight_from_file, \
    set_debugger_org_frc
from util import mkdir_if_not_exist, save_dic_to_json

set_debugger_org_frc()

parser = get_src_only_training_parser()
parser.add_argument('--n_dropout', type=int, default=10,
                    help='how many steps to repeat the generator update')

args = parser.parse_args()
args = add_additional_params_to_args(args)

weight = get_class_weight_from_file(n_class=args.n_class, weight_filename=args.loss_weights_file,
                                    add_bg_loss=args.add_bg_loss)

if torch.cuda.is_available():
    weight = weight.cuda()

criterion = CrossEntropyLoss2d(weight)

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

    model = get_full_model(net=args.net, res=args.res, n_class=args.n_class, input_ch=args.input_ch)
    optimizer = get_optimizer(model.parameters(), opt=args.opt, lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}'".format(args.resume))

    json_fn = os.path.join(args.outdir, "param_%s_resume.json" % args.savename)
    check_if_done(json_fn)
    args.machine = os.uname()[1]
    save_dic_to_json(args.__dict__, json_fn)

    start_epoch = checkpoint['epoch']

else:
    model = get_full_uncertain_model(net=args.net, res=args.res, n_class=args.n_class, input_ch=args.input_ch,
                                     n_dropout=args.n_dropout, criterion=criterion, is_data_parallel=False)

    optimizer = get_optimizer(model.parameters(), opt=args.opt, lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)

    args.outdir = os.path.join(args.base_outdir,
                               "%s-%s_only_%sch_uncertain" % (args.src_dataset, args.split, args.input_ch))
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

use_crop = True if args.crop_size > 0 else False
joint_transform = get_joint_transform(crop_size=args.crop_size, rotate_angle=args.rotate_angle) if use_crop else None

img_transform = get_img_transform(img_shape=train_img_shape, normalize_way=args.normalize_way, use_crop=use_crop)

label_transform = get_lbl_transform(img_shape=train_img_shape, n_class=args.n_class, background_id=args.background_id,
                                    use_crop=use_crop)

src_dataset = get_dataset(dataset_name=args.src_dataset, split=args.split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args.input_ch,
                          joint_transform=joint_transform)

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

if torch.cuda.is_available():
    model.cuda()

configure(args.tflog_dir, flush_secs=5)

model.train()
if args.fix_bn:
    print (emphasize_str("BN layers are NOT trained!"))
    fix_batchnorm_when_training(model)

    # check_training(model)

for epoch in range(start_epoch, args.epochs):
    epoch_loss = 0
    epoch_base_loss = 0
    epoch_uncertain_loss = 0
    for ind, (images, labels) in tqdm(enumerate(train_loader)):

        imgs = Variable(images)
        lbls = Variable(labels)
        if torch.cuda.is_available():
            imgs, lbls = imgs.cuda(), lbls.cuda()

        # update generator and classifiers by source samples
        optimizer.zero_grad()

        base_loss, uncertain_loss = model.get_loss(imgs, lbls, separately_returning=True)
        loss = base_loss + uncertain_loss
        loss.backward()

        epoch_loss += loss.data[0]
        epoch_base_loss += base_loss.data[0]
        epoch_uncertain_loss += uncertain_loss.data[0]

        optimizer.step()

        if ind % 100 == 0:
            print("iter [%d] TotalLoss: %.4f (SegLoss: %.4f UncertainLoss: %.4f)"
                  % (ind, loss.data[0], base_loss.data[0], uncertain_loss.data[0]))

        if ind > args.max_iter:
            break

    print("Epoch [%d] Loss: %.4f" % (epoch + 1, epoch_loss))
    log_value('loss', epoch_loss, epoch)
    log_value('seg_loss', epoch_base_loss, epoch)
    log_value('uncertain_loss', epoch_uncertain_loss, epoch)
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
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)
