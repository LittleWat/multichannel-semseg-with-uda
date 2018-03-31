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
from models.model_util import get_optimizer, fix_batchnorm_when_training, \
    get_triple_multitask_models  # check_training
from transform import get_img_transform, \
    get_lbl_transform
from util import check_if_done, save_checkpoint, adjust_learning_rate, emphasize_str, get_class_weight_from_file, \
    set_debugger_org_frc
from util import mkdir_if_not_exist, save_dic_to_json

set_debugger_org_frc()

parser = get_src_only_training_parser()
parser.add_argument('--depth_shortcut', action="store_true", help='whether you use depth shortcut')
parser.add_argument('--semseg_shortcut', action="store_true", help='whether you use semseg shortcut')
parser.add_argument('--add_pred_seg_boundary_loss', action="store_true",
                    help='whether you use additional boundary loss')
parser.add_argument('--boundary_loss_converging_epoch', type=int, default=5,
                    help='epoch starting to include tgt boundary loss')
parser.add_argument('--scale_bd_loss', type=int, default=1,
                    help='epoch starting to include tgt boundary loss')

args = parser.parse_args()
args = add_additional_params_to_args(args)

weight = get_class_weight_from_file(n_class=args.n_class, weight_filename=args.loss_weights_file,
                                    add_bg_loss=args.add_bg_loss)

if torch.cuda.is_available():
    weight = weight.cuda()

criterion = CrossEntropyLoss2d(weight)

if args.resume:
    raise NotImplementedError("sorry")

else:
    model_enc, model_dec = get_triple_multitask_models(net_name=args.net, input_ch=args.input_ch,
                                                       n_class=args.n_class, is_data_parallel=args.is_data_parallel,
                                                       semseg_criterion=criterion,
                                                       depth_shortcut=args.depth_shortcut,
                                                       semseg_shortcut=args.semseg_shortcut,
                                                       add_pred_seg_boundary_loss=args.add_pred_seg_boundary_loss,
                                                       is_src_only=True)
    optimizer = get_optimizer(list(model_enc.parameters()) + list(model_dec.parameters()), opt=args.opt, lr=args.lr,
                              momentum=args.momentum,
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

use_crop = True if args.crop_size > 0 else False
joint_transform = get_joint_transform(crop_size=args.crop_size, rotate_angle=args.rotate_angle) if use_crop else None

img_transform = get_img_transform(img_shape=train_img_shape, normalize_way=args.normalize_way, use_crop=use_crop)

label_transform = get_lbl_transform(img_shape=train_img_shape, n_class=args.n_class, background_id=args.background_id,
                                    use_crop=use_crop)

src_dataset = get_dataset(dataset_name=args.src_dataset, split=args.split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=7)

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

weight = get_class_weight_from_file(n_class=args.n_class, weight_filename=args.loss_weights_file,
                                    add_bg_loss=args.add_bg_loss)

if torch.cuda.is_available():
    model_enc.cuda()
    model_dec.cuda()
    weight = weight.cuda()

# criterion = CrossEntropyLoss2d(weight)

configure(args.tflog_dir, flush_secs=5)

model_enc.train()
model_dec.train()
if args.fix_bn:
    print (emphasize_str("BN layers are NOT trained!"))
    fix_batchnorm_when_training(model_enc)
    fix_batchnorm_when_training(model_dec)

    # check_training(model_enc)

for epoch in range(start_epoch, args.epochs):
    epoch_loss = 0
    semseg_loss_per_epoch = 0
    depth_loss_per_epoch = 0
    boundary_loss_per_epoch = 0

    for ind, (images, labels) in tqdm(enumerate(train_loader)):

        imgs = Variable(images)
        lbls = Variable(labels)
        if torch.cuda.is_available():
            imgs, lbls = imgs.cuda(), lbls.cuda()

        rgbs = imgs[:, :3, :, :]
        depths = imgs[:, 3:-1, :, :]
        boundaries = imgs[:, -1:, :, :]

        # update generator and classifiers by source samples
        optimizer.zero_grad()
        fets = model_enc(rgbs)
        semseg_loss, depth_loss, boundary_loss = model_dec.get_loss(fets, lbls, depths,
                                                                    boundaries, separately_returning=True)

        semseg_loss_per_epoch += semseg_loss.data[0]
        depth_loss_per_epoch += depth_loss.data[0]
        boundary_loss_per_epoch += boundary_loss.data[0]

        loss = semseg_loss + depth_loss + boundary_loss
        loss.backward()

        epoch_loss += loss.data[0]
        semseg_loss_per_epoch += semseg_loss.data[0]
        depth_loss_per_epoch += depth_loss.data[0]
        boundary_loss_per_epoch += boundary_loss.data[0]

        optimizer.step()

        # if ind % 100 == 0:
        #     print("iter [%d] CLoss: %.4f" % (ind, c_loss))

        if ind > args.max_iter:
            break

    print("Epoch [%d] Loss: %.4f" % (epoch + 1, epoch_loss))
    log_value('loss', epoch_loss, epoch)
    log_value('lr', args.lr, epoch)
    log_value('semseg_loss', semseg_loss_per_epoch, epoch)
    log_value('depth_loss', depth_loss_per_epoch, epoch)
    log_value('boundary_loss', semseg_loss_per_epoch, epoch)

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
        'enc_state_dict': model_enc.state_dict(),
        'dec_state_dict': model_dec.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)
