from __future__ import division

import os

import torch
import tqdm
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from torch.utils import data

from argmyparse import add_additional_params_to_args, get_da_mcd_training_parser
from datasets import ConcatDataset, get_dataset, check_src_tgt_ok
from joint_transforms import get_joint_transform
from loss import CrossEntropyLoss2d, get_prob_distance_criterion
from models.model_util import fix_batchnorm_when_training, get_models, get_optimizer, fix_dropout_when_training
from transform import get_img_transform, \
    get_lbl_transform
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done, save_checkpoint, adjust_learning_rate, \
    emphasize_str, get_class_weight_from_file

parser = get_da_mcd_training_parser()
args = parser.parse_args()
args = add_additional_params_to_args(args)

check_src_tgt_ok(args.src_dataset, args.tgt_dataset)

resume_flg = True if args.resume else False
start_epoch = 0
if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    if not os.path.exists(args.resume):
        raise OSError("%s does not exist!" % args.resume)

    indir, infn = os.path.split(args.resume)

    old_savename = args.savename
    args.savename = infn.split("-")[0]
    print ("savename is %s (original savename %s was overwritten)" % (args.savename, old_savename))

    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    # ---------- Replace Args!!! ----------- #
    args = checkpoint['args']
    # -------------------------------------- #
    model_g, model_f1, model_f2 = get_models(net_name=args.net, res=args.res, input_ch=args.input_ch,
                                             n_class=args.n_class, method=args.method,
                                             is_data_parallel=args.is_data_parallel)
    optimizer_g = get_optimizer(model_g.parameters(), lr=args.lr, momentum=args.momentum, opt=args.opt,
                                weight_decay=args.weight_decay)
    optimizer_f = get_optimizer(list(model_f1.parameters()) + list(model_f2.parameters()), lr=args.lr, opt=args.opt,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_f1.load_state_dict(checkpoint['f1_state_dict'])
    if not args.uses_one_classifier:
        model_f2.load_state_dict(checkpoint['f2_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_f.load_state_dict(checkpoint['optimizer_f'])
    print("=> loaded checkpoint '{}'".format(args.resume))

else:
    model_g, model_f1, model_f2 = get_models(net_name=args.net, res=args.res, input_ch=args.input_ch,
                                             n_class=args.n_class,
                                             method=args.method, is_data_parallel=args.is_data_parallel)
    optimizer_g = get_optimizer(model_g.parameters(), lr=args.lr, momentum=args.momentum, opt=args.opt,
                                weight_decay=args.weight_decay)
    optimizer_f = get_optimizer(list(model_f1.parameters()) + list(model_f2.parameters()), opt=args.opt,
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
if args.uses_one_classifier:
    print ("f1 and f2 are same!")
    model_f2 = model_f1

mode = "%s-%s2%s-%s_%sch" % (args.src_dataset, args.src_split, args.tgt_dataset, args.tgt_split, args.input_ch)
if args.net in ["fcn", "psp"]:
    model_name = "%s-%s-%s-res%s" % (args.method, args.savename, args.net, args.res)
else:
    model_name = "%s-%s-%s" % (args.method, args.savename, args.net)

outdir = os.path.join(args.base_outdir, mode)

# Create Model Dir
pth_dir = os.path.join(outdir, "pth")
mkdir_if_not_exist(pth_dir)

# Create Model Dir and  Set TF-Logger
tflog_dir = os.path.join(outdir, "tflog", model_name)
mkdir_if_not_exist(tflog_dir)
configure(tflog_dir, flush_secs=5)

# Save param dic
if resume_flg:
    json_fn = os.path.join(args.outdir, "param-%s_resume.json" % model_name)
else:
    json_fn = os.path.join(outdir, "param-%s.json" % model_name)

check_if_done(json_fn)
save_dic_to_json(args.__dict__, json_fn)

train_img_shape = tuple([int(x) for x in args.train_img_shape])

use_crop = True if args.crop_size > 0 else False
joint_transform = get_joint_transform(crop_size=args.crop_size, rotate_angle=args.rotate_angle) if use_crop else None

img_transform = get_img_transform(img_shape=train_img_shape, normalize_way=args.normalize_way, use_crop=use_crop)

label_transform = get_lbl_transform(img_shape=train_img_shape, n_class=args.n_class, background_id=args.background_id,
                                    use_crop=use_crop)

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

weight = get_class_weight_from_file(n_class=args.n_class, weight_filename=args.loss_weights_file,
                                    add_bg_loss=args.add_bg_loss)

if torch.cuda.is_available():
    model_g.cuda()
    model_f1.cuda()
    model_f2.cuda()
    weight = weight.cuda()

criterion = CrossEntropyLoss2d(weight)
criterion_d = get_prob_distance_criterion(args.d_loss)

model_g.train()
model_f1.train()
model_f2.train()

if args.no_dropout:
    print ("NO DROPOUT")
    fix_dropout_when_training(model_g)
    fix_dropout_when_training(model_f1)
    fix_dropout_when_training(model_f2)

if args.fix_bn:
    print (emphasize_str("BN layers are NOT trained!"))
    fix_batchnorm_when_training(model_g)
    fix_batchnorm_when_training(model_f1)
    fix_batchnorm_when_training(model_f2)

for epoch in range(start_epoch, args.epochs):
    d_loss_per_epoch = 0
    c_loss_per_epoch = 0

    for ind, (source, target) in tqdm.tqdm(enumerate(train_loader)):
        src_imgs, src_lbls = Variable(source[0]), Variable(source[1])
        tgt_imgs = Variable(target[0])

        if torch.cuda.is_available():
            src_imgs, src_lbls, tgt_imgs = src_imgs.cuda(), src_lbls.cuda(), tgt_imgs.cuda()

        # update generator and classifiers by source samples
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        loss = 0
        loss_weight = [1.0, 1.0]
        outputs = model_g(src_imgs)

        # for k, v in outputs.items():
        #     try:
        #         print ("%s: %s" % (k, v.size()))
        #     except AttributeError:
        #         print ("%s: %s" % (k, v))

        outputs1 = model_f1(outputs)
        outputs2 = model_f2(outputs)

        loss += criterion(outputs1, src_lbls)
        loss += criterion(outputs2, src_lbls)
        loss.backward()
        c_loss = loss.data[0]
        c_loss_per_epoch += c_loss

        optimizer_g.step()
        optimizer_f.step()
        # update for classifiers
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        outputs = model_g(src_imgs)
        outputs1 = model_f1(outputs)
        outputs2 = model_f2(outputs)
        loss = 0
        loss += criterion(outputs1, src_lbls)
        loss += criterion(outputs2, src_lbls)
        outputs = model_g(tgt_imgs)
        outputs1 = model_f1(outputs)
        outputs2 = model_f2(outputs)
        loss -= criterion_d(outputs1, outputs2)
        loss.backward()
        optimizer_f.step()

        d_loss = 0.0
        # update generator by discrepancy
        for i in xrange(args.num_k):
            optimizer_g.zero_grad()
            loss = 0
            outputs = model_g(tgt_imgs)
            outputs1 = model_f1(outputs)
            outputs2 = model_f2(outputs)
            loss += criterion_d(outputs1, outputs2) * args.num_multiply_d_loss
            loss.backward()
            optimizer_g.step()

        d_loss += loss.data[0] / args.num_k
        d_loss_per_epoch += d_loss
        if ind % 100 == 0:
            print("iter [%d] DLoss: %.6f CLoss: %.4f" % (ind, d_loss, c_loss))

        if ind > args.max_iter:
            break

    print("Epoch [%d] DLoss: %.4f CLoss: %.4f" % (epoch, d_loss_per_epoch, c_loss_per_epoch))

    log_value('c_loss', c_loss_per_epoch, epoch)
    log_value('d_loss', d_loss_per_epoch, epoch)
    log_value('lr', args.lr, epoch)

    if args.adjust_lr:
        args.lr = adjust_learning_rate(optimizer_g, args.lr, args.weight_decay, epoch, args.epochs)
        args.lr = adjust_learning_rate(optimizer_f, args.lr, args.weight_decay, epoch, args.epochs)

    checkpoint_fn = os.path.join(pth_dir, "%s-%s.pth.tar" % (model_name, epoch + 1))
    args.start_epoch = epoch + 1
    save_dic = {
        'epoch': epoch + 1,
        'args': args,
        'g_state_dict': model_g.state_dict(),
        'f1_state_dict': model_f1.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_f': optimizer_f.state_dict(),
    }
    if not args.uses_one_classifier:
        save_dic['f2_state_dict'] = model_f2.state_dict()

    save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)
