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
from loss import CrossEntropyLoss2d, get_prob_distance_criterion, ProbCrossEntropyLoss2d
from models.model_util import fix_batchnorm_when_training, get_models, get_optimizer, fix_dropout_when_training
from transform import get_img_transform, \
    get_lbl_transform
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done, save_checkpoint, emphasize_str, \
    get_class_weight_from_file, adjust_learning_rate

parser = get_da_mcd_training_parser()
parser.add_argument("--method_detail", type=str, default="MFNet-GateFusion",
                    choices=["MFNet-AddFusion", "MFNet-ConcatFusion", "MFNet-ConcatConvFusion", "MFNet-GateFusion",
                             "MFNet-ScoreAddFusion", "MFNet-ScoreConcatConvFusion", "MFNet-ScoreGateFusion"])
parser.add_argument("checkpoint", type=str, metavar="PTH.TAR",
                    default=None)
parser.add_argument('extra_checkpoint', type=str, metavar="PTH.TAR", default=None)

args = parser.parse_args()
args = add_additional_params_to_args(args)
assert args.input_ch in [4, 6]
detailed_method = args.method + "-" + args.method_detail
print ("method: %s" % detailed_method)

check_src_tgt_ok(args.src_dataset, args.tgt_dataset)

print("=> loading checkpoint '{}'".format(args.checkpoint))
if not os.path.exists(args.checkpoint):
    raise OSError("%s does not exist!" % args.checkpoint)

indir, infn = os.path.split(args.checkpoint)


def get_model_name_from_path(path):
    return path.split(os.path.sep)[-1].replace(".tar", "")


args.savename = get_model_name_from_path(args.checkpoint) + "AND" + get_model_name_from_path(args.extra_checkpoint)

print ("savename is %s " % (args.savename))

checkpoint = torch.load(args.checkpoint)
model_g_3ch, model_g_1ch, model_f1, model_f2 = get_models(net_name=args.net, res=args.res, input_ch=args.input_ch,
                                                          n_class=args.n_class, method=detailed_method,
                                                          is_data_parallel=args.is_data_parallel)
optimizer_g = get_optimizer(list(model_g_3ch.parameters()) + list(model_g_1ch.parameters()), lr=args.lr,
                            opt=args.opt,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
optimizer_f = get_optimizer(list(model_f1.parameters()) + list(model_f2.parameters()), lr=args.lr, opt=args.opt,
                            momentum=args.momentum, weight_decay=args.weight_decay)

model_g_3ch.load_state_dict(checkpoint['g_state_dict'])

print("=> loading checkpoint '{}'".format(args.extra_checkpoint))
model_g_1ch.load_state_dict(torch.load(args.extra_checkpoint)['g_state_dict'])

if args.uses_one_classifier:
    model_f2 = model_f1

print("=> loaded checkpoint '{}'".format(args.checkpoint))

if args.uses_one_classifier:
    print ("f1 and f2 are same!")
    model_f2 = model_f1

mode = "%s-%s2%s-%s_%sch_Finetune_MFNet" % (
    args.src_dataset, args.src_split, args.tgt_dataset, args.tgt_split, args.input_ch)
if args.net in ["fcn", "psp"]:
    model_name = "%s-%s-%s-res%s" % (detailed_method, args.savename, args.net, args.res)
else:
    model_name = "%s-%s-%s" % (detailed_method, args.savename, args.net)

outdir = os.path.join(args.base_outdir, mode)

# Create Model Dir
pth_dir = os.path.join(outdir, "pth")
mkdir_if_not_exist(pth_dir)

# Create Model Dir and  Set TF-Logger
tflog_dir = os.path.join(outdir, "tflog", model_name)
mkdir_if_not_exist(tflog_dir)
configure(tflog_dir, flush_secs=5)

# Save param dic

json_fn = os.path.join(outdir, "param-%s-finetune_MFNet.json" % model_name)
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
    model_g_3ch.cuda()
    model_g_1ch.cuda()
    model_f1.cuda()
    model_f2.cuda()
    weight = weight.cuda()

criterion = CrossEntropyLoss2d(weight) if "Gate" not in args.method_detail else ProbCrossEntropyLoss2d(weight)
criterion_d = get_prob_distance_criterion(args.d_loss)

model_g_3ch.train()
model_g_1ch.train()
model_f1.train()
model_f2.train()

if args.no_dropout:
    print ("NO DROPOUT")
    fix_dropout_when_training(model_g_3ch)
    fix_dropout_when_training(model_g_1ch)
    fix_dropout_when_training(model_f1)
    fix_dropout_when_training(model_f2)

if args.fix_bn:
    print (emphasize_str("BN layers are NOT trained!"))
    fix_batchnorm_when_training(model_g_3ch)
    fix_batchnorm_when_training(model_g_1ch)
    fix_batchnorm_when_training(model_f1)
    fix_batchnorm_when_training(model_f2)

for epoch in range(args.epochs):
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
        g_rgb_outdic = model_g_3ch(src_imgs[:, :3, :, :])
        g_ir_outdic = model_g_1ch(src_imgs[:, 3:, :, :])
        outputs1 = model_f1(g_rgb_outdic, g_ir_outdic)
        outputs2 = model_f2(g_rgb_outdic, g_ir_outdic)

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

        g_rgb_outdic = model_g_3ch(src_imgs[:, :3, :, :])
        g_ir_outdic = model_g_1ch(src_imgs[:, 3:, :, :])
        outputs1 = model_f1(g_rgb_outdic, g_ir_outdic)
        outputs2 = model_f2(g_rgb_outdic, g_ir_outdic)

        loss = 0
        loss += criterion(outputs1, src_lbls)
        loss += criterion(outputs2, src_lbls)

        # accumulate loss
        g_rgb_outdic = model_g_3ch(tgt_imgs[:, :3, :, :])
        g_ir_outdic = model_g_1ch(tgt_imgs[:, 3:, :, :])
        outputs1 = model_f1(g_rgb_outdic, g_ir_outdic)
        outputs2 = model_f2(g_rgb_outdic, g_ir_outdic)

        loss -= criterion_d(outputs1, outputs2)
        loss.backward()
        optimizer_f.step()
        optimizer_f.zero_grad()
        d_loss = 0.0

        # update generator by discrepancy
        for i in xrange(args.num_k):
            optimizer_g.zero_grad()
            loss = 0
            g_rgb_outdic = model_g_3ch(tgt_imgs[:, :3, :, :])
            g_ir_outdic = model_g_1ch(tgt_imgs[:, 3:, :, :])
            outputs1 = model_f1(g_rgb_outdic, g_ir_outdic)
            outputs2 = model_f2(g_rgb_outdic, g_ir_outdic)
            loss += criterion_d(outputs1, outputs2)
            loss.backward()
            optimizer_g.step()

        d_loss += loss.data[0] / args.num_k
        d_loss_per_epoch += d_loss

        if ind % 100 == 0:
            print("iter [%d] DLoss: %.4f CLoss: %.4f" % (ind, d_loss, c_loss))

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
        'g_3ch_state_dict': model_g_3ch.state_dict(),
        'g_1ch_state_dict': model_g_1ch.state_dict(),
        'f1_state_dict': model_f1.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_f': optimizer_f.state_dict(),
    }
    if not args.uses_one_classifier:
        save_dic['f2_state_dict'] = model_f2.state_dict()

    save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)
