#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import shutil
import sys
import time
from os.path import exists, join, split

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.nn import Parameter
from torchvision import transforms

import drn
from loss import Diff2d, bce2d
from models.fusion import get_fusion_model, ConcatFusion


def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


CITYSCAPE_PALLETE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, model_name, n_class, input_ch=3, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()

        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000, input_ch=input_ch)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, n_class,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4,
                                    output_padding=0, groups=n_class,
                                    bias=False)
            # fill_up_weights(up)
            # up.weight.requires_grad = False  # WHY?

            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        # return self.softmax(y), x
        # return self.softmax(y)
        return y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


class UncertainDRNSeg(nn.Module):
    def __init__(self, model_name, n_class, input_ch=3, pretrained_model=None,
                 pretrained=True, use_torch_up=False, criterion=None,
                 n_dropout_exp=10, uncertain_weight=0.1):
        super(UncertainDRNSeg, self).__init__()

        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000, input_ch=input_ch)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, n_class,
                             kernel_size=1, bias=True)

        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        self.dropout = nn.Dropout2d(0.2)
        self.criterion = criterion
        self.n_dropout_exp = n_dropout_exp
        self.uncertain_weight = uncertain_weight

        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
            self.up_std = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4,
                                    output_padding=0, groups=n_class,
                                    bias=False)
            # fill_up_weights(up)
            # up.weight.requires_grad = False  # WHY?
            up_std = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4, output_padding=0, groups=n_class,
                                        bias=False)

            self.up = up
            self.up_std = up_std

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        x = self.dropout(x)
        y = self.up(x)
        y_std = F.relu(self.up_std(x))  # Std must be more than 0

        # y_std = y_std.sum(dim=1) # TODO is this necessary?
        return y, y_std

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

    def get_loss(self, x, gt, separately_returning=False):
        batchsize = gt.size(0)
        n_pixel = gt.numel() / batchsize
        self.train()

        base_loss = 0
        uncertain_loss = 0
        for i in range(self.n_dropout_exp):
            pred_mean, pred_std = self.forward(x)
            base_loss += self.criterion(pred_mean, gt)

            rand = np.random.rand()  # TODO: Or torch.randn(pred_std.size())
            y_hat = pred_mean + pred_std * rand

            y_hat_c = None
            for i in range(batchsize):
                one_gt = gt[i]
                gt_size = one_gt.size()
                one_gt1d = one_gt.view(1, -1)

                one_y_hat1d = y_hat[i].view(y_hat[i].size(0), -1)

                one_y_hat1d_c = torch.gather(one_y_hat1d, 0, one_gt1d)  # size [1, n_pixel]

                one_y_hat_c = one_y_hat1d_c.view([1] + list(gt_size))
                if y_hat_c is None:
                    y_hat_c = one_y_hat_c
                else:
                    y_hat_c = torch.cat([y_hat_c, one_y_hat_c])

            logsumexp_y_hat = torch.log(torch.exp(y_hat).sum(1))
            uncertain_loss += torch.exp(y_hat_c - logsumexp_y_hat)

        base_loss /= self.n_dropout_exp
        uncertain_loss = torch.log(uncertain_loss / self.n_dropout_exp).sum(1).sum(1) / n_pixel
        total_loss = base_loss + uncertain_loss * self.uncertain_weight
        # print ("baseloss")
        # print (base_loss)
        # print ("uncertain")
        # print (uncertain_loss)
        # print ("total_loss")
        # print (total_loss)
        if separately_returning:
            return base_loss, uncertain_loss * self.uncertain_weight

        return total_loss


class DRNSegBase(nn.Module):
    def __init__(self, model_name, n_class, pretrained=True, input_ch=3, ver="ver1"):
        super(DRNSegBase, self).__init__()

        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000, input_ch=input_ch)
        self.base = nn.Sequential(*list(model.children())[:-2])
        self.ver = ver

        if ver == "ver1":
            self.seg = nn.Conv2d(model.out_dim, n_class,
                                 kernel_size=1, bias=True)
            m = self.seg
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()

        elif ver == "ver2":
            print ("ver2 will be used")
            pass

    def forward(self, x):
        x = self.base(x)
        if self.ver == "ver2":
            return x

        x = self.seg(x)
        return x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


class FuseDRNSegBase(nn.Module):
    def __init__(self, model_name, n_class, pretrained=True, input_ch=3, ver="ver1"):
        super(FuseDRNSegBase, self).__init__()
        assert input_ch in [4, 6]
        self.ver = ver

        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000, input_ch=3)

        sub_model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000, input_ch=input_ch - 3)
        # self.base = nn.Sequential(*list(model.children())[:-2])
        assert model.arch == "D"

        self.main_layer0 = model.layer0
        self.main_layer1 = model.layer1
        self.main_layer2 = model.layer2
        self.main_layer3 = model.layer3
        self.main_layer4 = model.layer4
        self.main_layer5 = model.layer5
        self.main_layer6 = model.layer6
        self.main_layer7 = model.layer7
        self.main_layer8 = model.layer8

        self.sub_layer0 = sub_model.layer0
        self.sub_layer1 = sub_model.layer1
        self.sub_layer2 = sub_model.layer2
        self.sub_layer3 = sub_model.layer3
        self.sub_layer4 = sub_model.layer4
        self.sub_layer5 = sub_model.layer5
        self.sub_layer6 = sub_model.layer6
        self.sub_layer7 = sub_model.layer7
        self.sub_layer8 = sub_model.layer8

        self.seg = nn.Conv2d(model.out_dim, n_class,
                             kernel_size=1, bias=True)
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()

    def forward(self, x):
        rgb_inputs = x[:, :3, :, :]
        depth_inputs = x[:, 3:, :, :]

        x_d0 = self.main_layer0(depth_inputs)
        x_d1 = self.main_layer1(x_d0)
        x_d2 = self.main_layer2(x_d1)
        x_d3 = self.main_layer3(x_d2)
        x_d4 = self.main_layer4(x_d3)
        x_d5 = self.main_layer5(x_d4)
        x_d6 = self.main_layer6(x_d5)
        x_d7 = self.main_layer7(x_d6)
        x_d8 = self.main_layer8(x_d7)

        x = self.main_layer0(rgb_inputs)
        x = torch.add(x, x_d0)

        x = self.main_layer1(x)
        x = torch.add(x, x_d1)

        x = self.main_layer2(x)
        x = torch.add(x, x_d2)

        x = self.main_layer3(x)
        x = torch.add(x, x_d3)
        x = self.main_layer4(x)
        x = torch.add(x, x_d4)
        x = self.main_layer5(x)
        x = torch.add(x, x_d5)
        x = self.main_layer6(x)
        x = torch.add(x, x_d6)
        x = self.main_layer7(x)
        x = torch.add(x, x_d7)
        x = self.main_layer8(x)
        x = torch.add(x, x_d8)

        x = self.seg(x)
        return x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


class DRNSegPixelClassifier(nn.Module):
    def __init__(self, n_class, use_torch_up=False, dropout=False, ver="ver1"):
        super(DRNSegPixelClassifier, self).__init__()
        self.dropout = dropout
        self.ver = ver

        if ver == "ver2":
            self.seg = nn.Conv2d(512, n_class, kernel_size=1, bias=True)  # TODO in_ch (500) is hard coding
            m = self.seg
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()

        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:

            up = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4,
                                    output_padding=0, groups=n_class,
                                    bias=False)
            self.up = up

    def forward(self, x):
        if self.ver == "ver2":
            x = self.seg(x)
        x = self.up(x)
        return x


class DRNSegBase_2(nn.Module):
    def __init__(self, model_name, n_class, pretrained_model=None, pretrained=True, input_ch=3):
        super(DRNSegBase_2, self).__init__()

        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000, input_ch=input_ch)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, n_class,
                             kernel_size=1, bias=True)

    def forward(self, x):
        x = self.base(x)
        # x = self.seg(x)
        return x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param


class DRNSegPixelClassifier_2(nn.Module):
    def __init__(self, n_class, model_name, use_torch_up=False, input_ch=3):
        super(DRNSegPixelClassifier_2, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=True, num_classes=1000, input_ch=input_ch)
        pmodel = nn.DataParallel(model)
        self.seg = nn.Conv2d(model.out_dim, n_class,
                             kernel_size=1, bias=True)
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(n_class, n_class, 1, stride=1, padding=0,
                                    output_padding=0, groups=n_class,
                                    bias=False)
            self.dropout = nn.Dropout(.1)
            self.dropout2 = nn.Dropout(.1)
            up2 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4,
                                     output_padding=0, groups=n_class,
                                     bias=False)

            self.up2 = up2
            # fill_up_weights(up)
            # up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.seg(x)
        x = self.dropout(x)
        x = self.up(x)
        x = self.dropout2(x)
        x = self.up2(x)
        return x


class FusionDRNSegPixelClassifier(nn.Module):
    def __init__(self, fusion_type, n_class, use_torch_up=False, ver="ver1"):
        super(FusionDRNSegPixelClassifier, self).__init__()
        if ver == "ver1":
            self.fusion = get_fusion_model(fusion_type, n_class)

        elif ver == "ver2":
            self.fusion = get_fusion_model(fusion_type, 512)  # Hard Coding

        self.ver = ver

        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            if type(self.fusion) == ConcatFusion:
                up = nn.ConvTranspose2d(2 * n_class, n_class, 16, stride=8, padding=4,
                                        output_padding=0, groups=n_class,
                                        bias=False)

            else:
                up = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4,
                                        output_padding=0, groups=n_class,
                                        bias=False)
            # fill_up_weights(up)
            # up.weight.requires_grad = False
            self.up = up

        if ver == "ver2":
            self.seg = nn.Conv2d(512, n_class, kernel_size=1, bias=True)  # TODO in_ch (500) is hard coding
            m = self.seg
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()

    def forward(self, x1, x2):
        h = self.fusion(x1, x2)
        if self.ver == "ver2":
            h = self.seg(h)
        h = self.up(h)
        return h


class ScoreFusionDRNSegPixelClassifier(nn.Module):
    def __init__(self, fusion_type, n_class):
        super(ScoreFusionDRNSegPixelClassifier, self).__init__()

        self.fusion = get_fusion_model(fusion_type, n_class)

        self.up1 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4,
                                      output_padding=0, groups=n_class,
                                      bias=False)

        self.up2 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4,
                                      output_padding=0, groups=n_class,
                                      bias=False)

    def forward(self, x1, x2):
        h1 = self.up1(x1)
        h2 = self.up2(x2)
        h = self.fusion(h1, h2)
        return h


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


class DownConv(nn.Module):
    """
    copied from https://github.com/jaxony/unet-pytorch/blob/master/model.py
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.pooling:
            x = self.pool(x)
        return x


class DRNSegDomainClassifier(nn.Module):
    def __init__(self, n_class):
        super(DRNSegDomainClassifier, self).__init__()

        self.conv1 = DownConv(n_class, 10)
        self.conv2 = DownConv(10, 1)
        self.fc1 = nn.Linear(1089, 10)  # TODO This is for 4ch
        self.bn1 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        h = self.conv1(x)  # (1L, 10L, 32L, 64L)
        h = self.conv2(h)  # (1L, 1L, 16L, 32L)
        h = h.view(h.size(0), -1)  # [1, 1, 16, 32] -> [1, 512]
        # print (h.size())
        out = F.relu(self.bn1(self.fc1(h)))
        out = self.fc2(out)
        return out


class MultiTaskEncoder(nn.Module):
    def __init__(self, model_name, pretrained=True, input_ch=3):
        super(MultiTaskEncoder, self).__init__()

        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000, input_ch=input_ch)
        self.base = nn.Sequential(*list(model.children())[:-2])
        # self.up = nn.Upsample(scale_factor=8, mode="bilinear")

    def forward(self, x):
        x = self.base(x)
        # x = self.up(x)
        return x


class MultiTaskEncoderReturningMultipleFeaturemaps(nn.Module):
    """
    ('h0', (1L, 16L, 480L, 640L))
    ('h1', (1L, 16L, 480L, 640L))
    ('h2', (1L, 32L, 240L, 320L))
    ('h3', (1L, 64L, 120L, 160L))
    ('h4', (1L, 128L, 60L, 80L))
    ('h5', (1L, 256L, 60L, 80L))
    ('h6', (1L, 512L, 60L, 80L))
    ('h7', (1L, 512L, 60L, 80L))
    ('h8', (1L, 512L, 60L, 80L))
    """

    def __init__(self, model_name, pretrained=True, input_ch=3):
        super(MultiTaskEncoderReturningMultipleFeaturemaps, self).__init__()

        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000, input_ch=input_ch)
        # self.base = nn.Sequential(*list(model.children())[:-2])

        if "drn_d" in model_name:
            self.main_layer0 = model.layer0

        else:
            self.main_layer0 = nn.Sequential(model.conv1,
                                             model.bn1,
                                             model.relu)
        self.main_layer1 = model.layer1
        self.main_layer2 = model.layer2
        self.main_layer3 = model.layer3
        self.main_layer4 = model.layer4
        self.main_layer5 = model.layer5
        self.main_layer6 = model.layer6
        self.main_layer7 = model.layer7
        self.main_layer8 = model.layer8

        self.up = nn.Upsample(scale_factor=8, mode="bilinear")

    def forward(self, x):
        h0 = self.main_layer0(x)
        h1 = self.main_layer1(h0)
        h2 = self.main_layer2(h1)
        h3 = self.main_layer3(h2)
        h4 = self.main_layer4(h3)
        h5 = self.main_layer5(h4)
        h6 = self.main_layer6(h5)
        h7 = self.main_layer7(h6)
        h8 = self.main_layer8(h7)
        out_dic = {
            "h0": h0,
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "h4": h4,
            "h5": h5,
            "h6": h6,
            "h7": h7,
            "h8": h8,
        }

        return out_dic


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ThreeLayerDecoder(nn.Module):
    def __init__(self, output_ch, input_ch=512):  # TODO input_ch should be 2048?
        super(ThreeLayerDecoder, self).__init__()
        self.cbr1 = CBR(input_ch, 512, kernel_size=3, padding=1)
        self.cbr2 = CBR(512, 512, kernel_size=1)
        self.conv3 = nn.Conv2d(512, output_ch, kernel_size=1)

    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.conv3(x)
        return x


class MCDMultiTaskDecoder(nn.Module):
    def __init__(self, n_class, depth_ch, semseg_criterion=None, discrepancy_criterion=None):  # Diff2d()
        super(MCDMultiTaskDecoder, self).__init__()
        self.s_semsegcls = Parameter(torch.Tensor(1))  # s is log var, semsegcls means semantic segmentation classifier
        self.s_deprgr = Parameter(torch.Tensor(1))  # depreg means depth regressor

        self.s_semsegcls.data.fill_(1)
        self.s_deprgr.data.fill_(1)

        self.semsegcls_dec1 = ThreeLayerDecoder(n_class)
        self.semsegcls_dec2 = ThreeLayerDecoder(n_class)
        self.deprgr_dec = ThreeLayerDecoder(depth_ch)
        self.semseg_criterion = semseg_criterion
        self.discrepancy_criterion = discrepancy_criterion

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')

    def semseg_forward(self, x):
        x1 = self.semsegcls_dec1(x)
        x2 = self.semsegcls_dec2(x)

        x1 = self.upsample(x1)  # NEW!
        x2 = self.upsample(x2)  # NEW!
        return x1, x2

    def depth_forward(self, x):
        x = self.deprgr_dec(x)
        x = self.upsample(x)  # NEW!
        return x

    def forward(self, x):
        pred_semseg1, pred_semseg2 = self.semseg_forward(x)
        pred_dep = self.depth_forward(x)

        return pred_semseg1, pred_semseg2, pred_dep

    def get_cls_descrepancy(self, x):
        pred_semseg1, pred_semseg2 = self.semseg_forward(x)
        return self.discrepancy_criterion(pred_semseg1, pred_semseg2)

    def get_semseg_loss(self, x, gt_semseg, separately_returning=False):
        pred_semseg1, pred_semseg2 = self.semseg_forward(x)
        loss1 = self.semseg_criterion(pred_semseg1, gt_semseg)
        loss2 = self.semseg_criterion(pred_semseg2, gt_semseg)

        if separately_returning:
            return loss1, loss2
        return loss1 + loss2

    def get_depth_loss(self, x, gt_dep):
        pred_dep = self.depth_forward(x)
        return F.mse_loss(pred_dep, gt_dep)

    def get_loss(self, x, gt_semseg, gt_dep, separately_returning=False):
        # pred_semseg1, pred_semseg2, pred_dep = self.forward(x)
        #
        # semseg_loss1 = torch.exp(-self.s_semsegcls) * self.semseg_criterion(pred_semseg1, gt_semseg) + self.s_semsegcls
        #
        # semseg_loss2 = torch.exp(-self.s_semsegcls) * self.semseg_criterion(pred_semseg2, gt_semseg) + self.s_semsegcls
        #
        # semseg_loss = (semseg_loss1 + semseg_loss2) / 2
        #
        # depreg_loss = torch.exp(-self.s_deprgr) * F.mse_loss(pred_dep, gt_dep) + self.s_deprgr
        org_semseg_loss1, org_semseg_loss2 = self.get_semseg_loss(x, gt_semseg, separately_returning=True)
        semseg_loss1 = torch.exp(-self.s_semsegcls) * org_semseg_loss1 + self.s_semsegcls
        semseg_loss2 = torch.exp(-self.s_semsegcls) * org_semseg_loss2 + self.s_semsegcls
        semseg_loss = (semseg_loss1 + semseg_loss2) / 2

        depreg_loss = torch.exp(-self.s_deprgr) * self.get_depth_loss(x, gt_dep) + self.s_deprgr

        if separately_returning:
            return semseg_loss, depreg_loss

        return semseg_loss + depreg_loss

    def get_task_weights(self):
        std_semseg = np.sqrt(np.exp(2 * self.s_semsegcls.data.cpu().numpy()))
        std_depth = np.sqrt(np.exp(2 * self.s_deprgr.data.cpu().numpy()))
        return std_semseg, std_depth


### TODO This has mistake (change Variable to numpy...)
def get_boundary_loss(pred, gt, pred_type="semseg", gt_type="semseg"):
    assert pred_type in ["semseg", "boundary"]
    assert gt_type in ["semseg", "boundary"]

    # def toTorch(np_variable):
    #     np_variable = np_variable.astype(np.float32)
    #     res = torch.from_numpy(np_variable)
    #     res = torch.autograd.Variable(res)
    #     return res
    #
    # extra_loss = 0
    # for i in six.moves.range(gt.size(0)):
    #     if gt_type == "semseg":
    #         gt_np = gt[i, ...].data.cpu().numpy()
    #         gt_boundary = find_boundaries(gt_np)
    #         gt_boundary = toTorch(gt_boundary).cuda()
    #
    #     else:
    #         gt_boundary = torch.autograd.Variable(gt[i, 0, ...].clone().data)
    #
    #     pred_np = pred_semseg[i, :].data.max(0)[1].cpu().numpy()
    #     pred_boundary = find_boundaries(pred_np)
    #     pred_boundary = toTorch(pred_boundary).cuda()
    #
    #     extra_loss += bce2d(pred_boundary, gt_boundary)

    # TODO This is not exactly correct
    def get_boundary(var):
        var = var.float()
        dilation = F.max_pool2d(var, kernel_size=3, stride=1, padding=1)
        erosion = -F.max_pool2d(-var, kernel_size=3, stride=1, padding=1)
        return dilation != erosion

    if gt_type == "semseg":
        gt_boundary = get_boundary(gt)
    else:
        gt_boundary = torch.autograd.Variable(gt.clone().data)

    if pred_type == "semseg":
        pred_boundary = get_boundary(pred)
    else:
        pred_boundary = pred
    extra_loss = bce2d(pred_boundary.float(), gt_boundary.float())

    return extra_loss


class MCDTripleMultiTaskDecoder(nn.Module):
    """
    Triple means "Semantic Segmentataion", "Depth Estimation", "Boundary Detection"

    https://github.com/EliasVansteenkiste/edge_detection_framework/blob/master/configs/hed_resnet34_pretrained.py
    """

    def __init__(self, n_class, depth_ch, semseg_criterion=None, discrepancy_criterion=None,
                 semseg_shortcut=False, depth_shortcut=False, add_pred_seg_boundary_loss=False,
                 use_seg2bd_conv=False):  # Diff2d()
        super(MCDTripleMultiTaskDecoder, self).__init__()
        self.s_semsegcls = Parameter(torch.Tensor(1))  # s is log var, semsegcls means semantic segmentation classifier
        self.s_deprgr = Parameter(torch.Tensor(1))  # depreg means depth regressor
        self.s_boundary = Parameter(torch.Tensor(1))

        self.s_semsegcls.data.fill_(1)
        self.s_deprgr.data.fill_(1)
        self.s_boundary.data.fill_(1)

        self.semsegcls_dec1 = ThreeLayerDecoder(n_class)
        self.semsegcls_dec2 = ThreeLayerDecoder(n_class)

        self.deprgr_dec = ThreeLayerDecoder(depth_ch)
        self.nmlrgr_dec = ThreeLayerDecoder(depth_ch)
        self.semseg_criterion = semseg_criterion
        self.discrepancy_criterion = discrepancy_criterion

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.conv1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)

        self.semseg_shortcut = semseg_shortcut
        self.depth_shortcut = depth_shortcut
        self.add_pred_seg_boundary_loss = add_pred_seg_boundary_loss

        if self.add_pred_seg_boundary_loss:
            self.s_pred_seg_boundary = Parameter(torch.Tensor(1))
            self.s_pred_seg_boundary.data.fill_(1)

        # if self.semseg_shortcut:
        #     self.seg_conv1_1 = nn.Conv2d(32, n_class, kernel_size=1, stride=1, padding=0)
        #     self.seg_conv2_1 = nn.Conv2d(64, n_class, kernel_size=1, stride=1, padding=0)
        #     self.seg_conv3_1 = nn.Conv2d(512, n_class, kernel_size=1, stride=1, padding=0)
        #
        #     self.seg_conv1_2 = nn.Conv2d(32, n_class, kernel_size=1, stride=1, padding=0)
        #     self.seg_conv2_2 = nn.Conv2d(64, n_class, kernel_size=1, stride=1, padding=0)
        #     self.seg_conv3_2 = nn.Conv2d(512, n_class, kernel_size=1, stride=1, padding=0)
        #
        # if self.depth_shortcut:
        #     self.dep_conv1 = nn.Conv2d(32, depth_ch, kernel_size=1, stride=1, padding=0)
        #     self.dep_conv2 = nn.Conv2d(64, depth_ch, kernel_size=1, stride=1, padding=0)
        #     self.dep_conv3 = nn.Conv2d(512, depth_ch, kernel_size=1, stride=1, padding=0)

        self.use_seg2bd_conv = use_seg2bd_conv

        if self.semseg_shortcut:
            self.seg_conv1_1 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0)
            self.seg_conv2_1 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
            self.seg_conv3_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

            self.seg_conv1_2 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0)
            self.seg_conv2_2 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
            self.seg_conv3_2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        if self.depth_shortcut:
            self.dep_conv1 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0)
            self.dep_conv2 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
            self.dep_conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        if self.use_seg2bd_conv:
            self.seg2bd_conv = nn.Conv2d(n_class, 1, kernel_size=5, padding=2)

    def semseg_forward(self, x_dic):
        if self.semseg_shortcut:
            h1 = self.seg_conv1_1(x_dic["h2"])
            h1 = self.upsample1(h1)
            h2 = self.seg_conv2_1(x_dic["h3"])
            h2 = self.upsample2(h2)
            h3 = self.seg_conv3_1(x_dic["h8"])
            h3 = self.upsample3(h3)
            x1 = h1 + h2 + h3

            h1 = self.seg_conv1_2(x_dic["h2"])
            h1 = self.upsample1(h1)
            h2 = self.seg_conv2_2(x_dic["h3"])
            h2 = self.upsample2(h2)
            h3 = self.seg_conv3_2(x_dic["h8"])
            h3 = self.upsample3(h3)
            x2 = h1 + h2 + h3

            # return x1, x2
            return self.semsegcls_dec1.forward(x1), self.semsegcls_dec2.forward(x2)

        # x = self.upsample3(x_dic["h8"])
        # return self.semsegcls_dec1.forward(x), self.semsegcls_dec2.forward(x)

        x1 = self.semsegcls_dec1.forward(x_dic["h8"])
        x2 = self.semsegcls_dec2.forward(x_dic["h8"])

        return self.upsample3(x1), self.upsample3(x2)

    def depth_forward(self, x_dic):
        if self.depth_shortcut:
            h1 = self.dep_conv1(x_dic["h2"])
            h1 = self.upsample1(h1)
            h2 = self.dep_conv2(x_dic["h3"])
            h2 = self.upsample2(h2)
            h3 = self.dep_conv3(x_dic["h8"])
            h3 = self.upsample3(h3)

            # return h1 + h2 + h3
            return self.deprgr_dec.forward(h1 + h2 + h3)

        # x = self.upsample3(x_dic["h8"])
        # return self.deprgr_dec.forward(x)

        x = self.deprgr_dec.forward(x_dic["h8"])
        return self.upsample3(x)

    def boundary_forward(self, x_dic):
        h1 = self.conv1(x_dic["h2"])
        h1 = self.upsample1(h1)
        h2 = self.conv2(x_dic["h3"])
        h2 = self.upsample2(h2)
        h3 = self.conv3(x_dic["h8"])
        h3 = self.upsample3(h3)

        boundary_pred = (F.sigmoid(h1) + F.sigmoid(h2) + F.sigmoid(h3)) / 3

        return boundary_pred

    def forward(self, x_dic):
        pred_semseg1, pred_semseg2 = self.semseg_forward(x_dic)
        pred_dep = self.depth_forward(x_dic)
        pred_boundary = self.boundary_forward(x_dic)

        return pred_semseg1, pred_semseg2, pred_dep, pred_boundary

    def get_cls_descrepancy(self, x_dic):
        pred_semseg1, pred_semseg2 = self.semseg_forward(x_dic)
        return self.discrepancy_criterion(pred_semseg1, pred_semseg2)

    def get_semseg_loss(self, x_dic, gt_semseg, separately_returning=False):
        pred_semseg1, pred_semseg2 = self.semseg_forward(x_dic)
        loss1 = self.semseg_criterion(pred_semseg1, gt_semseg)
        loss2 = self.semseg_criterion(pred_semseg2, gt_semseg)

        if self.add_pred_seg_boundary_loss:
            extra_loss1 = get_boundary_loss(pred_semseg1.max(1)[1], gt_semseg)
            extra_loss2 = get_boundary_loss(pred_semseg2.max(1)[1], gt_semseg)
            # print ("ExtraLoss1 %.4f" % extra_loss1.data[0])
            # print ("ExtraLoss2 %.4f" % extra_loss2.data[0])
            # print ("-"*100)

            loss1 += extra_loss1.cuda()
            loss2 += extra_loss2.cuda()

        if separately_returning:
            return loss1, loss2

        return loss1 + loss2

    def get_depth_loss(self, x_dic, gt_dep):
        pred_dep = self.depth_forward(x_dic)
        return F.mse_loss(pred_dep, gt_dep)

    def get_boundary_loss_by_extra_conv(self, x_dic, gt_bdry=None, separately_returning=False):
        assert self.use_seg2bd_conv

        pred_semseg1, pred_semseg2 = self.semseg_forward(x_dic)

        pred_bdry1 = F.sigmoid(self.seg2bd_conv(pred_semseg1))
        pred_bdry2 = F.sigmoid(self.seg2bd_conv(pred_semseg2))

        if gt_bdry is None:
            psuedo_boundary = self.boundary_forward(x_dic)
            psuedo_boundary = torch.autograd.Variable(
                psuedo_boundary.clone().data)  # do not compute gradients w.r.t target
            loss1 = bce2d(pred_bdry1, psuedo_boundary)
            loss2 = bce2d(pred_bdry2, psuedo_boundary)
        else:
            loss1 = bce2d(pred_bdry1, gt_bdry)
            loss2 = bce2d(pred_bdry2, gt_bdry)

        if separately_returning:
            return loss1, loss2

        return loss1 + loss2

    def get_psuedo_boundary_loss(self, x_dic, separately_returning=False):
        # type: (dic, bool) -> torch.autograd.Variable
        assert self.add_pred_seg_boundary_loss

        psuedo_boundary = self.boundary_forward(x_dic)
        pred_semseg1, pred_semseg2 = self.semseg_forward(x_dic)

        loss1 = get_boundary_loss(pred_semseg=pred_semseg1.max(1)[1], gt=psuedo_boundary, gt_type="boundary")
        loss2 = get_boundary_loss(pred_semseg=pred_semseg2.max(1)[1], gt=psuedo_boundary, gt_type="boundary")

        # print ("Loss1 %.4f" % loss1.data[0])
        # print ("Loss2 %.4f" % loss2.data[0])
        # print ("-" * 100)

        if separately_returning:
            return loss1, loss2

        return loss1 + loss2

    def get_boundary_loss(self, x_dic, gt_boundary):
        pred_boundary = self.boundary_forward(x_dic)
        return bce2d(pred_boundary, gt_boundary)

    def get_loss(self, x, gt_semseg, gt_dep, gt_boundary, separately_returning=False):
        org_semseg_loss1, org_semseg_loss2 = self.get_semseg_loss(x, gt_semseg, separately_returning=True)
        semseg_loss1 = torch.exp(-self.s_semsegcls) * org_semseg_loss1 + self.s_semsegcls
        semseg_loss2 = torch.exp(-self.s_semsegcls) * org_semseg_loss2 + self.s_semsegcls
        semseg_loss = (semseg_loss1 + semseg_loss2) / 2

        depreg_loss = torch.exp(-self.s_deprgr) * self.get_depth_loss(x, gt_dep) + self.s_deprgr

        boundary_loss = torch.exp(-self.s_boundary) * self.get_boundary_loss(x, gt_boundary) + self.s_boundary

        if separately_returning:
            return semseg_loss, depreg_loss, boundary_loss

        return semseg_loss + depreg_loss + boundary_loss

    def get_task_weights(self):
        std_semseg = np.sqrt(np.exp(2 * self.s_semsegcls.data.cpu().numpy()))
        std_depth = np.sqrt(np.exp(2 * self.s_deprgr.data.cpu().numpy()))
        return std_semseg, std_depth


class MCDSegBDMultiTaskDecoder(nn.Module):
    """
    Triple means "Semantic Segmentataion", "Depth Estimation", "Boundary Detection"

    https://github.com/EliasVansteenkiste/edge_detection_framework/blob/master/configs/hed_resnet34_pretrained.py
    """

    def __init__(self, n_class, depth_ch, semseg_criterion=None, discrepancy_criterion=None,
                 semseg_shortcut=False, depth_shortcut=False, add_pred_seg_boundary_loss=False,
                 use_seg2bd_conv=False):  # Diff2d()
        super(MCDSegBDMultiTaskDecoder, self).__init__()
        self.s_semsegcls = Parameter(torch.Tensor(1))  # s is log var, semsegcls means semantic segmentation classifier
        self.s_boundary = Parameter(torch.Tensor(1))

        self.s_semsegcls.data.fill_(1)
        self.s_boundary.data.fill_(1)

        self.semsegcls_dec1 = ThreeLayerDecoder(n_class)
        self.semsegcls_dec2 = ThreeLayerDecoder(n_class)

        self.semseg_criterion = semseg_criterion
        self.discrepancy_criterion = discrepancy_criterion

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.conv1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)

        self.semseg_shortcut = semseg_shortcut
        self.depth_shortcut = depth_shortcut
        self.add_pred_seg_boundary_loss = add_pred_seg_boundary_loss

        if self.add_pred_seg_boundary_loss:
            self.s_pred_seg_boundary = Parameter(torch.Tensor(1))
            self.s_pred_seg_boundary.data.fill_(1)

        # if self.semseg_shortcut:
        #     self.seg_conv1_1 = nn.Conv2d(32, n_class, kernel_size=1, stride=1, padding=0)
        #     self.seg_conv2_1 = nn.Conv2d(64, n_class, kernel_size=1, stride=1, padding=0)
        #     self.seg_conv3_1 = nn.Conv2d(512, n_class, kernel_size=1, stride=1, padding=0)
        #
        #     self.seg_conv1_2 = nn.Conv2d(32, n_class, kernel_size=1, stride=1, padding=0)
        #     self.seg_conv2_2 = nn.Conv2d(64, n_class, kernel_size=1, stride=1, padding=0)
        #     self.seg_conv3_2 = nn.Conv2d(512, n_class, kernel_size=1, stride=1, padding=0)

        self.use_seg2bd_conv = use_seg2bd_conv

        if self.semseg_shortcut:
            self.seg_conv1_1 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0)
            self.seg_conv2_1 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
            self.seg_conv3_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

            self.seg_conv1_2 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0)
            self.seg_conv2_2 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
            self.seg_conv3_2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        if self.use_seg2bd_conv:
            self.seg2bd_conv = nn.Conv2d(n_class, 1, kernel_size=5, padding=2)

    def semseg_forward(self, x_dic):
        if self.semseg_shortcut:
            h1 = self.seg_conv1_1(x_dic["h2"])
            h1 = self.upsample1(h1)
            h2 = self.seg_conv2_1(x_dic["h3"])
            h2 = self.upsample2(h2)
            h3 = self.seg_conv3_1(x_dic["h8"])
            h3 = self.upsample3(h3)
            x1 = h1 + h2 + h3

            h1 = self.seg_conv1_2(x_dic["h2"])
            h1 = self.upsample1(h1)
            h2 = self.seg_conv2_2(x_dic["h3"])
            h2 = self.upsample2(h2)
            h3 = self.seg_conv3_2(x_dic["h8"])
            h3 = self.upsample3(h3)
            x2 = h1 + h2 + h3

            # return x1, x2
            return self.semsegcls_dec1.forward(x1), self.semsegcls_dec2.forward(x2)

        # x = self.upsample3(x_dic["h8"])
        # return self.semsegcls_dec1.forward(x), self.semsegcls_dec2.forward(x)

        x1 = self.semsegcls_dec1.forward(x_dic["h8"])
        x2 = self.semsegcls_dec2.forward(x_dic["h8"])

        return self.upsample3(x1), self.upsample3(x2)

    def boundary_forward(self, x_dic):
        h1 = self.conv1(x_dic["h2"])
        h1 = self.upsample1(h1)
        h2 = self.conv2(x_dic["h3"])
        h2 = self.upsample2(h2)
        h3 = self.conv3(x_dic["h8"])
        h3 = self.upsample3(h3)

        boundary_pred = (F.sigmoid(h1) + F.sigmoid(h2) + F.sigmoid(h3)) / 3

        return boundary_pred

    def forward(self, x_dic):
        pred_semseg1, pred_semseg2 = self.semseg_forward(x_dic)
        pred_boundary = self.boundary_forward(x_dic)

        return pred_semseg1, pred_semseg2, pred_boundary

    def get_cls_descrepancy(self, x_dic):
        pred_semseg1, pred_semseg2 = self.semseg_forward(x_dic)
        return self.discrepancy_criterion(pred_semseg1, pred_semseg2)

    def get_semseg_loss(self, x_dic, gt_semseg, separately_returning=False):
        pred_semseg1, pred_semseg2 = self.semseg_forward(x_dic)
        loss1 = self.semseg_criterion(pred_semseg1, gt_semseg)
        loss2 = self.semseg_criterion(pred_semseg2, gt_semseg)

        extra_loss1 = get_boundary_loss(pred_semseg1.max(1)[1], gt_semseg)
        extra_loss2 = get_boundary_loss(pred_semseg2.max(1)[1], gt_semseg)

        # print ("ExtraLoss1 %.4f" % extra_loss1.data[0])
        # print ("ExtraLoss2 %.4f" % extra_loss2.data[0])
        # print ("-"*100)

        loss1 += extra_loss1
        loss2 += extra_loss2

        if separately_returning:
            return loss1, loss2

        return loss1 + loss2

    def get_boundary_loss_by_extra_conv(self, x_dic, gt_bdry=None, separately_returning=False):
        assert self.use_seg2bd_conv

        pred_semseg1, pred_semseg2 = self.semseg_forward(x_dic)

        pred_bdry1 = F.sigmoid(self.seg2bd_conv(pred_semseg1))
        pred_bdry2 = F.sigmoid(self.seg2bd_conv(pred_semseg2))

        if gt_bdry is None:
            psuedo_boundary = self.boundary_forward(x_dic)
            psuedo_boundary = torch.autograd.Variable(
                psuedo_boundary.clone().data)  # do not compute gradients w.r.t target
            loss1 = bce2d(pred_bdry1, psuedo_boundary)
            loss2 = bce2d(pred_bdry2, psuedo_boundary)
        else:
            loss1 = bce2d(pred_bdry1, gt_bdry)
            loss2 = bce2d(pred_bdry2, gt_bdry)

        if separately_returning:
            return loss1, loss2

        return loss1 + loss2

    def get_psuedo_boundary_loss(self, x_dic, separately_returning=False):
        # type: (dic, bool) -> torch.autograd.Variable
        assert self.add_pred_seg_boundary_loss

        psuedo_boundary = self.boundary_forward(x_dic)
        pred_semseg1, pred_semseg2 = self.semseg_forward(x_dic)

        loss1 = get_boundary_loss(pred=pred_semseg1.max(1)[1], gt=psuedo_boundary, gt_type="boundary")
        loss2 = get_boundary_loss(pred=pred_semseg2.max(1)[1], gt=psuedo_boundary, gt_type="boundary")

        # print ("Loss1 %.4f" % loss1.data[0])
        # print ("Loss2 %.4f" % loss2.data[0])
        # print ("-" * 100)

        if separately_returning:
            return loss1, loss2

        return loss1 + loss2

    def get_boundary_loss(self, x_dic, gt_semseg):
        pred_boundary = self.boundary_forward(x_dic)
        return get_boundary_loss(pred=pred_boundary, gt=gt_semseg, pred_type="boundary")

    def get_loss(self, x, gt_semseg, separately_returning=False):
        org_semseg_loss1, org_semseg_loss2 = self.get_semseg_loss(x, gt_semseg, separately_returning=True)
        semseg_loss1 = torch.exp(-self.s_semsegcls) * org_semseg_loss1 + self.s_semsegcls
        semseg_loss2 = torch.exp(-self.s_semsegcls) * org_semseg_loss2 + self.s_semsegcls
        semseg_loss = (semseg_loss1 + semseg_loss2) / 2

        boundary_loss = torch.exp(-self.s_boundary) * self.get_boundary_loss(x, gt_semseg) + self.s_boundary

        if separately_returning:
            return semseg_loss, boundary_loss

        return semseg_loss + boundary_loss

    def get_task_weights(self):
        std_semseg = np.sqrt(np.exp(2 * self.s_semsegcls.data.cpu().numpy()))
        std_depth = np.sqrt(np.exp(2 * self.s_deprgr.data.cpu().numpy()))
        return std_semseg, std_depth


class MultiTaskDecoder(nn.Module):
    def __init__(self, n_class, depth_ch, semseg_criterion, discrepancy_criterion=Diff2d()):
        super(MultiTaskDecoder, self).__init__()
        self.s_semsegcls = Parameter(torch.Tensor(1))  # s is log var, semsegcls means semantic segmentation classifier
        self.s_deprgr = Parameter(torch.Tensor(1))  # depreg means depth regressor
        self.semsegcls_dec = ThreeLayerDecoder(n_class)
        self.deprgr_dec = ThreeLayerDecoder(depth_ch)
        self.semseg_criterion = semseg_criterion
        self.discrepancy_criterion = discrepancy_criterion

    def forward(self, x):
        pred_semseg = self.semsegcls_dec(x)
        pred_dep = self.deprgr_dec(x)

        return pred_semseg, pred_dep

    def get_loss(self, x, gt_semseg, gt_dep, separately_returning=False):
        pred_semseg, pred_dep = self.forward(x)

        semseg_loss = torch.exp(-self.s_semsegcls) * self.semseg_criterion(pred_semseg, gt_semseg) + self.s_semsegcls
        depreg_loss = torch.exp(-self.s_deprgr) * F.mse_loss(pred_dep, gt_dep) + self.s_deprgr

        if separately_returning:
            return semseg_loss, depreg_loss

        return semseg_loss + depreg_loss

    def get_task_weights(self):
        std_semseg = np.sqrt(np.exp(2 * self.s_semsegcls))
        std_depth = np.sqrt(np.exp(2 * self.s_deprgr))
        return std_semseg, std_depth


class TripleMultiTaskDecoder(nn.Module):
    """
    Triple means "Semantic Segmentataion", "Depth Estimation", "Boundary Detection"

    https://github.com/EliasVansteenkiste/edge_detection_framework/blob/master/configs/hed_resnet34_pretrained.py
    """

    def __init__(self, n_class, depth_ch=3, semseg_criterion=None,
                 semseg_shortcut=False, depth_shortcut=False, add_pred_seg_boundary_loss=False,
                 conv_seg2bd=False):  # Diff2d()
        super(TripleMultiTaskDecoder, self).__init__()
        self.s_semsegcls = Parameter(torch.Tensor(1))  # s is log var, semsegcls means semantic segmentation classifier
        self.s_deprgr = Parameter(torch.Tensor(1))  # depreg means depth regressor
        self.s_boundary = Parameter(torch.Tensor(1))

        self.s_semsegcls.data.fill_(1)
        self.s_deprgr.data.fill_(1)
        self.s_boundary.data.fill_(1)

        self.semsegcls_dec = ThreeLayerDecoder(n_class)

        self.deprgr_dec = ThreeLayerDecoder(depth_ch)
        self.nmlrgr_dec = ThreeLayerDecoder(depth_ch)
        self.semseg_criterion = semseg_criterion

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.conv1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)

        self.semseg_shortcut = semseg_shortcut
        self.depth_shortcut = depth_shortcut
        self.add_pred_seg_boundary_loss = add_pred_seg_boundary_loss

        if self.add_pred_seg_boundary_loss:
            self.s_pred_seg_boundary = Parameter(torch.Tensor(1))
            self.s_pred_seg_boundary.data.fill_(1)

        if self.semseg_shortcut:
            self.seg_conv1_1 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0)
            self.seg_conv2_1 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
            self.seg_conv3_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

            self.seg_conv1_2 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0)
            self.seg_conv2_2 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
            self.seg_conv3_2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        if self.depth_shortcut:
            self.dep_conv1 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0)
            self.dep_conv2 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
            self.dep_conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

    def semseg_forward(self, x_dic):
        if self.semseg_shortcut:
            h1 = self.seg_conv1_1(x_dic["h2"])
            h1 = self.upsample1(h1)
            h2 = self.seg_conv2_1(x_dic["h3"])
            h2 = self.upsample2(h2)
            h3 = self.seg_conv3_1(x_dic["h8"])
            h3 = self.upsample3(h3)
            x1 = h1 + h2 + h3
            # return x1, x2
            return self.semsegcls_dec.forward(x1)

        # x = self.upsample3(x_dic["h8"])
        # return self.semsegcls_dec.forward(x), self.semsegcls_dec2.forward(x)

        x1 = self.semsegcls_dec.forward(x_dic["h8"])

        return self.upsample3(x1)

    def depth_forward(self, x_dic):
        if self.depth_shortcut:
            h1 = self.dep_conv1(x_dic["h2"])
            h1 = self.upsample1(h1)
            h2 = self.dep_conv2(x_dic["h3"])
            h2 = self.upsample2(h2)
            h3 = self.dep_conv3(x_dic["h8"])
            h3 = self.upsample3(h3)

            # return h1 + h2 + h3
            return self.deprgr_dec.forward(h1 + h2 + h3)

        # x = self.upsample3(x_dic["h8"])
        # return self.deprgr_dec.forward(x)

        x = self.deprgr_dec.forward(x_dic["h8"])
        return self.upsample3(x)

    def boundary_forward(self, x_dic):
        h1 = self.conv1(x_dic["h2"])
        h1 = self.upsample1(h1)
        h2 = self.conv2(x_dic["h3"])
        h2 = self.upsample2(h2)
        h3 = self.conv3(x_dic["h8"])
        h3 = self.upsample3(h3)

        boundary_pred = (F.sigmoid(h1) + F.sigmoid(h2) + F.sigmoid(h3)) / 3

        return boundary_pred

    def forward(self, x_dic):
        pred_semseg = self.semseg_forward(x_dic)
        pred_dep = self.depth_forward(x_dic)
        pred_boundary = self.boundary_forward(x_dic)
        return pred_semseg, pred_dep, pred_boundary

    def get_semseg_loss(self, x_dic, gt_semseg):
        pred_semseg = self.semseg_forward(x_dic)
        loss = self.semseg_criterion(pred_semseg, gt_semseg)

        return loss

    def get_depth_loss(self, x_dic, gt_dep):
        pred_dep = self.depth_forward(x_dic)
        return F.mse_loss(pred_dep, gt_dep)

    def get_boundary_loss(self, x_dic, gt_boundary):
        pred_boundary = self.boundary_forward(x_dic)
        return bce2d(pred_boundary, gt_boundary)

    def get_loss(self, x, gt_semseg, gt_dep, gt_boundary, separately_returning=False):
        org_semseg_loss = self.get_semseg_loss(x, gt_semseg)
        semseg_loss = torch.exp(-self.s_semsegcls) * org_semseg_loss + self.s_semsegcls

        depreg_loss = torch.exp(-self.s_deprgr) * self.get_depth_loss(x, gt_dep) + self.s_deprgr

        boundary_loss = torch.exp(-self.s_boundary) * self.get_boundary_loss(x, gt_boundary) + self.s_boundary

        if separately_returning:
            return semseg_loss, depreg_loss, boundary_loss

        return semseg_loss + depreg_loss + boundary_loss

    def get_task_weights(self):
        std_semseg = np.sqrt(np.exp(2 * self.s_semsegcls.data.cpu().numpy()))
        std_depth = np.sqrt(np.exp(2 * self.s_deprgr.data.cpu().numpy()))
        return std_semseg, std_depth


class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        if self.label_list is not None:
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)


def validate(val_loader, model, criterion, eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()
        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        if eval_score is not None:
            score.update(eval_score(output, target_var), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score))

    print(' * Score {top1.avg:.3f}'.format(top1=score))

    return score.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.data[0]


def train(train_loader, model, criterion, optimizer, epoch,
          eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        if eval_score is not None:
            scores.update(eval_score(output, target_var), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=scores))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, None,
                          pretrained=True)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()
    criterion = nn.NLLLoss2d(ignore_index=255)

    criterion.cuda()

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    train_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'train', transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'val', transforms.Compose([
            transforms.RandomCrop(crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True
    )

    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(single_model.optim_parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, eval_score=accuracy)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        print('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              eval_score=accuracy)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, eval_score=accuracy)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = 'checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % 10 == 0:
            history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind])
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    for ind in range(len(filenames)):
        im = Image.fromarray(palettes[predictions[ind].squeeze()])
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def test(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    for iter, (image, label, name) in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        image_var = Variable(image, requires_grad=False, volatile=True)
        final = model(image_var)[0]
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(pred, name, output_dir + '_color',
                                 CITYSCAPE_PALLETE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            print('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        print('Eval: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              .format(iter, len(eval_data_loader), batch_time=batch_time,
                      data_time=data_time))
    if has_gt:  # val
        ious = per_class_iu(hist) * 100
        print(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()

    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    test_loader = torch.utils.data.DataLoader(
        SegList(data_dir, phase, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), out_name=True),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    mAP = test(test_loader, model, args.classes, save_vis=True,
               has_gt=phase != 'test',
               output_dir='pred_{:03d}'.format(start_epoch))
    print('mAP: ', mAP)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default=None)
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    args = parser.parse_args()

    assert args.data_dir is not None
    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    return args


def main():
    args = parse_args()
    if args.cmd == 'train':
        train_seg(args)
    elif args.cmd == 'test':
        test_seg(args)

# if __name__ == '__main__':
#     main()
