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



class DT(nn.Module):
    def __init__(self, size):
        super(DT, self).__init__()



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