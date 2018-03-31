import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FuseBase(nn.Module):
    def __init__(self, input_ch=6):
        super(FuseBase, self).__init__()
        assert input_ch in [4, 6]
        self.depth_ch = input_ch - 3

        feats = list(models.vgg16(pretrained=True).features.children())
        feats2 = list(models.vgg16(pretrained=True).features.children())

        print('feats[0] shape: ', feats[0].weight.data.size())
        # print('feats[1] shape: ', feats[2].weight.data.size())

        ########  DEPTH ENCODER  ########
        if self.depth_ch == 1:
            # Take the average of the weights for the depth branch over channel dimension
            avg = torch.mean(feats[0].weight.data, dim=1)

            self.conv11d = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.conv11d.weight.data = avg

            self.CBR1_D = nn.Sequential(
                nn.BatchNorm2d(64),
                feats[1],
                feats[2],
                nn.BatchNorm2d(64),
                feats[3],
            )
        elif self.depth_ch == 3:
            self.CBR1_D = nn.Sequential(
                feats[0],
                nn.BatchNorm2d(64),
                feats[1],
                feats[2],
                nn.BatchNorm2d(64),
                feats[3],
            )
        else:
            raise NotImplementedError()

        self.CBR2_D = nn.Sequential(
            feats[5],
            nn.BatchNorm2d(128),
            feats[6],
            feats[7],
            nn.BatchNorm2d(128),
            feats[8],
        )
        self.CBR3_D = nn.Sequential(
            feats[10],
            nn.BatchNorm2d(256),
            feats[11],
            feats[12],
            nn.BatchNorm2d(256),
            feats[13],
            feats[14],
            nn.BatchNorm2d(256),
            feats[15],
        )

        self.dropout3_d = nn.Dropout(p=0.5)

        self.CBR4_D = nn.Sequential(
            feats[17],
            nn.BatchNorm2d(512),
            feats[18],
            feats[19],
            nn.BatchNorm2d(512),
            feats[20],
            feats[21],
            nn.BatchNorm2d(512),
            feats[22],
        )

        self.dropout4_d = nn.Dropout(p=0.5)

        self.CBR5_D = nn.Sequential(
            feats[24],
            nn.BatchNorm2d(512),
            feats[25],
            feats[26],
            nn.BatchNorm2d(512),
            feats[27],
            feats[28],
            nn.BatchNorm2d(512),
            feats[29],
        )

        ########  RGB ENCODER  ########

        self.CBR1_RGB = nn.Sequential(
            feats2[0],
            nn.BatchNorm2d(64),
            feats2[1],
            feats2[2],
            nn.BatchNorm2d(64),
            feats2[3],
        )

        self.CBR2_RGB = nn.Sequential(
            feats2[5],
            nn.BatchNorm2d(128),
            feats2[6],
            feats2[7],
            nn.BatchNorm2d(128),
            feats2[8],
        )

        self.CBR3_RGB = nn.Sequential(
            feats2[10],
            nn.BatchNorm2d(256),
            feats2[11],
            feats2[12],
            nn.BatchNorm2d(256),
            feats2[13],
            feats2[14],
            nn.BatchNorm2d(256),
            feats2[15],
        )

        self.dropout3 = nn.Dropout(p=0.5)

        self.CBR4_RGB = nn.Sequential(
            feats2[17],
            nn.BatchNorm2d(512),
            feats2[18],
            feats2[19],
            nn.BatchNorm2d(512),
            feats2[20],
            feats2[21],
            nn.BatchNorm2d(512),
            feats2[22],
        )

        self.dropout4 = nn.Dropout(p=0.5)

        self.CBR5_RGB = nn.Sequential(
            feats2[24],
            nn.BatchNorm2d(512),
            feats2[25],
            feats2[26],
            nn.BatchNorm2d(512),
            feats2[27],
            feats2[28],
            nn.BatchNorm2d(512),
            feats2[29],
        )

        self.dropout5 = nn.Dropout(p=0.5)

    def forward(self, x):
        rgb_inputs = x[:, :3, :, :]
        depth_inputs = x[:, 3:, :, :]

        ########  DEPTH ENCODER  ########

        # Stage 1
        if self.depth_ch == 1:
            depth_inputs = self.conv11d(depth_inputs)

        x_1 = self.CBR1_D(depth_inputs)

        x, id1_d = F.max_pool2d(x_1, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x_2 = self.CBR2_D(x)
        x, id2_d = F.max_pool2d(x_2, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x_3 = self.CBR3_D(x)
        x, id3_d = F.max_pool2d(x_3, kernel_size=2, stride=2, return_indices=True)
        x = self.dropout3_d(x)

        # Stage 4
        x_4 = self.CBR4_D(x)
        x, id4_d = F.max_pool2d(x_4, kernel_size=2, stride=2, return_indices=True)
        x = self.dropout4_d(x)

        # Stage 5
        x_5 = self.CBR5_D(x)

        ########  RGB ENCODER  ########

        # Stage 1
        y = self.CBR1_RGB(rgb_inputs)
        y = torch.add(y, x_1)
        y, id1 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        y = self.CBR2_RGB(y)
        y = torch.add(y, x_2)
        y, id2 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        y = self.CBR3_RGB(y)
        y = torch.add(y, x_3)
        y, id3 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout3(y)

        # Stage 4
        y = self.CBR4_RGB(y)
        y = torch.add(y, x_4)
        y, id4 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout4(y)

        # Stage 5
        y = self.CBR5_RGB(y)
        y = torch.add(y, x_5)
        y_size = y.size()

        y, id5 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout5(y)

        fet_dic = {
            "out": y,
            "id5": id5,
            "id4": id4,
            "id3": id3,
            "id2": id2,
            "id1": id1,
            "y_size": y_size

        }
        return fet_dic


class FuseClassifier(nn.Module):
    def __init__(self, n_class):
        super(FuseClassifier, self).__init__()
        batchNorm_momentum = 0.1

        ########  RGB DECODER  ########

        self.CBR5_Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.CBR4_Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.CBR3_Dec = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.CBR2_Dec = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(),
        )

        self.CBR1_Dec = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(64, n_class, kernel_size=3, padding=1),
        )

    def forward(self, fet_dic):
        ########  DECODER  ########
        # (1L, 512L, 14L, 14L)
        # (1L, 512L, 14L, 14L)
        # (1L, 512L, 28L, 28L)
        # (1L, 256L, 28L, 28L)
        # (1L, 256L, 56L, 56L)
        # (1L, 128L, 56L, 56L)
        # (1L, 128L, 112L, 112L)
        # (1L, 64L, 112L, 112L)
        # (1L, 64L, 224L, 224L)
        # (1L, 41L, 224L, 224L)

        # Stage 5 dec
        y = F.max_unpool2d(fet_dic["out"], fet_dic["id5"], kernel_size=2, stride=2, output_size=fet_dic["y_size"])
        # print (y.size())
        y = self.CBR5_Dec(y)
        # print (y.size())

        # Stage 4 dec
        y = F.max_unpool2d(y, fet_dic["id4"], kernel_size=2, stride=2)
        # print (y.size())
        y = self.CBR4_Dec(y)
        # print (y.size())

        # Stage 3 dec
        y = F.max_unpool2d(y, fet_dic["id3"], kernel_size=2, stride=2)
        # print (y.size())
        y = self.CBR3_Dec(y)
        # print (y.size())

        # Stage 2 dec
        y = F.max_unpool2d(y, fet_dic["id2"], kernel_size=2, stride=2)
        # print (y.size())
        y = self.CBR2_Dec(y)
        # print (y.size())

        # Stage 1 dec
        y = F.max_unpool2d(y, fet_dic["id1"], kernel_size=2, stride=2)
        # print (y.size())
        y = self.CBR1_Dec(y)
        # print (y.size())
        return y


class FuseNet(nn.Module):
    def __init__(self, n_class, input_ch):
        super(FuseNet, self).__init__()

        assert input_ch in [4, 6]
        self.depth_ch = input_ch - 3

        batchNorm_momentum = 0.1
        feats = list(models.vgg16(pretrained=True).features.children())
        feats2 = list(models.vgg16(pretrained=True).features.children())

        # print('feats[0] shape: ', feats[0].weight.data.size())1
        # print('feats[1] shape: ', feats[2].weight.data.size())

        # Take the average of the weights for the depth branch over channel dimension 
        avg = torch.mean(feats[0].weight.data, dim=1)

        ########  DEPTH ENCODER  ########

        # self.conv11d = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # self.conv11d.weight.data = avg
        #
        # self.CBR1_D = nn.Sequential(
        #     nn.BatchNorm2d(64),
        #     feats[1],
        #     feats[2],
        #     nn.BatchNorm2d(64),
        #     feats[3],
        # )

        if self.depth_ch == 1:
            # Take the average of the weights for the depth branch over channel dimension
            avg = torch.mean(feats[0].weight.data, dim=1)
            ########  DEPTH ENCODER  ########

            self.conv11d = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.conv11d.weight.data = avg

            self.CBR1_D = nn.Sequential(
                nn.BatchNorm2d(64),
                feats[1],
                feats[2],
                nn.BatchNorm2d(64),
                feats[3],
            )
        elif self.depth_ch == 3:
            self.CBR1_D = nn.Sequential(
                feats[0],
                nn.BatchNorm2d(64),
                feats[1],
                feats[2],
                nn.BatchNorm2d(64),
                feats[3],
            )
        else:
            raise NotImplementedError()

        self.CBR2_D = nn.Sequential(
            feats[5],
            nn.BatchNorm2d(128),
            feats[6],
            feats[7],
            nn.BatchNorm2d(128),
            feats[8],
        )
        self.CBR3_D = nn.Sequential(
            feats[10],
            nn.BatchNorm2d(256),
            feats[11],
            feats[12],
            nn.BatchNorm2d(256),
            feats[13],
            feats[14],
            nn.BatchNorm2d(256),
            feats[15],
        )

        self.dropout3_d = nn.Dropout(p=0.5)

        self.CBR4_D = nn.Sequential(
            feats[17],
            nn.BatchNorm2d(512),
            feats[18],
            feats[19],
            nn.BatchNorm2d(512),
            feats[20],
            feats[21],
            nn.BatchNorm2d(512),
            feats[22],
        )

        self.dropout4_d = nn.Dropout(p=0.5)

        self.CBR5_D = nn.Sequential(
            feats[24],
            nn.BatchNorm2d(512),
            feats[25],
            feats[26],
            nn.BatchNorm2d(512),
            feats[27],
            feats[28],
            nn.BatchNorm2d(512),
            feats[29],
        )

        ########  RGB ENCODER  ########

        self.CBR1_RGB = nn.Sequential(
            feats2[0],
            nn.BatchNorm2d(64),
            feats2[1],
            feats2[2],
            nn.BatchNorm2d(64),
            feats2[3],
        )

        self.CBR2_RGB = nn.Sequential(
            feats2[5],
            nn.BatchNorm2d(128),
            feats2[6],
            feats2[7],
            nn.BatchNorm2d(128),
            feats2[8],
        )

        self.CBR3_RGB = nn.Sequential(
            feats2[10],
            nn.BatchNorm2d(256),
            feats2[11],
            feats2[12],
            nn.BatchNorm2d(256),
            feats2[13],
            feats2[14],
            nn.BatchNorm2d(256),
            feats2[15],
        )

        self.dropout3 = nn.Dropout(p=0.5)

        self.CBR4_RGB = nn.Sequential(
            feats2[17],
            nn.BatchNorm2d(512),
            feats2[18],
            feats2[19],
            nn.BatchNorm2d(512),
            feats2[20],
            feats2[21],
            nn.BatchNorm2d(512),
            feats2[22],
        )

        self.dropout4 = nn.Dropout(p=0.5)

        self.CBR5_RGB = nn.Sequential(
            feats2[24],
            nn.BatchNorm2d(512),
            feats2[25],
            feats2[26],
            nn.BatchNorm2d(512),
            feats2[27],
            feats2[28],
            nn.BatchNorm2d(512),
            feats2[29],
        )

        self.dropout5 = nn.Dropout(p=0.5)

        ########  RGB DECODER  ########

        self.CBR5_Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.CBR4_Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.CBR3_Dec = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.CBR2_Dec = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(),
        )

        self.CBR1_Dec = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(64, n_class, kernel_size=3, padding=1),
        )

    def forward(self, x):
        rgb_inputs = x[:, :3, :, :]
        depth_inputs = x[:, 3:, :, :]

        ########  DEPTH ENCODER  ########

        # Stage 1
        # x = self.conv11d(depth_inputs)

        # Stage 1
        if self.depth_ch == 1:
            depth_inputs = self.conv11d(depth_inputs)

        x_1 = self.CBR1_D(depth_inputs)
        x, id1_d = F.max_pool2d(x_1, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x_2 = self.CBR2_D(x)
        x, id2_d = F.max_pool2d(x_2, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x_3 = self.CBR3_D(x)
        x, id3_d = F.max_pool2d(x_3, kernel_size=2, stride=2, return_indices=True)
        x = self.dropout3_d(x)

        # Stage 4
        x_4 = self.CBR4_D(x)
        x, id4_d = F.max_pool2d(x_4, kernel_size=2, stride=2, return_indices=True)
        x = self.dropout4_d(x)

        # Stage 5
        x_5 = self.CBR5_D(x)

        ########  RGB ENCODER  ########

        # Stage 1
        y = self.CBR1_RGB(rgb_inputs)
        y = torch.add(y, x_1)
        y, id1 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        y = self.CBR2_RGB(y)
        y = torch.add(y, x_2)
        y, id2 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        y = self.CBR3_RGB(y)
        y = torch.add(y, x_3)
        y, id3 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout3(y)

        # Stage 4
        y = self.CBR4_RGB(y)
        y = torch.add(y, x_4)
        y, id4 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout4(y)

        # Stage 5
        y = self.CBR5_RGB(y)
        y = torch.add(y, x_5)
        y_size = y.size()

        y, id5 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout5(y)

        ########  DECODER  ########

        # Stage 5 dec
        y = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)
        y = self.CBR5_Dec(y)

        # Stage 4 dec
        y = F.max_unpool2d(y, id4, kernel_size=2, stride=2)
        y = self.CBR4_Dec(y)

        # Stage 3 dec
        y = F.max_unpool2d(y, id3, kernel_size=2, stride=2)
        y = self.CBR3_Dec(y)

        # Stage 2 dec
        y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
        y = self.CBR2_Dec(y)

        # Stage 1 dec
        y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
        y = self.CBR1_Dec(y)

        return y

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print 'Saving model: %s' % path
        torch.save(self, path)
        print 'Model saved: %s' % path


def CrossEntropy2d():
    def wrap(inputs, targets, weight=None, pixel_average=True):
        n, c, h, w = inputs.size()

        if c == 37:
            weight = torch.cuda.FloatTensor(
                [0.31142759, 0.26649606, 0.45942909, 0.32240534, 0.54789394, 0.4269788, 0.76315141, 1.11409545,
                 0.96722591, 0.57659554, 1.66651666, 0.85155034, 1.03507304, 0.59151018, 1.07225466, 0.76207125,
                 0.67946768, 2.3853786, 1.64862466, 1.75271165, 3.24660635, 1.16477966, 2.37583423, 0.87280464,
                 1.55249476, 5.12412119, 1.94428802, 0.64293331, 3.18023825, 0.85495919, 3.15664768, 2.11753082,
                 0.55160081, 1.57176685, 5.1366291, 0.45877823, 4.90023994])
        elif c == 40:
            weight = torch.cuda.FloatTensor(
                [0.272491, 0.568953, 0.432069, 0.354511, 0.82178, 0.506488, 1.133686, 0.81217, 0.789383, 0.380358,
                 1.650497, 1, 0.650831, 0.757218, 0.950049, 0.614332, 0.483815, 1.842002, 0.635787, 1.176839, 1.196984,
                 1.111907, 1.927519, 0.695354, 1.057833, 4.179196, 1.571971, 0.432408, 3.705966, 0.549132, 1.282043,
                 2.329812, 0.992398, 3.114945, 5.466101, 1.085242, 6.968411, 1.093939, 1.33652, 1.228912])
        print("was here: NYU weight")

        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
        inputs = inputs[targets.view(n, h, w, 1).repeat(1, 1, 1, c) > 0].view(-1, c)

        targets_mask = targets > 0
        targets = targets[targets_mask] - 1

        loss = F.cross_entropy(inputs, targets, weight=weight, size_average=False)
        if pixel_average:
            loss /= targets_mask.data.sum()
        return loss

    return wrap
