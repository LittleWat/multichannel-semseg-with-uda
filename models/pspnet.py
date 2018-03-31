import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from models import dilated_resnet
from models.drn import drn_d_54


class DRNRes(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(DRNRes, self).__init__()
        drn = drn_d_54(pretrained=True)
        self.layer0 = drn.layer0
        self.layer1 = drn.layer1
        self.layer2 = drn.layer2
        self.layer3 = drn.layer3
        self.layer4 = drn.layer4
        self.layer5 = drn.layer5
        self.layer6 = drn.layer6
        self.layer7 = drn.layer7
        self.layer8 = drn.layer8

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        return x, x_3


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    @staticmethod
    def _make_stage(features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNetGradual(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6)):
        super(PSPNetGradual, self).__init__()
        self.feats = dilated_resnet.ResNet(Bottleneck, [3, 4, 23, 3])
        self.psp = PSPModule(2048, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

    def forward(self, x):
        f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        return self.final(p)


class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out


# TODO implement aux
class PSPNet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), layer='50', input_ch=3):
        super(PSPNet, self).__init__()
        if layer == '18':
            self.feats = dilated_resnet.resnet18(pretrained=True, input_ch=input_ch)
        elif layer == '34':
            self.feats = dilated_resnet.resnet34(pretrained=True, input_ch=input_ch)
        elif layer == '50':
            self.feats = dilated_resnet.resnet50(pretrained=True, input_ch=input_ch)
        elif layer == '101':
            self.feats = dilated_resnet.resnet101(pretrained=True, input_ch=input_ch)
        elif layer == '152':
            self.feats = dilated_resnet.resnet152(pretrained=True, input_ch=input_ch)
        else:
            NotImplementedError()
        # import drn
        # model = drn.drn_d_54()
        # self.feats = nn.Sequential(*list(model.children())[:-4]) # -2:512
        # self.psp = PSPModule(2048, 1024, sizes)
        # self.drop_1 = nn.Dropout2d(p=0.3)
        #
        # self.up_1 = PSPUpsample(1024, 256)
        # self.up_2 = PSPUpsample(256, 64)
        # self.up_3 = PSPUpsample(64, 64)
        #
        # self.drop_2 = nn.Dropout2d(p=0.15)
        # self.final = nn.Sequential(
        #     nn.Conv2d(64, n_classes, kernel_size=1),
        #     nn.LogSoftmax()
        # )
        #
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, n_classes, kernel_size=1))

    def forward(self, x):
        input_size = x.size()
        f, class_f = self.feats(x)  # class_f has 1024 channels and is 8x downsampled

        # print ("fet size")
        # print (f.size())
        # p = self.psp(f)
        # p = self.drop_1(p)
        # p = self.up_1(p)
        # p = self.drop_2(p)
        # p = self.up_2(p)
        # p = self.drop_2(p)
        # p = self.up_3(p)
        # p = self.drop_2(p)
        # p = self.final(p)

        p = self.ppm(f)
        p = self.final(p)
        p = F.upsample(p, input_size[2:], mode='bilinear')

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, 1024)

        return p, self.classifier(auxiliary)


class PSPBase(nn.Module):
    def __init__(self, layer="50", input_ch=3):
        super(PSPBase, self).__init__()
        # self.feats = ResNet(Bottleneck, [3, 4, 23, 3]) ## ResNet101: This Causes the Shortage of Memory
        # self.feats = ResNet(Bottleneck, [3, 4, 6, 3])  ## ResNet50
        # self.feats = DRNRes()

        if layer == '18':
            self.feats = dilated_resnet.resnet18(pretrained=True, input_ch=input_ch)
        elif layer == '34':
            self.feats = dilated_resnet.resnet34(pretrained=True, input_ch=input_ch)
        elif layer == '50':
            self.feats = dilated_resnet.resnet50(pretrained=True, input_ch=input_ch)
        elif layer == '101':
            self.feats = dilated_resnet.resnet101(pretrained=True, input_ch=input_ch)
        elif layer == '152':
            self.feats = dilated_resnet.resnet152(pretrained=True, input_ch=input_ch)
        else:
            NotImplementedError()

    def forward(self, x):
        f, class_f = self.feats(x)

        outdic = {
            "f": f,
            "class_f": class_f
        }
        return outdic


class PSPClassifier(nn.Module):
    def __init__(self, num_classes, sizes=(1, 2, 3, 6)):
        super(PSPClassifier, self).__init__()
        # changed 2048 to 512 to match drn
        # self.psp = PSPModule(512, 1024, sizes)

        self.psp = PSPModule(2048, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.LogSoftmax()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, g_outdic):
        g_outdic = edict(g_outdic)

        p = self.psp(g_outdic.f)
        p = self.drop_1(p)
        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        p = self.final(p)
        # changed 1024 to 256 to match drn
        auxiliary = F.adaptive_max_pool2d(input=g_outdic.class_f, output_size=(1, 1)).view(-1,
                                                                                           256)  # for cls_loss computation

        return p
