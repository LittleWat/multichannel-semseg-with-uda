import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetConv(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UNetConv, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(), )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(UNetUp, self).__init__()
        self.conv = UNetConv(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, shortcut, h):
        h_up = self.up(h)
        offset = h_up.size()[2] - shortcut.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        processed_shortcut = F.pad(shortcut, padding)
        return self.conv(torch.cat([processed_shortcut, h_up], 1))


class MultiUNetUp(nn.Module):
    def __init__(self, shortcut_ch, h_ch, out_size, is_deconv):
        super(MultiUNetUp, self).__init__()
        self.conv = UNetConv(shortcut_ch * 3, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(h_ch, shortcut_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, shortcut, shortcut2, h):
        # print (map(lambda x: x.size(), [shortcut, shortcut2, h]))
        h_up = self.up(h)
        offset = h_up.size()[2] - shortcut.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        processed_shortcut = F.pad(shortcut, padding)
        processed_shortcut2 = F.pad(shortcut2, padding)
        # print (map(lambda x: x.size(), [processed_shortcut, processed_shortcut2, h_up]))

        return self.conv(torch.cat([processed_shortcut, processed_shortcut2, h_up], 1))


class UNet(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, input_ch=3, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.input_ch = input_ch
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UNetConv(self.input_ch, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UNetConv(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UNetConv(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UNetConv(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UNetConv(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UNetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UNetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UNetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UNetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        return final


class UNetBase(nn.Module):
    def __init__(self, feature_scale=4, is_deconv=True, input_ch=3, is_batchnorm=True):
        super(UNetBase, self).__init__()
        self.is_deconv = is_deconv
        self.input_ch = input_ch
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UNetConv(self.input_ch, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UNetConv(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UNetConv(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UNetConv(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UNetConv(filters[3], filters[4], self.is_batchnorm)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        final = {
            "center": center,
            "conv1": conv1,
            "conv2": conv2,
            "conv3": conv3,
            "conv4": conv4
        }

        return final


class UNetClassifier(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True):
        super(UNetClassifier, self).__init__()
        self.is_deconv = is_deconv
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # upsampling
        self.up_concat4 = UNetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UNetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UNetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UNetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        up4 = self.up_concat4(inputs["conv4"], inputs["center"])
        up3 = self.up_concat3(inputs["conv3"], up4)
        up2 = self.up_concat2(inputs["conv2"], up3)
        up1 = self.up_concat1(inputs["conv1"], up2)

        final = self.final(up1)
        return final


class MultiUNetClassifier(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True):
        super(MultiUNetClassifier, self).__init__()
        self.is_deconv = is_deconv
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # upsampling
        self.up_concat4 = MultiUNetUp(shortcut_ch=filters[3], h_ch=filters[4] * 2, out_size=filters[3],
                                      is_deconv=self.is_deconv)
        self.up_concat3 = MultiUNetUp(shortcut_ch=filters[2], h_ch=filters[3], out_size=filters[2],
                                      is_deconv=self.is_deconv)
        self.up_concat2 = MultiUNetUp(shortcut_ch=filters[1], h_ch=filters[2], out_size=filters[1],
                                      is_deconv=self.is_deconv)
        self.up_concat1 = MultiUNetUp(shortcut_ch=filters[0], h_ch=filters[1], out_size=filters[0],
                                      is_deconv=self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs1, inputs2):
        center = torch.cat([inputs1["center"], inputs2["center"]], 1)
        # center = inputs1["center"] + inputs2["center"]
        # print (center.size(), center2.size())

        up4 = self.up_concat4(inputs1["conv4"], inputs2["conv4"], center)
        up3 = self.up_concat3(inputs1["conv3"], inputs2["conv3"], up4)
        up2 = self.up_concat2(inputs1["conv2"], inputs2["conv2"], up3)
        up1 = self.up_concat1(inputs1["conv1"], inputs1["conv1"], up2)

        final = self.final(up1)
        return final
