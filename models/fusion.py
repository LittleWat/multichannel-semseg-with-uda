import torch
import torch.nn as nn
import torch.nn.functional as F


class GateFusion(nn.Module):
    def __init__(self, inplanes, apply_softmax=False):
        super(GateFusion, self).__init__()
        self.conv = nn.Conv2d(inplanes * 2, inplanes, kernel_size=1, stride=1)
        self.apply_softmax = apply_softmax

    def forward(self, x1, x2):
        if self.apply_softmax:
            x1 = F.softmax(x1)
            x2 = F.softmax(x2)

        concat = torch.cat([x1, x2], 1)
        gate = F.sigmoid(self.conv(concat))
        p1 = x1 * gate
        p2 = x2 * (1 - gate)
        return p1 + p2


class AddFusion(nn.Module):
    def __init__(self):
        super(AddFusion, self).__init__()

    def forward(self, x1, x2):
        return x1 + x2


class ConcatFusion(nn.Module):
    def __init__(self):
        super(ConcatFusion, self).__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], 1)


class ConcatConvFusion(nn.Module):
    def __init__(self, inplanes):
        super(ConcatConvFusion, self).__init__()

        # self.conv = nn.Conv2d(inplanes * 2, inplanes, kernel_size=1, stride=1) # BAD?
        self.conv = nn.Conv2d(inplanes * 2, inplanes, kernel_size=3, padding=1)  # GOOD?

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], 1)
        h = self.conv(concat)
        return h


def get_fusion_model(fusion_type, n_ch):
    if "ScoreGateFusion" in fusion_type:
        return GateFusion(n_ch, apply_softmax=True)
    if "GateFusion" in fusion_type:
        return GateFusion(n_ch)
    elif "AddFusion" in fusion_type:
        return AddFusion()
    elif "ConcatFusion" in fusion_type:
        return ConcatFusion()
    elif "ConcatConvFusion" in fusion_type:
        return ConcatConvFusion(n_ch)
    else:
        raise NotImplementedError()
