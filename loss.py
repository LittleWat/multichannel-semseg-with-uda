import torch
import torch.nn as nn
import torch.nn.functional as F


# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class ProbCrossEntropyLoss2d(nn.Module):
    """
    Calc cross entropy loss between prob tensor (0~1) and GT
    """

    def __init__(self, weight=None, size_average=True):
        super(ProbCrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        # n_element = np.prod(np.array(inputs.size()))
        # assert torch.sum(inputs.data >= 0) == n_element
        # assert torch.sum(inputs.data <= 1) == n_element

        return self.nll_loss(torch.log(inputs), targets)


class BalanceLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BalanceLoss2d, self).__init__()
        # self.nll_loss = nn.NLLLoss2d(weight, size_average)
        self.weight = weight

    def forward(self, inputs1, inputs2):
        prob1 = F.softmax(inputs1)[0, :19]
        prob2 = F.softmax(inputs2)[0, :19]
        print prob1
        prob1 = torch.mean(prob1, 0)
        prob2 = torch.mean(prob2, 0)
        print prob1
        entropy_loss = - torch.mean(torch.log(prob1 + 1e-6))
        entropy_loss -= torch.mean(torch.log(prob2 + 1e-6))
        return entropy_loss


class Entropy(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Entropy, self).__init__()
        self.weight = weight

    def forward(self, inputs1):
        prob1 = F.softmax(inputs1[0, :19])
        # prob1 = nn.LogSoftmax()(inputs1)
        # prob2 = F.softmax(inputs2)[0,:19]
        # prob1 = torch.mean(prob1,0)
        # print prob1
        entropy_loss = torch.mean(torch.log(prob1))  # torch.mean(torch.mean(torch.log(prob1),1),0
        # print(entropy_loss)
        return entropy_loss


# This metric is strange but somehow works well
class MisSymKLD(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MisSymKLD, self).__init__()
        # self.nll_loss = nn.NLLLoss2d(weight, size_average)
        self.weight = weight

    def forward(self, inputs1, inputs2):
        return 0.5 * (F.kl_div(F.softmax(inputs1)[:, :, :, :], F.softmax(inputs2)[:, :, :, :]) + F.kl_div(
            F.softmax(inputs2)[:, :, :, :], F.softmax(inputs1)[:, :, :, :]))


class JSD(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(JSD, self).__init__()
        # self.nll_loss = nn.NLLLoss2d(weight, size_average)
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs1, inputs2):
        m = 0.5 * (inputs1 + inputs2)
        return 0.5 * (
            F.kl_div(F.log_softmax(m), F.softmax(inputs1), size_average=self.size_average) +
            F.kl_div(F.log_softmax(m), F.softmax(inputs2), size_average=self.size_average))


class Diff2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Diff2d, self).__init__()
        # self.nll_loss = nn.NLLLoss2d(weight, size_average)
        self.weight = weight

    def forward(self, inputs1, inputs2):
        return torch.mean(torch.abs(F.softmax(inputs1) - F.softmax(inputs2)))


class Symkl2d(nn.Module):
    def __init__(self, weight=None, n_target_ch=None, size_average=True):
        super(Symkl2d, self).__init__()
        # self.nll_loss = nn.NLLLoss2d(weight, size_average)
        self.weight = weight
        self.size_average = size_average
        self.n_target_ch = n_target_ch

    def forward(self, inputs1, inputs2):
        self.prob1 = F.softmax(inputs1).view(-1, self.n_target_ch)
        self.prob2 = F.softmax(inputs2).view(-1, self.n_target_ch)
        self.log_prob1 = F.log_softmax(inputs1).view(-1, self.n_target_ch)
        self.log_prob2 = F.log_softmax(inputs2).view(-1, self.n_target_ch)
        return 0.5 * (F.kl_div(self.log_prob1, self.prob2, size_average=self.size_average)
                      + F.kl_div(self.log_prob2, self.prob1, size_average=self.size_average))


def kl_calc(prob1, prob2):
    return prob1 * torch.log(prob1 / prob2)


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"


# binary cross entropy loss in 2D
def bce2d(input, target):
    # do not compute gradients w.r.t target
    _assert_no_grad(target)

    beta = 1 - torch.mean(target)
    weights = 1 - beta + (2 * beta - 1) * target
    loss = F.binary_cross_entropy(input, target, weights, size_average=True)
    return loss


class MySymkl2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MySymkl2d, self).__init__()
        # self.nll_loss = nn.NLLLoss2d(weight, size_average)
        self.weight = weight

    def forward(self, inputs1, inputs2):
        prob1 = F.softmax(inputs1)
        prob2 = F.softmax(inputs2)
        loss = 0.5 * (kl_calc(prob1, prob2) + kl_calc(prob2, prob1))
        return torch.mean(loss)


class SpatialJSD2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SpatialJSD2d, self).__init__()
        # self.nll_loss = nn.NLLLoss2d(weight, size_average)
        self.weight = weight

    def forward(self, inputs1, inputs2):
        input1 = inputs1[0, :, :, :]
        input2 = inputs2[0, :, :, :]
        input_size = input1.size()
        input1 = input1.view(-1, input_size[1] * input_size[2])

        input2 = input2.view(-1, input_size[1] * input_size[2])
        # input1 = input1.transpose(1,0)
        # input2 = input2.transpose(1,0)
        # print(torch.sum(F.softmax(input1)[1,:]))
        return 0.5 * (
            F.kl_div(F.softmax(inputs1), F.softmax(inputs2)) + F.kl_div(F.softmax(inputs2), F.softmax(inputs1)))


# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    # loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss


def get_prob_distance_criterion(criterion_name, n_class=None):
    if criterion_name == "jsd":
        criterion = JSD()
    elif criterion_name == 'diff':
        criterion = Diff2d()
    elif criterion_name == "symkl":
        criterion = Symkl2d(n_target_ch=n_class)
    elif criterion_name == "nmlsymkl":
        criterion = Symkl2d(n_target_ch=n_class, size_average=True)
    elif criterion_name == "mysymkl":
        criterion = MySymkl2d()
    elif criterion_name == "spatial_jsd":
        criterion = SpatialJSD2d()
    elif criterion_name == 'mis_symkl':
        criterion = MisSymKLD()
    else:
        raise NotImplementedError()

    return criterion
