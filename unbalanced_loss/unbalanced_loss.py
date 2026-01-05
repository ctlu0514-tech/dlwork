import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinaryFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        target = target.unsqueeze(dim=1)
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss

#  some code reforenced from  https://github.com/DHPO/GHM_Loss.pytorch
class GHM_Loss(nn.Module):
    def __init__(self, bins, alpha):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx])


class GHMC_Loss(GHM_Loss):
    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight.cuda())

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


class WBCEWithLogitLoss(nn.Module):
    """
    Weighted Binary Cross Entropy.
    `WBCE(p,t)=-β*t*log(p)-(1-t)*log(1-p)`
    To decrease the number of false negatives, set β>1.
    To decrease the number of false positives, set β<1.
    Args:
            @param weight: positive sample weight
        Shapes：
            output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
    """

    def __init__(self, weight=1.0, ignore_index=None, reduction='mean'):
        super(WBCEWithLogitLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        weight = float(weight)
        self.weight = weight
        self.reduction = reduction
        self.smooth = 0.01

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        output = torch.sigmoid(output)
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)

        # loss = self.bce(output, target)
        loss = -self.weight * target.mul(torch.log(output)) - ((1.0 - target).mul(torch.log(1.0 - output)))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss

