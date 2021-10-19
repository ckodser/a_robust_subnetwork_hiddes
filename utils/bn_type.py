import torch.nn as nn
import torch

LearnedBatchNorm = nn.BatchNorm2d


class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


class MeanNorm(nn.Module):
    # code from https://github.com/zbh2047/L_inf-dist-net/blob/main/model/norm_dist.py

    def __init__(self, dim, momentum=0.1):
        super(MeanNorm, self).__init__()
        self.out_channels = dim
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(self.out_channels))

    def forward(self, x):
        y = x.view(x.size(0), x.size(1), -1)
        if self.training:
            if x.dim() > 2:
                mean = y.mean(dim=-1).mean(dim=0)
            else:
                mean = x.mean(dim=0)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
        else:
            mean = self.running_mean
        x = (y - mean.unsqueeze(-1)).view_as(x)
        return x

    def extra_repr(self):
        return '{num_features}'.format(num_features=self.out_channels)
