import torch.nn as nn
import torch


class MaxMin(nn.Module):
    # code from https://github.com/cemanil/LNets/blob/master/lnets/models/activations/maxout.py
    def __init__(self):
        super(MaxMin, self).__init__()

    def forward(self, x):
        y = torch.reshape(x, (x.shape[0], x.shape[1] // 2, 2, x.shape[2], x.shape[3]))
        maxes, _ = torch.max(y, 2)
        mins, _ = torch.min(y, 2)
        maxmin = torch.cat((maxes, mins), dim=1)
        return maxmin

    def extra_repr(self):
        return 'num_units: {}'.format(self.num_units)