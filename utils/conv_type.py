import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math

from args import args as parser_args

DenseConv = nn.Conv2d


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


"""
Sample Based Sparsification
"""


class StraightThroughBinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class BinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        subnet, = ctx.saved_variables

        grad_inputs = grad_outputs.clone()
        grad_inputs[subnet == 0.0] = 0.0

        return grad_inputs, None


# Not learning weights, finding subnet
class SampleSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    @property
    def clamped_scores(self):
        return torch.sigmoid(self.scores)

    def forward(self, x):
        subnet = StraightThroughBinomialSample.apply(self.clamped_scores)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x


"""
Fixed subnets 
"""


class FixedSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print("prune_rate_{}".format(self.prune_rate))

    def set_subnet(self):
        output = self.clamped_scores().clone()
        _, idx = self.clamped_scores().flatten().abs().sort()
        p = int(self.prune_rate * self.clamped_scores().numel())
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        flat_oup[idx[p:]] = 1
        self.scores = torch.nn.Parameter(output)
        self.scores.requires_grad = False

    def clamped_scores(self):
        return self.scores.abs()

    def get_subnet(self):
        return self.weight * self.scores

    def forward(self, x):
        w = self.get_subnet()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class GetFixFanInSubnet(GetSubnet):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten(start_dim=1).sort()
        neuron = scores.size()[0]
        j = int((1 - k) * (scores.numel() / neuron))

        # flat_out and out access the same memory.
        flat_out = out.flatten(start_dim=1)
        for i in range(neuron):
            flat_out[i, idx[i, :j]] = 0
            flat_out[i, idx[i, j:]] = 1

        return out


class GetLipschitzSubnet(GetSubnet):
    @staticmethod
    def forward(ctx, scores, k, weight, lipschitz):
        # Get the subnetwork by sorting the scores for each neuron and using the tops till weights sum reach lipschitz
        goodness = scores  # torch.div(scores, torch.abs(weight))
        out = torch.zeros_like(goodness.shape)
        _, idx = goodness.flatten(start_dim=1).sort(descending=True)
        neuron = goodness.size()[0]
        # flat_out and out access the same memory.
        flat_out = out.flatten(start_dim=1)
        ordered_weight = torch.abs(torch.gather(weight.flatten(start_dim=1), dim=1, index=idx))
        weight_sum = torch.cumsum(ordered_weight, dim=1)
        lim = (weight_sum <= lipschitz).sum(dim=1)
        for i in range(neuron):
            j = lim[i]
            flat_out[i, idx[i, :j]] = 1
            flat_out[i, idx[i, j + 1:]] = 0
            if j < flat_out.shape[1]:
                flat_out[i, idx[i, j]] = torch.div(torch.add(lipschitz, - weight_sum[i, j - 1]),
                                                   ordered_weight[i, j]) if j != 0 else torch.div(lipschitz,
                                                                                                  ordered_weight[i, j])
        # # connection_rate = lim.sum() / scores.numel()
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None, None


# Not learning weights, finding subnet
class FixFanInSubnetConv(SubnetConv):
    def forward(self, x):
        subnet = GetFixFanInSubnet.apply(self.clamped_scores, self.prune_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


# Not learning weights, finding subnet
class LipschitzSubnetConv(SubnetConv):
    def __init__(self, *args, **kwargs):
        super(LipschitzSubnetConv, self).__init__(*args, **kwargs)
        self.lipschitz = 1

    def forward(self, x):
        subnet = GetLipschitzSubnet.apply(self.clamped_scores, self.prune_rate, self.weight, self.lipschitz)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class FixedFixFanInSubnetConv(FixedSubnetConv):
    def set_subnet(self):
        output = self.clamped_scores().clone()
        _, idx = self.clamped_scores().flatten(start_dim=1).abs().sort()
        neuron = self.clamped_scores().size()[0]
        p = int(self.prune_rate * (self.clamped_scores().numel() / neuron))
        flat_oup = output.flatten(start_dim=1)
        for i in range(neuron):
            flat_oup[i, idx[i, :p]] = 0
            flat_oup[i, idx[i, p:]] = 1
        self.scores = torch.nn.Parameter(output)
        self.scores.requires_grad = False