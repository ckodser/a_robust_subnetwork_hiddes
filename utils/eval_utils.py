import torch
import numpy as np


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def robustness(output, target, percentile):
    with torch.no_grad():
        percentile = (np.array(percentile) * output.size(0)).astype(np.int)
        confidence, pred = output.topk(2, 1, True, True)
        certified_robustness = confidence[:, 0] - confidence[:, 1]
        certified_robustness, _ = torch.sort(certified_robustness)
        print(certified_robustness)
        print(certified_robustness[percentile])
        return certified_robustness[percentile]
