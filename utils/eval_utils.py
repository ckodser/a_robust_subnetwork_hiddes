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
        confidence, pred = output.topk(2, 1, True, True)
        certified_robustness = confidence[:, 1] - confidence[:, 0]
        dists = certified_robustness.quantile(np.array(percentile), dim=1)
        print(dists.size())
        print(dists)
        print("confidence size:", confidence.size(), "pred size:", pred.size(), "batch size", target.size(0))
        return 0, 0, 0
