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


def robustness(output, target, perturbation, lipschitz):
    with torch.no_grad():
        res = []
        batch_size = target.size(0)
        second_confidence, _ = output.topk(2, 1, True, True)
        second_confidence = second_confidence[:, 1]
        target = torch.reshape(target, (-1, 1))
        target_class_confidence = (output.gather(dim=1, index=target)).squeeze()
        bound = target_class_confidence - second_confidence
        for eps in perturbation:
            res.append(((bound > 2 * eps * lipschitz).sum()) * 100 / batch_size)
        return res
