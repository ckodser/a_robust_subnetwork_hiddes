import numpy as np
import itertools

__all__ = ["multistep_lr", "cosine_lr", "constant_lr", "get_policy"]

from utils.conv_type import LipschitzSubnetConv


def get_policy(name):
    if name is None:
        return constant_lr

    out_dict = {
        "constant_lr": constant_lr,
        "cosine_lr": cosine_lr,
        "multistep_lr": multistep_lr,
    }

    return out_dict[name]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def multistep_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        lr = args.lr * (args.lr_gamma ** (epoch // args.lr_adjust))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length


def lipschitzSubnetConv_count(model):
    lipschitz_subnet_conv_count = 0
    for layer in itertools.chain(model.module.convs, model.module.linear):
        if isinstance(layer, LipschitzSubnetConv):
            lipschitz_subnet_conv_count += 1
    return lipschitz_subnet_conv_count


def lipschitz_schedulers_linear(args, layers, epoch):
    warm_up = (args.epochs - args.score_initialization_rounds) / 2
    epoch -= args.score_initialization_rounds
    if epoch >= warm_up:
        return 1
    else:
        return ((warm_up - epoch) / warm_up * 5000 + 1) ** (1 / layers)


def lipschitz_schedulers_inverse(args, layers, epoch):
    warm_up = (args.epochs - args.score_initialization_rounds) / 2
    epoch -= args.score_initialization_rounds
    if epoch >= warm_up:
        return 1
    else:
        return (5000 * ((1 / (epoch + 1)) - (1 / warm_up) + (1 / 5000))) ** (1 / layers)


def lipschitz_schedulers_x_to_layers_num(args, epoch):
    warm_up = (args.epochs - args.score_initialization_rounds) / 2
    epoch -= args.score_initialization_rounds
    if epoch >= warm_up:
        return 1
    else:
        return 1 + (warm_up - epoch) / warm_up * 7


def get_lipschitz(args, model, epoch):
    if args.score_initialization_rounds > epoch:
        return 9
    if args.lipschitz_schedulers == "linear":
        return lipschitz_schedulers_linear(args, lipschitzSubnetConv_count(model), epoch)
    elif args.lipschitz_schedulers == "1onx":
        return lipschitz_schedulers_inverse(args, lipschitzSubnetConv_count(model), epoch)
    elif args.lipschitz_schedulers == "xtolayersnum":
        return lipschitz_schedulers_x_to_layers_num(args, epoch)
    else:
        print(" lipschitz schedulers option isn't in the list!")
