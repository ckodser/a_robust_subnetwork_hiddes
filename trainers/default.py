import time
import torch
import tqdm

from utils.eval_utils import accuracy, robustness
from utils.logging import AverageMeter, ProgressMeter
from utils.conv_type import LipschitzSubnetConv, GetLipschitzSubnet
from utils.schedulers import get_lipschitz

import itertools

__all__ = ["train", "validate", "modifier"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
            enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    # if for each test we calculate max(class)-max2(class) we get a number which we want to increase
    # q1_dist print first quarter value and se on.
    q1_255_dist = AverageMeter(" 1/255 perturbation", ":6.2f", write_val=False)
    q8_255_dist = AverageMeter(" 8/255 perturbation", ":6.2f", write_val=False)
    q1_dist = AverageMeter(" 0.1 perturbation", ":6.2f", write_val=False)
    q2_dist = AverageMeter(" 0.2 perturbation", ":6.2f", write_val=False)
    q3_dist = AverageMeter(" 0.3 perturbation", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, q1_255_dist, q8_255_dist, q1_dist, q2_dist, q3_dist],
        prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    model_lipschitz = 1
    for layer in itertools.chain(model.module.convs, model.module.linear):
        if isinstance(layer, LipschitzSubnetConv):
            model_lipschitz *= layer.lipschitz

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
                enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            q1_255, q8_255, q1, q2, q3 = robustness(output, target, perturbation=(1 / 255, 8 / 255, 0.1, 0.2, 0.3),
                                                    lipschitz=model_lipschitz)
            q1_255_dist.update(q1_255, images.size(0))
            q8_255_dist.update(q8_255, images.size(0))
            q1_dist.update(q1, images.size(0))
            q2_dist.update(q2, images.size(0))
            q3_dist.update(q3, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg


def modifier(args, epoch, model):
    if args.conv_type == "LipschitzSubnetConv":
        model_lipschitz = 1

        count = 0
        for layer in itertools.chain(model.module.convs, model.module.linear):
            if isinstance(layer, LipschitzSubnetConv):
                count += 1
                lipschitz = get_lipschitz(args, layer, epoch)
                layer.lipschitz = lipschitz
                model_lipschitz *= layer.lipschitz
                s = layer.scores.data
                w = layer.weight.data
                s = torch.abs(torch.flatten(s))
                w = torch.abs(torch.flatten(w))
                similarity = torch.dot(s, w) / torch.sqrt(torch.dot(s, s) * torch.dot(w, w))

                subnet = GetLipschitzSubnet.apply(layer.clamped_scores, layer.prune_rate, layer.weight, lipschitz)
                w = layer.weight * subnet

                print("score/weight similarity of layer ", count, similarity.cpu().numpy(), " sum weight:",
                      torch.sum(torch.abs(layer.weight.data[0])).cpu().numpy(), " lipschitz= ", lipschitz,
                      "lipschitz check= ", torch.sum(torch.abs(w.detach()[0])).cpu().numpy())
        print("lipschitz of whole model:", model_lipschitz)
