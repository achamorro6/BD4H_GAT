import os
import time
import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_batch_accuracy(output, target):
    """Computes the accuracy for a batch"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum()

        return correct * 100.0 / batch_size


def compute_batch_micro_f1(output, target):
    """Computes the mico F1 for a batch"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.max(1)

        true_positive = (pred * target).sum()
        false_positive = (pred * (1 - target)).sum()
        false_negative = ((1 - pred) * target).sum()
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        micro_f1 = (2 * precision * recall) / (precision + recall)

        return micro_f1 * 100.0 / batch_size


def train(model, device, data, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    f1 = AverageMeter()

    model.train()
    end = time.time()
    # measure data loading time
    data_time.update(time.time() - end)

    data = data.to(device)
    optimizer.zero_grad()
    output = model(data)
    mask = data.train_mask
    target = data.y[mask]
    output = output[mask]
    loss = criterion(output, target)
    assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    losses.update(loss.item(), target.size(0))
    accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))
    f1.update(compute_batch_micro_f1(output, target).item(), target.size(0))

    return losses.avg, accuracy.avg, f1.avg


def evaluate(model, device, data, criterion, mask):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    f1 = AverageMeter()
    results = []
    model.eval()
    with torch.no_grad():
        end = time.time()
        data = data.to(device)
        output = model(data)
        target = data.y[mask]
        output = output[mask]
        loss = criterion(output, target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), target.size(0))

        accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))
        f1.update(compute_batch_micro_f1(output, target).item(), target.size(0))

        y_true = target.detach().to('cpu').numpy().tolist()
        y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
        results.extend(list(zip(y_true, y_pred)))

    return losses.avg, accuracy.avg, f1.avg, results
