import os
import time
import numpy as np
import torch
from sklearn.metrics import f1_score


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

        return correct * 100 / batch_size


def compute_batch_micro_f1(output, target):
    """Computes the mico F1 for a batch"""
    with torch.no_grad():
        _, pred = output.max(1)
        micro_f1 = f1_score(target.cpu(), pred.cpu(), average='micro')
        return micro_f1


def compute_batch_accuracy_sigmoid(output, target):
    """Computes the accuracy for a batch using sigmoid activation."""
    with torch.no_grad():
        pred = torch.sigmoid(output)
        pred = (pred > 0.5)
        correct = float(pred.eq(target).sum().item())

        return correct / len(target)


def compute_batch_micro_f1_sigmoid(output, target):
    """Computes the micro F1 for a batch using sigmoid activation."""
    with torch.no_grad():
        pred = torch.sigmoid(output)
        pred = (pred > 0.5)
        micro_f1 = f1_score(target.cpu(), pred.cpu(), average='micro')
        return micro_f1


def train(model, device, data_loader, criterion, optimizer, mask_type):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    f1 = AverageMeter()
    model.train()
    end = time.time()
    for i, data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        data = data.to(device)
        optimizer.zero_grad()
        if mask_type != 'PPI':
            mask = data.train_mask
            output = model(data)
            target = data.y[mask]
            output = output[mask]
            func_accuracy = compute_batch_accuracy
            func_micro_f1 = compute_batch_micro_f1
        else:
            output = model(data)
            target = data.y
            func_accuracy = compute_batch_accuracy_sigmoid
            func_micro_f1 = compute_batch_micro_f1_sigmoid

        loss = criterion(output, target)
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), target.size(0))
        accuracy.update(func_accuracy(output, target), target.size(0))
        f1.update(func_micro_f1(output, target), target.size(0))
    return losses.avg, accuracy.avg, f1.avg


def evaluate(model, device, data_loader, criterion, mask_type):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    f1 = AverageMeter()
    results = []
    model.eval()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            end = time.time()
            data = data.to(device)
            if mask_type != 'PPI':
                if mask_type == 'val':
                    mask = data.val_mask
                elif mask_type == 'test':
                    mask = data.test_mask
                output = model(data)
                target = data.y[mask]
                output = output[mask]
                func_accuracy = compute_batch_accuracy
                func_micro_f1 = compute_batch_micro_f1
            else:
                output = model(data)
                target = data.y
                func_accuracy = compute_batch_accuracy_sigmoid
                func_micro_f1 = compute_batch_micro_f1_sigmoid

            loss = criterion(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            accuracy.update(func_accuracy(output, target), target.size(0))
            f1.update(func_micro_f1(output, target), target.size(0))

            y_true = target.detach().to('cpu').numpy().tolist()
            y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
            results.extend(list(zip(y_true, y_pred)))

    return losses.avg, accuracy.avg, f1.avg, results
