from utils import *
from relabel_data_get_tree_target import *
import torch

def train_hierarchy(train_loader, model, criterion, optimizer):
    losses = AverageMeter('Loss', ':.4e')
    
    leaf_top1 = AverageMeter('Acc@1', ':6.2f')
    parents_top1 = AverageMeter('Acc@1', ':6.2f')

    leaf_top5 = AverageMeter('Acc@5', ':6.2f')
    parents_top5 = AverageMeter('Acc@1', ':6.2f')

    model.train()

    ce_criterion = nn.CrossEntropyLoss().cuda()
        
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()

        [leaf_out, leaf_target], [parents_out, parents_target] = model(input, target)
        
        parents_loss = ce_criterion(parents_out, parents_target)
        leaf_loss = criterion(leaf_out, leaf_target)
        loss = parents_loss + leaf_loss
        # print(leaf_loss)
        # print(parents_loss)

        # autograd.set_detect_anomaly(True)
        if len(leaf_target) >= 1:
            leaf_acc1, leaf_acc5 = accuracy(leaf_out, leaf_target, topk=(1, 5))
            leaf_top1.update(leaf_acc1, input.size(0))
            leaf_top5.update(leaf_acc5, input.size(0))

        if len(parents_target) >= 1:        
            parent_acc1, parent_acc5 = accuracy(parents_out, parents_target, topk=(1, 5))
            parents_top1.update(parent_acc1, input.size(0))
            parents_top5.update(parent_acc5, input.size(0))

        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=3)
        optimizer.step()

    return losses.avg, parents_top1.avg, leaf_top1.avg

def train(train_loader, model, criterion, optimizer, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        input, target = input.cuda(), target.cuda()

        # compute output
        if args.loss_type == 'KD':
            output, loss = criterion(input, model, target)
        else:
            if args.mixup_alpha > 0:
                mixed_inputs, targets_1, targets_2, lam = mixup_data(input, target, args.mixup_alpha)
                output = model(mixed_inputs)
                loss = mixed_criterion(output, targets_1, targets_2, lam, criterion)
            else:
                output = model(input)
                loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=3)
        optimizer.step()

    return losses.avg, top1.avg