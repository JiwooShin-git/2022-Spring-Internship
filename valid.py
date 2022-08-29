from utils import *
from relabel_data_get_tree_target import *

def validate_hierarchy(val_loader, model):
    
    leaf_top1 = AverageMeter('Acc@1', ':6.2f')
    parents_top1 = AverageMeter('Acc@1', ':6.2f')

    leaf_top5 = AverageMeter('Acc@5', ':6.2f')
    parents_top5 = AverageMeter('Acc@1', ':6.2f')

    model.eval()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()
            # input, target = Variable(input), Variable(target)
        
            [leaf_out, leaf_target], [parents_out, parents_target] = model(input, target)

            # measure accuracy and record loss
            if len(leaf_target) >= 1:
                leaf_acc1, leaf_acc5 = accuracy(leaf_out, leaf_target, topk=(1, 5))
                leaf_top1.update(leaf_acc1, input.size(0))
                leaf_top5.update(leaf_acc5, input.size(0))
            
            if len(parents_target) >= 1:
                parent_acc1, parent_acc5 = accuracy(parents_out, parents_target, topk=(1, 5))
                parents_top1.update(parent_acc1, input.size(0))
                parents_top5.update(parent_acc5, input.size(0))            
            
    return parents_top1.avg, leaf_top1.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if args.loss_type == 'KD':
                output, loss = criterion(input, model, target)
            else:
                output = model(input)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1, input.size(0))
            top5.update(acc5, input.size(0))

    return top1.avg