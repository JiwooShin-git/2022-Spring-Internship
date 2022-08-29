import argparse
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from train import train, train_hierarchy
from valid import validate, validate_hierarchy
from utils import *
from relabel_data_get_tree_target import *
from losses import LDAMLoss, FocalLoss, KD


parser = argparse.ArgumentParser(description='Classs imbalance - using hierarchy')
parser.add_argument('--arch', default='resnet18')
parser.add_argument('--data', default='data_imbal')
parser.add_argument('--loss_type', default='LDAM', type=str, help='loss type')
parser.add_argument('--train_rule', default='DRW', type=str, help='data sampling strategy for train loader')
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--is_plot', default='f', type=str2bool)
parser.add_argument('--use_hierarchy', default='f', type=str2bool)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.1, type=float) # 0.1에서 0.001로 바꿈 -> 안 그러면 nan 발생 (0.1로도 한 번 해보자!)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=2e-4, type=float)
parser.add_argument('--mixup_alpha', default=0, type=float)
parser.add_argument('--is_retrain', default='f', type=str2bool)
# if tau>=0인 경우에만 tau-normalization 진행 (*is_retrain이 false이면 실행x)
parser.add_argument('--tau', default=-1, type=float)
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
args = parser.parse_args()


def main(args):
    cudnn.benchmark = True
    num_classes = 35

    if args.arch == 'resnet18':
        model = models.resnet18(pretrained=False)
        feature_dim = 512
    elif args.arch == 'resnet34':
        model = models.resnet34(pretrained=False)
        feature_dim = 512
    elif args.arch == 'resnet50':
        model = models.resnet50(pretrained=False)
        feature_dim = 2048
    
    use_norm = True if args.loss_type == 'LDAM' else False

    if args.use_hierarchy:
        model = model_hierarchy(model, feature_dim = feature_dim, use_norm=use_norm)
    else:
        model = model_original(model, feature_dim = feature_dim, use_norm=use_norm)

    model.cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    args_path = f'{args.data}_{args.loss_type}_{args.lr}_{args.arch}_{args.train_rule}_{args.use_hierarchy}'
    args.path_log = os.path.join('logs', args_path)
    if not os.path.exists(args.path_log):
        os.makedirs(args.path_log)

    if args.is_plot:
        if not os.path.exists(os.path.join(args.path_log, 'train_plot')):
            os.makedirs(os.path.join(args.path_log, 'train_plot'))
            os.makedirs(os.path.join(args.path_log, 'valid_plot'))
    logger = create_logging(os.path.join(args.path_log, '%s.txt' % args.rand_number))

    for param in sorted(vars(args).keys()):
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    train_path = args.data + '/train'
    val_path = args.data + '/valid'
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
    val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=transform_test)

    if args.data == 'data_imbal':
        cls_num_list = get_cls_num_list()
    elif args.data == 'data_relabel':
        cls_num_list = get_relabel_cls_num_list()

    # init log for training
    best_acc1 = 0
    if not args.is_retrain:
        for epoch in range(args.epochs):
            s = time.time()
            adjust_learning_rate(optimizer, epoch, args)

            if args.train_rule == 'None':
                per_cls_weights = None
            elif args.train_rule == 'Reweight':
                beta = 0.9999
                effective_num = 1.0 - np.power(beta, cls_num_list)
                per_cls_weights = (1.0 - beta) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            elif args.train_rule == 'DRW':
                idx = epoch // 160
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            else:
                warnings.warn('Sample rule is not listed')

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last = False)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, drop_last = False)

            if args.loss_type == 'CE':
                criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda()
            elif args.loss_type == 'LDAM':
                criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda()
            elif args.loss_type == 'Focal':
                criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda()
            elif args.loss_type == 'KD':
                criterion = KD(args, model).cuda()
            else:
                warnings.warn('Loss type is not listed')
                return

            if args.use_hierarchy:
                train_loss, parents_train_acc, leaf_train_acc = train_hierarchy(train_loader, model, criterion, optimizer)
                parents_test_acc, leaf_test_acc = validate_hierarchy(val_loader, model)

                # remember best acc@1 and save checkpoint
                best_acc1 = max(leaf_test_acc, best_acc1)
                logger.info(f'Epoch: {epoch + 1}| Loss: {train_loss:2.4f}| Train Parents Acc: {parents_train_acc:.4f}| Train Leaf Acc: {leaf_train_acc:.4f}'
                            f'| Test Parents Acc: {parents_test_acc:.4f}| Test Leaf Acc: {leaf_test_acc:.4f}| Best Leaf Acc: {best_acc1:.4f}| Time: {time.time() - s:3.1f}')
                
            else:
                train_loss, train_acc = train(train_loader, model, criterion, optimizer, args)
                test_acc = validate(val_loader, model, criterion, args)

                # remember best acc@1 and save checkpoint
                best_acc1 = max(test_acc, best_acc1)
                logger.info(f'Epoch: {epoch + 1}| Loss: {train_loss:2.4f}| Train Acc: {train_acc:.4f}| Test Acc: {test_acc:.4f}'
                            f'| Best Acc: {best_acc1:.4f}| Time: {time.time() - s:3.1f}')

        torch.save(model.state_dict(), os.path.join(args.path_log, 'model.pth'))


    if args.is_retrain:
        model.load_state_dict(torch.load(os.path.join(args.path_log, 'model.pth')))
        # tau-normalization
        if args.tau >= 0:
            with torch.no_grad():
                # given code
                criterion = nn.CrossEntropyLoss().cuda()
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last = False)

                for i in range(11):
                    tau = round(i * 0.1, 1)

                    if args.use_hierarchy:
                        w = model.linear_leaf.weight
                        normB = torch.norm(w, 2, 1)
                        ws = w.clone()
                        for i in range(w.size(0)):
                            ws[i] = ws[i] / torch.pow(normB[i], tau)
                        model.linear_leaf.weight.data = ws
                        s = time.time()

                        parents_test_acc, leaf_test_acc, cls_acc = validate_hierarchy(val_loader, model, args, 1)
                        logger.info(
                            f'tau: {tau}| Test Parents Acc: {parents_test_acc:.4f}| Test Leaf Acc: {leaf_test_acc:.4f}'
                            f'| Cls Acc: {cls_acc:.4f}| Time: {time.time() - s:3.1f}')

                    else:
                        w = model.linear.weight
                        normB = torch.norm(w, 2, 1)
                        ws = w.clone()
                        for i in range(w.size(0)):
                            ws[i] = ws[i] / torch.pow(normB[i], tau)
                        model.linear.weight.data = ws
                        s = time.time()

                        test_acc, cls_acc = validate(val_loader, model, criterion, args, 1)
                        logger.info(
                            f'tau: {tau}| Test Acc: {test_acc:.4f}'
                            f'| Cls Acc: {cls_acc:.4f}| Time: {time.time() - s:3.1f}')
                    
        # cRT
        else:
            train_epoch = 10
            if args.use_hierarchy:
                model.linear_leaf = DotProduct_Classifier(model, num_classes)
            else:
                model.linear = DotProduct_Classifier(model, num_classes)
            
            for n, p in model.feat.named_parameters():
                p.requires_grad = False

            optimizer = torch.optim.SGD(model.linear.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epoch, 0.0)
            criterion = nn.CrossEntropyLoss().cuda()

            for epoch in range(train_epoch):
                s = time.time()
                scheduler.step()

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last = False)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last = False)

                if args.use_hierarchy:
                    train_loss, parents_train_acc, leaf_train_acc = train_hierarchy(train_loader, model, optimizer, args, epoch)
                    parents_test_acc, leaf_test_acc, cls_acc = validate_hierarchy(val_loader, model, args, epoch)

                    # remember best acc@1 and save checkpoint
                    best_acc1 = max(leaf_test_acc, best_acc1)
                    logger.info(f'Epoch: {epoch + 1}| Loss: {train_loss:2.4f}| Train Parents Acc: {parents_train_acc:.4f}| Train Leaf Acc: {leaf_train_acc:.4f}'
                                f'| Test Parents Acc: {parents_test_acc:.4f}| Test Leaf Acc: {leaf_test_acc:.4f}| Best Leaf Acc: {best_acc1:.4f}| Cls Acc: {cls_acc:.4f}| Time: {time.time() - s:3.1f}')
                    
                else:
                    train_loss, train_acc = train(train_loader, model, criterion, optimizer, args, epoch)
                    test_acc, cls_acc = validate(val_loader, model, criterion, args, epoch)

                    # remember best acc@1 and save checkpoint
                    best_acc1 = max(test_acc, best_acc1)
                    logger.info(f'Epoch: {epoch + 1}| Loss: {train_loss:2.4f}| Train Acc: {train_acc:.4f}| Test Acc: {test_acc:.4f}'
                                f'| Best Acc: {best_acc1:.4f}| Cls Acc: {cls_acc:.4f}| Time: {time.time() - s:3.1f}')

        # 동일한 model에 대해서 반복할 수 없으므로 잠깐 꺼둔다.
        # torch.save(model.state_dict(), os.path.join(args.path_log, 'model.pth'))

if __name__ == '__main__':
    main(args)