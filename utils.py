import os
import torch
import shutil
import logging
import numpy as np
import matplotlib
import torch.nn as nn
import torch.nn.functional as F
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from relabel_data_get_tree_target import *

def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).cuda()
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixed_criterion(pred, y_a, y_b, lam, criterion):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_logging(path_log):
    logger = logging.getLogger('Result_log')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(path_log)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    return logger


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

def calc_confusion_mat(val_loader, model, args):
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    classes = [str(x) for x in args.cls_num_list]
    plot_confusion_matrix(all_targets, all_preds, classes)
    plt.savefig(os.path.join(args.root_log, args.store_name, 'confusion_matrix.png'))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    if normalize:
        norm_confusion = 'true'
    else:
        norm_confusion = None
    cm = confusion_matrix(y_true, y_pred, normalize=norm_confusion)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def prepare_folders(args):
    
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(args, state, is_best):
    
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

# main에서 옮긴 funcitons, classes ----------------------------------------------------------------
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def str2bool(b):
    if b.lower() in ['t', 'true']:
        return True
    return False

class DotProduct_Classifier(nn.Module):
    def __init__(self, model, num_classes):
        super(DotProduct_Classifier, self).__init__()
        # print('<DotProductClassifier> contains bias: {}'.format(bias))
        self.fc = model.linear
        self.scales = Parameter(torch.ones(num_classes).to(self.fc.weight.device))
        for param_name, param in self.fc.named_parameters():
            param.requires_grad = False

    def forward(self, x, *args):
        x = self.fc(x)
        x *= self.scales
        return x

class model_hierarchy(nn.Module):
    def __init__(self, model, parents_num = 7, leaf_num = 35, feature_dim = 512, use_norm = False):
        super(model_hierarchy, self).__init__()

        self.get_features =  nn.Sequential(*list(model.children())[:-1])
        self.parents_classifier = nn.Linear(feature_dim, parents_num)
        self.feature_dim = feature_dim

        if use_norm:
            self.leaf_classifier = NormedLinear(feature_dim//2, leaf_num)
        else:
            self.leaf_classifier = nn.Linear(feature_dim//2, leaf_num)

    def forward(self, x, targets):
        feature = self.get_features(x)
        feature = torch.flatten(feature, 1)
        feature_1 =  feature[:,  0 : self.feature_dim//2]
        feature_2 =  feature[:, self.feature_dim//2 : self.feature_dim]

        paretns_input  = torch.cat([feature_1, feature_2.detach()],1)
        leaf_input = feature_2

        parents_out = self.parents_classifier(paretns_input)
        parents_target = get_parents_target(targets)        
        
        leaf_out = self.leaf_classifier(leaf_input)
        leaf_out = leaf_out[targets <= 34]
        leaf_target = targets[targets <= 34]
        
        return [leaf_out, leaf_target], [parents_out, parents_target]


class model_original(nn.Module):
    def __init__(self, model, classes_num = 35, feature_dim = 512, use_norm = False):
        super(model_original, self).__init__()

        self.get_features =  nn.Sequential(*list(model.children())[:-1])

        if use_norm:
            self.classifier = NormedLinear(feature_dim, classes_num)
        else:
            self.classifier = nn.Linear(feature_dim, classes_num)
        
    def forward(self, x):
        feature = self.get_features(x)
        feature = torch.flatten(feature, 1)
        output = self.classifier(feature)
        
        return output


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out