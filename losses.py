import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class KD(nn.Module):
    def __init__(self, args, model):
        super(KD, self).__init__()
        self.teacher = copy.deepcopy(model)
        self.teacher.load_state_dict(torch.load(args.model_path))
        self.teacher.eval()
        self.lamb = args.lamb
        self.t = 4

    def forward(self, input, model, target):
        with torch.no_grad():
            pred_t = F.softmax(self.teacher(input)/self.t, dim=-1)
        output = model(input)
        pred_s = F.log_softmax(output/self.t, dim=-1)
        loss = F.kl_div(pred_s, pred_t.detach(), reduction='sum') * (self.t**2) / input.shape[0]
        loss += F.cross_entropy(output, target)
        #     pred_t[range(input.size(0)), target] = 0
        #     pred_t = pred_t / pred_t.max(1)[0][:, None]
        #     weight = pred_t * self.lamb
        #
        # output = model(input)

        # weighted_output = output + weight
        return output, loss

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)