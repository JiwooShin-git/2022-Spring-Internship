import torchvision.models as models
import torch.nn as nn
import torch
from relabel_data_get_tree_target import *

model = models.resnet50(pretrained=False)

print(*list(model.children())[:])

class model_hierarchy(nn.Module):
    def __init__(self, model, parents_num = 7, leaf_num=35):

        super(model_hierarchy, self).__init__()

        self.get_features =  nn.Sequential(*list(model.children())[:-1])

        self.parents_classifier = nn.Linear(2048, parents_num)

        self. leaf_classifier = nn.Linear(1024, leaf_num)
        
    def forward(self, x, targets):

        feature = self.get_features(x)
        feature = torch.flatten(feature, 1)

        feature_1 =  feature[:,  0:1024]
        feature_2 =  feature[:, 1024:2048]

        paretns_input  = torch.cat([feature_1, feature_2.detach()],1)
        leaf_input = feature_2

        parents_out = self.parents_classifier(paretns_input)
        parents_target = get_parents_target(targets)        
        
        leaf_out = self.leaf_classifier(leaf_input)
        leaf_out = leaf_out[targets <= 34]
        leaf_target = targets[targets <= 34]
        
        return [leaf_out, leaf_target], [parents_out, parents_target]

class model_bn(nn.Module):
    def __init__(self, model, classes_num=35):

        super(model_hierarchy, self).__init__()

        self.get_features =  nn.Sequential(*list(model.children())[:-1])

        self.classifier = nn.Linear(2048, classes_num)
        
    def forward(self, x):

        feature = self.get_features(x)
        feature = torch.flatten(feature, 1)

        output = self.classifier(feature)
        
        return output