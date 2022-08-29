from typing import ValuesView
import numpy as np
from pyrsistent import v
import torch
from torch.autograd import Variable

trees = [
[1, 1],
[2, 1],
[3, 1],
[4, 1],
[5, 1],
[6, 2],
[7, 2],
[8, 2],
[9, 2],
[10, 2],
[11, 3],
[12, 3],
[13, 3],
[14, 3],
[15, 3],
[16, 4],
[17, 4],
[18, 4],
[19, 4],
[20, 4],
[21, 5],
[22, 5],
[23, 5],
[24, 5],
[25, 5],
[26, 6],
[27, 6],
[28, 6],
[29, 6],
[30, 6],
[31, 7],
[32, 7],
[33, 7],
[34, 7],
[35, 7],
[36, 1],
[37, 1],
[38, 1],
[39, 1],
[40, 1],
[41, 2],
[42, 2],
[43, 2],
[44, 2],
[45, 2],
[46, 3],
[47, 3],
[48, 3],
[49, 3],
[50, 3],
[51, 4],
[52, 4],
[53, 4],
[54, 4],
[55, 4],
[56, 5],
[57, 5],
[58, 5],
[59, 5],
[60, 5],
[61, 6],
[62, 6],
[63, 6],
[64, 6],
[65, 6],
[66, 7],
[67, 7],
[68, 7],
[69, 7],
[70, 7]
]

cls_num_tuple = [
    [1, 13],
    [2, 27],
    [3, 40],
    [4, 54],
    [5, 67], 
    [6, 12],
    [7, 24],
    [8, 36],
    [9, 48],
    [10, 60],
    [11, 4],
    [12, 8],
    [13, 12],
    [14, 16],
    [15, 20],
    [16, 16],
    [17, 32],
    [18, 48],
    [19, 64],
    [20, 82],
    [21, 20],
    [22, 40],
    [23, 60],
    [24, 80],
    [25, 100],
    [26, 9],
    [27, 18],
    [28, 27],
    [29, 36],
    [30, 41],
    [31, 20],
    [32, 40],
    [33, 60],
    [34, 80],
    [35, 100]
    ]

relabel_cls_num_tuple = [
    [1, 13],
    [1, 13],
    [2, 13],
    [3, 13],
    [5, 13],
    [6, 12],
    [7, 12],
    [8, 12],
    [9, 12],
    [10, 12],
    [11, 4],
    [12, 4],
    [13, 4],
    [14, 4],
    [15, 4],
    [16, 16],
    [17, 16],
    [18, 16],
    [19, 16],
    [20, 16],
    [21, 20],
    [22, 20],
    [23, 20],
    [24, 20],
    [25, 20],
    [26, 9],
    [27, 9],
    [28, 9],
    [29, 9],
    [30, 9],
    [31, 20],
    [32, 20],
    [33, 20],
    [34, 20],
    [35, 20],
]

cls_name_tuple = [
    [1, 'boeing707-320'],
    [2, 'boeing727-200'],
    [3, 'boeing737-200'],
    [4, 'boeing737-300'],
    [5, 'boeing737-400'], 
    [6, 'ak47'],
    [7, 'american flag'],
    [8, 'backpack'],
    [9, 'baseball bat'],
    [10, 'baseball glove'],
    [11, 'flower1'],
    [12, 'flower2'],
    [13, 'flower3'],
    [14, 'flower4'],
    [15, 'flower5'],
    [16, 'airport inside'],
    [17, 'artstudio'],
    [18, 'auditorium'],
    [19, 'bakery'],
    [20, 'bar'],
    [21, 'abyssinian'],
    [22, 'american buldog'],
    [23, 'american pit bull terrier'],
    [24, 'basset hound'],
    [25, 'beagle'],
    [26, 'acura integra type r 2001'],
    [27, 'acura rl sedan 2012'],
    [28, 'acura tl sedan 2012'],
    [29, 'acura tl type s 2008'],
    [30, 'acura tsx sedan 2012'],
    [31, 'applauding'],
    [32, 'blowing bubbles'],
    [33, 'brushing teeth'],
    [34, 'cleaning the floor'],
    [35, 'climbing']
]

def get_parents_target(targets):
    parents_target_list = []

    for i in range(targets.size(0)):
        parents_target_list.append(trees[targets[i]][1]-1)

    parents_target_list = Variable(torch.from_numpy(np.array(parents_target_list)).cuda())   
    
    return parents_target_list

def get_cls_num_list():
    cls_num_list = []

    for i in range(35):
        cls_num_list.append(cls_num_tuple[i][1])
    
    return cls_num_list


def get_relabel_cls_num_list():
    relabel_cls_num_list = []

    for i in range(35):
        relabel_cls_num_list.append(relabel_cls_num_tuple[i][1])

    return relabel_cls_num_list

def get_parents_num_list():
    parents_num_list = []
    for i in range(7):
        parents_num = 0
        for j in range(5):
            idx = i*5 + j 
            parents_num += cls_num_tuple[idx][1]
        parents_num_list[i] = parents_num
    return parents_num_list

def get_cls_name_list():    
    cls_name_list = []

    for i in range(35):
        cls_name_list.append(cls_name_tuple[i][1])
    
    return cls_name_list