"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2025/3/11 16:01
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""
import os

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.readData import read_dataset
from utils.ResNet import ResNet18
from torch.utils.tensorboard import SummaryWriter
import torchvision
import utils.my_utils as my
import matplotlib.pyplot as plt  # Import matplotlib
a = np.load("a.npy",allow_pickle=True).item()
list=[]
def flatten_dict(nested_dict, parent_key='', sep='_'):
    flattened_dict = {}
    for k, v in nested_dict.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            flattened_dict.update(flatten_dict(v, new_key, sep))
        else:
            flattened_dict[new_key] = v
    return flattened_dict

flattened_dict = flatten_dict(a)
weights = torch.load("/home/project/xupf/Projects/ResNet18_Cifar10/checkpoint/ResNet18_I8W8_ch4_4_newrelu.pt")

# 2. 创建一个字典来存储转换后的权重
quantized_weights_list = []
# 3. 遍历权重字典
for key, weight in weights.items():
    # 4. 将权重转换为 NumPy 数组
    weight_np = weight.detach().cpu().numpy()
    if weight.ndim == 4:
        cout, cin, kernel_height, kernel_width = weight_np.shape

        # 6. 转换为二维形式 KxN
        K = cin * kernel_height * kernel_width  # K = cin * kernel_height * kernel_width
        N = cout  # N = cout

        # 创建二维数组
        weights_2d = weight_np.transpose(0, 3, 2, 1).reshape(N, K)
    else:
        weights_2d = weight_np
    if weights_2d.ndim == 2:
        weights_2d = weights_2d.transpose(1,0)
    quantized_weights, scale = my.data_quantization_sym(weights_2d,half_level=127,isint=1,boundary_refine = 1)

    quantized_weights_list.append(quantized_weights)
for x in flattened_dict.values():
    x = x * 127
    x = x.round()
    quantized_weights = x.cpu().detach().numpy()
    if quantized_weights.ndim == 4:
        quantized_weights = quantized_weights.transpose(0,1,3,2)
    list.append(quantized_weights)

print()