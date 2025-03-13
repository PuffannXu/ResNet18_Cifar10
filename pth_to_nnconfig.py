import os
import pandas as pd
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



# ======================================== #
# 量化训练参数
# ======================================== #
img_quant_flag = 0
isint = 0
qn_on = 0
left_shift_bit = 0

SAVE_TB = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_bit = 8
weight_bit = 4
output_bit = 8
clamp_std = 0
noise_scale = 0
channel_number = 4
height = 224
width = 224
fp_on = 0
quant_type = "layer"  # "layer" "channel" "group"
group_number = 72
batch_size = 128
n_class = 10
# 开始训练
n_epochs = 30
model = ResNet18(qn_on = qn_on,
                     fp_on=fp_on,
                     weight_bit=weight_bit,
                     output_bit=output_bit,
                     isint=isint, clamp_std=clamp_std,
                     quant_type=quant_type,
                     group_number=group_number,
                     left_shift_bit=left_shift_bit) # 得到预训练模型

model.conv1 = nn.Conv2d(in_channels=channel_number, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)



model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层改掉
model = model.to(device)
# 加载模型
x = torch.load("/home/project/xupf/Projects/ResNet18_Cifar10/checkpoint/ResNet18_fp32_ch4_1.pt", map_location='cpu')
model.load_state_dict(x, strict=False)

# 获取层信息（示例输出）
layer_info = []
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d)):
        layer_info.append({
            'name': name,
            'type': type(module).__name__,
            'params': dict(module.named_parameters()),
            'config': {
                'in_channels': getattr(module, 'in_channels', 0),
                'out_channels': getattr(module, 'out_channels', 0),
                'kernel_size': getattr(module, 'kernel_size', (1,1)),
                'stride': getattr(module, 'stride', 1),
                'padding': getattr(module, 'padding', 0),
                'bias': module.bias is not None
            }
        })
    elif isinstance(module, (torch.nn.Linear)):
        layer_info.append({
            'name': name,
            'type': type(module).__name__,
            'params': dict(module.named_parameters()),
            'config': {
                'in_channels': getattr(module, 'in_features', 0),
                'out_channels': getattr(module, 'out_features', 0),
                'kernel_size': getattr(module, 'kernel_size', (1,1)),
                'stride': getattr(module, 'stride', 1),
                'padding': getattr(module, 'padding', 0),
                'bias': module.bias is not None
            }
        })
    elif isinstance(module, (torch.nn.MaxPool2d)):
        layer_info.append({
            'name': name,
            'type': type(module).__name__,
            'params': dict(module.named_parameters()),
            'config': {
                'in_channels': getattr(module, 'in_channels', None),
                'out_channels': getattr(module, 'out_channels', None),
                'kernel_size': getattr(module, 'kernel_size', None),
                'stride': getattr(module, 'stride', None),
                'padding': getattr(module, 'padding', None),
                'bias': 0
            }

        })


def calculate_feature_map_size(input_size, layer_config):
    if layer_config['type'] == 'Conv2d':
        h = (input_size[0] + 2*layer_config['config']['padding'][0] - layer_config['config']['kernel_size'][0]) // layer_config['config']['stride'][0] + 1
        w = (input_size[1] + 2*layer_config['config']['padding'][1] - layer_config['config']['kernel_size'][1]) // layer_config['config']['stride'][1] + 1
        return (h, w)
    elif layer_config['type'] == 'MaxPool2d':
        h = input_size[0] // layer_config['config']['kernel_size'][0]
        w = input_size[1] // layer_config['config']['kernel_size'][1]
        return (h, w)
    else:
        return input_size


table_rows = []
current_size = (32, 32)  # 假设初始输入尺寸为32x32
prev_layer_name = None
for layer in layer_info:
    row = {
        'type':layer['type'],
        'col': current_size[0],
        'row': current_size[1],
        'cin': layer['config']['in_channels'],
        'kh': layer['config']['kernel_size'][0] if isinstance(layer['config']['kernel_size'], tuple) else layer['config']['kernel_size'],
        'kw': layer['config']['kernel_size'][1] if isinstance(layer['config']['kernel_size'], tuple) else layer['config']['kernel_size'],
        'padding':1 if layer['config']['kernel_size'][0] == 3 else 0,
        'stride': layer['config']['stride'][0] if isinstance(layer['config']['stride'], tuple) else layer['config']['stride'],
        'relu': 1,
        # 'add': 1 if layer['name'] in add_layers else 0,
        'cout': layer['config']['out_channels'],
        # 其他字段根据需求补充计算...
    }
    table_rows.append(row)

    # 更新特征图尺寸
    current_size = calculate_feature_map_size(current_size, layer)

df = pd.DataFrame(table_rows)
df.to_excel('network_config.xlsx', index=False)