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

# ======================================== #
# 量化训练参数
# ======================================== #
img_quant_flag = 0
isint = 0
qn_on = 0
left_shift_bit = 0

SAVE_TB = False

fp_on = 0  # 0:off 1:wo hw 2:hw
quant_type = "layer"  # "layer" "channel" "group"
group_number = 72
input_bit = 8
weight_bit = 4
output_bit = 8
clamp_std = 0
noise_scale = 0
channel_number = 4
height = 224
width = 224

batch_size = 128
n_class = 10
# 开始训练
n_epochs = 30
RELOAD_CHECKPOINT = 0
PATH_TO_PTH_CHECKPOINT = f'checkpoint/ResNet18_fp32.pt'
# PATH_TO_PTH_CHECKPOINT = f'checkpoint/{model_name}.pt'
model = ResNet18(qn_on = qn_on,
                     fp_on=fp_on,
                     weight_bit=weight_bit,
                     output_bit=output_bit,
                     isint=isint, clamp_std=clamp_std,
                     quant_type=quant_type,
                     group_number=group_number,
                     left_shift_bit=left_shift_bit) # 得到预训练模型
if fp_on == 1:
    model.conv1 = my.Conv2d_fp8(in_channels=channel_number, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
elif fp_on == 2:
    model.conv1 = my.Conv2d_fp8_hw(in_channels=channel_number, out_channels=64, kernel_size=3, stride=1, padding=1,
                               bias=False, quant_type=quant_type, group_number=group_number, left_shift_bit=left_shift_bit)
elif (qn_on):
    model.conv1 = my.Conv2d_quant(qn_on=qn_on, in_channels=channel_number, out_channels=64,
                              kernel_size=3,
                              stride=1, padding=1,
                              bias=False,
                              weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std)
else:
    model.conv1 = nn.Conv2d(in_channels=channel_number, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层改掉
# Initialize weights using fp8e4m3 distribution
# with torch.no_grad():
#     for name, param in model.named_parameters():
#         # param.uniform_(-448, 448)
#         torch.nn.init.normal_(param, mean=0, std=1)
#         # param.data = my.initialize_weights_fp8e4m3(param.data.shape)
#         plt.hist(param.cpu().numpy().flatten(), bins=50)
#         plt.title(f"Distribution of {name}")
#         plt.show()
        # break  # 只显示一个参数的分布
model = model.to(device)

checkpoint = torch.load('/home/project/xupf/Projects/ResNet18_Cifar10/checkpoint/ResNet18_fp32_ch4_3.pt', map_location=device)
model.load_state_dict(checkpoint,strict=True)


# model = torch.load('/home/project/FDheadclass.pt')
# model = torch.load('FDheadclass.pt', map_location='cpu')
x = torch.randn(1, 4, 32, 32).to(device)

torch.onnx.export(model,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  "/home/project/xupf/Projects/ResNet18_Cifar10/ResNet18_fp32_ch4_3.onnx",  # where to save the model
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['modelInput'],  # the model's input names
                  output_names=['modelOutput'],  # the model's output names
                  )