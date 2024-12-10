"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2024/12/10 9:16
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""
import torch
import numpy as np
from utils.common_train import initialize_model, train_and_evaluate
from utils.readData import read_dataset
# FP8 w bn wo hw
# 参数设置
img_quant_flag = 0
isint = 0
qn_on = 0
fp_on = 1
quant_type = "group"
group_number = 0
left_shift_bit = 0
input_bit = 8
weight_bit = 4
output_bit = 8
clamp_std = 0
noise_scale = 0

SAVE_TB = False

batch_size = 128
lr = 0.001
n_class = 200
n_epochs = 100

RELOAD_CHECKPOINT = 1
PATH_TO_PTH_CHECKPOINT = f'checkpoint/ResNet18_fp32_w_bn_w_sym_loss_imagenet.pt'

# 自定义的对称性损失函数
def symmetry_loss(weights):
    mean_value = torch.mean(weights)
    return torch.abs(mean_value)

def symmetry_loss_model(model):
    loss = 0.0
    for param in model.parameters():
        if param.requires_grad:
            mean = torch.mean(param)
            loss += torch.abs(mean)
    return loss
def main():
    model_name = f"ResNet18_fp8_w_bn_wo_hw_w_sym_loss_imagenet"
    print(f"current model name is {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=========================== run on {device} ===========================")
    train_loader, valid_loader, _ = read_dataset(batch_size=batch_size, pic_path='/home/project/xupf/Databases/tiny-imagenet-200', dataset="IMAGENET")
    model = initialize_model(qn_on=qn_on, fp_on=fp_on, weight_bit=weight_bit, output_bit=output_bit,
                             isint=isint, clamp_std=clamp_std, quant_type=quant_type, group_number=group_number,
                             left_shift_bit=left_shift_bit, n_class=n_class, device=device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    train_and_evaluate(model, train_loader, valid_loader, criterion, symmetry_loss_model, n_epochs, device, model_name, SAVE_TB,
                       PATH_TO_PTH_CHECKPOINT, RELOAD_CHECKPOINT, lr)


if __name__ == '__main__':
    print(f'\n==================== FP8 w bn wo hw ====================')
    main()
