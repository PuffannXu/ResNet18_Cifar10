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
import os
# FP32
# 参数设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
SAVE_TB = False
lr = 0.1
batch_size = 32
# Imagenet
n_class = 200
n_epochs = 200

RELOAD_CHECKPOINT = 0
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
    model_name = f"ResNet18_fp32_w_bn_w_sym_loss_imagenet_1"
    print(f"current model name is {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=========================== run on {device} ===========================")
    train_loader, valid_loader, _ = read_dataset(batch_size=batch_size, pic_path='/home/project/xupf/Databases/tiny-imagenet-200', dataset="IMAGENET", num_workers=4)
    model = initialize_model(n_class=n_class, device=device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    train_and_evaluate(model, train_loader, valid_loader, criterion, symmetry_loss_model, n_epochs, device, model_name, SAVE_TB, PATH_TO_PTH_CHECKPOINT, RELOAD_CHECKPOINT, lr)

if __name__ == '__main__':
    print(f'\n==================== FP32 ====================')
    main()
