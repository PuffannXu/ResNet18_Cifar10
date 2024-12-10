"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2024/12/10 9:12
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""
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
import time

def initialize_model(qn_on=0, fp_on=0, weight_bit=0, output_bit=0, isint=0, clamp_std=0, quant_type="none", group_number=0, left_shift_bit=0, n_class=10, device='cuda'):
    start_time = time.time()
    print("---- in function【initialize_model】")
    model = ResNet18(qn_on=qn_on,
                     fp_on=fp_on,
                     weight_bit=weight_bit,
                     output_bit=output_bit,
                     isint=isint, clamp_std=clamp_std,
                     quant_type=quant_type,
                     group_number=group_number,
                     left_shift_bit=left_shift_bit)
    if fp_on == 1:
        model.conv1 = my.Conv2d_fp8(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    elif fp_on == 2:
        model.conv1 = my.Conv2d_fp8_hw(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1,
                                       bias=False, quant_type=quant_type, group_number=group_number, left_shift_bit=left_shift_bit)
    elif qn_on:
        model.conv1 = my.Conv2d_quant(qn_on=qn_on, in_channels=3, out_channels=64,
                                      kernel_size=3,
                                      stride=1, padding=1,
                                      bias=False,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std)
    else:
        model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, n_class)  # 将最后的全连接层改掉
    model = model.to(device)
    end_time = time.time()
    #print(f"---- exit function【initialize_model】. Using time: {end_time-start_time}s.")
    print()
    return model

def train_and_evaluate(model, train_loader, valid_loader, criterion, my_loss, n_epochs, device, model_name, SAVE_TB, PATH_TO_PTH_CHECKPOINT, RELOAD_CHECKPOINT, lr):
    start_time = time.time()
    print("---- in function【train_and_evaluate】")
    valid_loss_min = np.Inf  # track change in validation loss
    accuracy = []
    counter = 0
    train_losses = []
    valid_losses = []
    accuracies = []
    lrs = []

    if SAVE_TB:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(log_dir='/'.join(["output", "tensorboard", model_name]))

    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
        model.load_state_dict(torch.load(PATH_TO_PTH_CHECKPOINT, map_location=device),strict=False)
        t1 = time.time()
        #print(f"---- in function【train_and_evaluate】: RELOAD_CHECKPOINT done. Using time: {t1-start_time}s.")
    else:
        t1 = start_time

    for epoch in tqdm(range(1, n_epochs + 1)):
        train_loss = 0.0
        valid_loss = 0.0
        total_sample = 0
        right_sample = 0
        if counter / 10 == 1:
            counter = 0
            lr = lr * 0.5
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        model.train()
        PRINT_FLAG = 1
        for data, target in train_loader:
            if PRINT_FLAG:
                t2 = time.time()
                #print(f"---- in function【train_and_evaluate】: train_loader done. Using time: {t2 - t1}s.")
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)[0].to(device)
            a = model(data)[1]
            loss = criterion(output, target)
            total_loss = loss + 0.001 * my_loss(model)
            total_loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            if PRINT_FLAG:
                t3 = time.time()
                #print(f"---- in function【train_and_evaluate】: optimizer done. Using time: {t3 - t2}s.")
                PRINT_FLAG = 0
        PRINT_FLAG = 1
        t3 = time.time()
        #print(f"---- in function【train_and_evaluate】: train_loader ALL done. Using time: {t3 - t1}s.")
        model.eval()
        for data, target in valid_loader:
            if PRINT_FLAG:
                t4 = time.time()
                #print(f"---- in function【train_and_evaluate】: train_loader done. Using time: {t4 - t3}s.")
            data = data.to(device)
            target = target.to(device)
            output = model(data)[0].to(device)
            loss = criterion(output, target) + 0.001 * my_loss(model)
            valid_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            total_sample += data.size(0)
            right_sample += correct_tensor.sum().item()
            t5 = time.time()
            if PRINT_FLAG:
                #print(f"---- in function【train_and_evaluate】: train_loader done. Using time: {t5 - t4}s.")
                print()
                PRINT_FLAG = 0
        print("Accuracy:", 100 * right_sample / total_sample, "%")
        accuracy.append(right_sample / total_sample)
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        accuracies.append(right_sample / total_sample)
        lrs.append(lr)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), f'checkpoint/{model_name}.pt')
            valid_loss_min = valid_loss
            counter = 0
        else:
            counter += 1

        if SAVE_TB:
            tb_writer.add_scalar("train_loss", train_loss, epoch)
            tb_writer.add_scalar("val_loss", valid_loss, epoch)
            tb_writer.add_scalar("Accuracy", accuracy[-1], epoch)
            for key, value in model.state_dict().items():
                tb_writer.add_histogram(tag=key, values=value.cpu(), global_step=epoch)

    plt.figure(figsize=(15, 5))
    fig, (ax1, ax3) = plt.subplots(1, 2)
    ax1.plot(train_losses, label='Training Loss', color='b')
    ax1.plot(valid_losses, label='Validation Loss', color='g')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss and Learning Rate over Epochs')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(lrs, label='Learning Rate', color='r')
    ax2.set_ylabel('Learning Rate')
    ax2.legend(loc='upper right')
    ax3.plot(accuracies, label='Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()
