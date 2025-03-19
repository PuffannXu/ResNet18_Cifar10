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
from torch.optim.lr_scheduler import MultiStepLR


# ======================================== #
# 量化训练参数
# ======================================== #
img_quant_flag = 1
isint = 1
qn_on = 0
left_shift_bit = 0

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

def main():
    os.makedirs("checkpoint",exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print(f"current model name is {model_name}")
    logger = my.setup_logger(name='Logger', log_file=f'logs/{model_name}.log')
    valid_loss_min = np.Inf # track change in validation loss
    accuracy = []
    lr = 0.0000001
    counter = 0
    # Initialize lists to store losses and accuracies
    train_losses = []
    valid_losses = []
    accuracies = []
    lrs = []
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=========================== run on {device} ===========================")

    print(f"        ...... batch size is {batch_size}, loading data ......")
    train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='/home/project/Resnet18/dataset/')
    # 加载模型(使用预处理模型，修改最后一层，固定之前的权重)

    """
    ResNet18网络的7x7降采样卷积和池化操作容易丢失一部分信息,
    所以在实验中我们将7x7的降采样层和最大池化层去掉,替换为一个3x3的降采样卷积,
    同时减小该卷积层的步长和填充大小
    """
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
    model.fc = torch.nn.Linear(512, n_class)  # 将最后的全连接层改掉
    # if (qn_on):
    #     model.fc = my.Linear_quant_noise(qn_on,512, n_class,weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,noise_scale=0)
    # else:
    #     model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层改掉

    model = model.to(device)
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
        x = torch.load(PATH_TO_PTH_CHECKPOINT, map_location=device)
        # del x['conv1.weight']
        model.load_state_dict(x, strict=False)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr*0.1, max_lr=lr, step_size_up=5)
    for epoch in tqdm(range(1, n_epochs+1)):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        total_sample = 0
        right_sample = 0
        # 动态调整学习率
        # if counter/10 ==1:
        #     counter = 0
        #     lr = lr*0.5
        ###################
        # 训练集的模型 #
        ###################
        model.train() #作用是启用batch normalization和drop out
        for data, target in train_loader:
            data = data.to(device)
            if img_quant_flag == 1:
                data, _ = my.data_quantization_sym(data, half_level=127)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)[0].to(device)
            # a = model(data)[1] #（等价于output = model.forward(data).to(device) ）
            # calculate the batch loss（计算损失值）
            _, pred = torch.max(output, 1)
            # print(f"pred{pred},target{target}")
            loss = criterion(output, target)
            # # 计算对称性损失
            # sym_loss = symmetry_loss_model_min_max(model)
            # # 总损失
            # total_loss = loss
            # backward pass: compute gradient of the loss with respect to model parameters
            # （反向传递：计算损失相对于模型参数的梯度）
            try:
                loss.backward()
            except RuntimeError as e:
                if 'nan' in str(e):
                    print("NaN detected in loss, skipping update.")
                    continue

            # ⚠️⚠️️打印梯度信息⚠️⚠️
            def print_grad_stats(model):
                print("\n===== Gradient Check =====")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad = param.grad.data
                        print(f"{name}:")
                        print(f"  Shape: {grad.shape}")
                        print(f"  Min: {grad.min().item():.6f}")
                        print(f"  Mean: {grad.mean().item():.6f}")
                        print(f"  Max: {grad.max().item():.6f}")
                        print(f"  NaN: {torch.isnan(grad).any().item()}")
                        ratio = param.grad.norm() / (param.norm() + 1e-8)
                        print(f"{name}: grad/param = {ratio:.3e}")
                        # 正常范围：1e-5 ~ 1e-2
                    else:
                        print(f"{name}: No gradient")

            # # 在反向传播后添加梯度裁剪
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(),
            #     max_norm=1.0,  # 根据实际情况调整
            #     norm_type=2.0
            # )
            # print_grad_stats(model)  # 首次迭代时打印完整梯度
            # perform a single optimization step (parameter update)
            # 执行单个优化步骤（参数更新）
            optimizer.step()
            # update training loss（更新损失）
            train_loss += loss.item()*data.size(0)

        ######################
        # 验证集的模型#
        ######################

        model.eval()  # 验证模型
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)[0].to(device)
                # convert output probabilities to predicted class(将输出概率转换为预测类)
                _, pred = torch.max(output, 1)
                # print(f"pred{output},target{target}")
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item()*data.size(0)

                # compare predictions to true label(将预测与真实标签进行比较)
                correct_tensor = pred.eq(target.data.view_as(pred))
                # correct = np.squeeze(correct_tensor.to(device).numpy())
                total_sample += batch_size
                for i in correct_tensor:
                    if i:
                        right_sample += 1

        # 在每个 epoch 结束后调用学习率调度器
        scheduler.step()

        # 打印当前学习率
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}, Current Learning Rate: {current_lr}")

        print("Accuracy:",100*right_sample/total_sample,"%")
        accuracy.append(right_sample/total_sample)

        # 计算平均损失
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        # Append losses and accuracy for plotting
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        accuracies.append(right_sample / total_sample)
        lrs.append(lr)
        # 显示训练集与验证集的损失函数
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # 如果验证集损失函数减少，就保存模型。
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(), f'checkpoint/{model_name}.pt')
            valid_loss_min = valid_loss
            counter = 0
        else:
            counter += 1

    # Plotting after training
    plt.figure(figsize=(15, 5))
    fig, (ax1,ax3) = plt.subplots(1, 2)
    # Plotting training and validation losses
    ax1.plot(train_losses, label='Training Loss', color='b')
    ax1.plot(valid_losses, label='Validation Loss', color='g')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss / Learning Rate / Accuracy over Epochs in {model_name}')
    ax1.legend(loc='upper left')

    # Creating a second y-axis for the learning rate
    ax2 = ax1.twinx()
    ax2.plot(lrs, label='Learning Rate', color='r')
    ax2.set_ylabel('Learning Rate')
    ax2.legend(loc='upper right')

    # plt.subplot(1, 2, 2)
    ax3.plot(accuracies, label='Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    # ax3.set_title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    img_quant_flag = 1
    isint = 0
    qn_on = 1
    input_bit = 8
    weight_bit = 8
    output_bit = 8
    left_shift_bit = 0
    n_epochs = 20
    RELOAD_CHECKPOINT = 1
    batch_size = 32
    PATH_TO_PTH_CHECKPOINT = f'checkpoint/ResNet18_fp32_ch{channel_number}_4_newrelu.pt'#checkpoint/ResNet18_fp32_ch{channel_number}_3.pt'
    # PATH_TO_PTH_CHECKPOINT = f'checkpoint/{model_name}.pt'


    fp_on = 0  # 0:off 1:wo hw 2:hw
    quant_type = "layer"  # "layer" "channel" "group"
    group_number = 72
    model_name = f"ResNet18_I8W8_ch{channel_number}_5_newrelu"#f'ResNet18_fp8_hw_{quant_type}{group_number}'
    main()

