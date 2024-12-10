import torch
import torch.nn as nn
from utils.readData import read_dataset
from utils.ResNet import ResNet18
from tqdm import tqdm
from utils import my_utils as my
import os
import time
import matplotlib.pyplot as plt
import numpy as np
# ======================================== #
# 量化训练参数
# ======================================== #


# 设置日志
def main():
    model_name = f"ResNet18_fp8_w_bn_w_sym_loss_min_max"
    # Where to save the generated visualizations
    if fp_on == 0:
        PATH_TO_SAVED = os.path.join(
            f"output/test/{model_name}_qn{qn_on}isint{isint}I{input_bit}W{weight_bit}_{str(time.strftime('%Y%m%d_%H%M', time.localtime(time.time())))}")
    elif fp_on == 1:
        PATH_TO_SAVED = os.path.join(f"output/test/{model_name}_fp8_wo_hw_{str(time.strftime('%Y%m%d_%H%M', time.localtime(time.time())))}")
    elif fp_on == 2:
        if quant_type == "group":
            PATH_TO_SAVED = os.path.join(
                f"output/test/{model_name}_fp8_w_hw_{quant_type}{group_number}_{str(time.strftime('%Y%m%d_%H%M', time.localtime(time.time())))}")
        else:
            PATH_TO_SAVED = os.path.join(
                f"output/test/{model_name}_fp8_w_hw_{quant_type}_{str(time.strftime('%Y%m%d_%H%M', time.localtime(time.time())))}")
    os.makedirs(PATH_TO_SAVED, exist_ok=True)
    logger = my.setup_logger(name='FP8Logger', log_file=f'{PATH_TO_SAVED}/vis.log')

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_class = 10
    batch_size = 100
    train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='/home/project/Resnet18/dataset/')
    model = ResNet18(qn_on = qn_on,
                     fp_on=fp_on,
                     weight_bit=weight_bit,
                     output_bit=output_bit,
                     isint=isint, clamp_std=clamp_std,
                     quant_type=quant_type,
                     group_number=group_number, left_shift_bit=left_shift_bit) # 得到预训练模型
    if fp_on == 1:
        model.conv1 = my.Conv2d_fp8(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    elif fp_on == 2:
        model.conv1 = my.Conv2d_fp8_hw(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1,
                                   bias=False, quant_type=quant_type, group_number=group_number, left_shift_bit=left_shift_bit)
    elif (qn_on):
        model.conv1 = my.Conv2d_quant(qn_on=qn_on, in_channels=3, out_channels=64,
                                  kernel_size=3,
                                  stride=1, padding=1,
                                  bias=False,
                                  weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std)
    else:
        model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层改掉

    # 载入权重
    model.load_state_dict(torch.load(f'checkpoint/{model_name}.pt'), strict=False)
    model = model.to(device)

    total_sample = 0
    right_sample = 0
    model.eval()  # 验证模型
    for j, (data, target) in enumerate(tqdm(test_loader, desc="Processing rows")):
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)[0].to(device)
        a = model(data)[1]  # （等价于output = model.forward(data).to(device) ）
        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)

        # compare predictions to true label(将预测与真实标签进行比较)
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1
        # print(f"num is {j}")
    acc = 100*right_sample/total_sample
    print(f"Accuracy: {acc:.3f}%")


    model_weights = torch.load(f"checkpoint/{model_name}.pt")
    bin_num = 2 << 8 - 1

    # 遍历每一层的权重
    # for layer_name, weights in model_weights.items():
    #     if isinstance(weights, torch.Tensor):
    #         # 转换为 NumPy 数组
    #         weights_np = weights.cpu().numpy()
    #
    #         # 计算统计信息
    #         max_val = np.max(weights_np)
    #         min_val = np.min(weights_np)
    #         mean_val = np.mean(weights_np)
    #         std_val = np.std(weights_np)
    #
    #         print(f"Layer: {layer_name}")
    #         print(f"Max: {max_val}, Min: {min_val}, Mean: {mean_val}, Std: {std_val}")
    #
    #         # 绘制整体权重直方图
    #         plt.figure(figsize=(10, 5))
    #         plt.hist(weights_np.flatten(), bins=bin_num, alpha=0.7, color='blue')
    #         plt.title(f'Overall Distribution of weights in {model_name} ({acc:.3f}')
    #         plt.xlabel('Weight Value')
    #         plt.ylabel('Layer: {layer_name}')
    #         plt.text(f"Max: {max_val:.3f}, Min: {min_val:.3f}, Mean: {mean_val:.3f}, Std: {std_val:.3f}")
    #         plt.grid(True)
    #         plt.show()
    #
    #         # # 如果是卷积层或线性层，按输出通道绘制
    #         # if len(weights_np.shape) >= 2:
    #         #     num_output_channels = weights_np.shape[0]
    #         #     for i in range(num_output_channels):
    #         #         channel_weights = weights_np[i].flatten()
    #         #
    #         #         # 计算每个通道的统计信息
    #         #         max_val_channel = np.max(channel_weights)
    #         #         min_val_channel = np.min(channel_weights)
    #         #         mean_val_channel = np.mean(channel_weights)
    #         #         std_val_channel = np.std(channel_weights)
    #         #
    #         #         print(f"Layer: {layer_name}, Channel: {i}")
    #         #         print(f"Max: {max_val_channel}, Min: {min_val_channel}, Mean: {mean_val_channel}, Std: {std_val_channel}")
    #         #
    #         #         # 绘制每个通道的权重直方图
    #         #         plt.figure(figsize=(10, 5))
    #         #         plt.hist(channel_weights, bins=bin_num, alpha=0.7, color='green')
    #         #         plt.title(f'Distribution of weights in layer: {layer_name}, Channel: {i}')
    #         #         plt.xlabel('Weight Value')
    #         #         plt.ylabel('Frequency')
    #         #         plt.grid(True)
    #         #         plt.show()

if __name__ == '__main__':
    # ==========================全精度==============================
    fp_on = 2  # 0:off 1:wo hw 2:hw
    quant_type = "group"  # "layer" "channel" "group"
    group_number = 1
    group_number_list = [9, 18, 36, 72, 144, 288, 576]
    left_shift_bit = 0
    isint = 0
    qn_on = 0
    img_quant_flag = qn_on
    input_bit, weight_bit, output_bit = 8, 4, 8
    clamp_std, noise_scale = 0, 0
    channel_number, height, width = 4, 224, 224
    # print(f'\n==================== left_shift_bit is {left_shift_bit} ====================')
    # main()
    # left_shift_bit = 0
    # print(f'\n==================== left_shift_bit is {left_shift_bit} ====================')
    # main()

    # # ==============================INT:I8W8O8============================
    # isint = 0
    # qn_on = 1
    # img_quant_flag = qn_on
    # input_bit, weight_bit, output_bit = 8, 8, 8
    # clamp_std, noise_scale = 0, 0
    # channel_number, height, width = 4, 224, 224
    # print(f'\n==================== qn_on is {qn_on} ====================')
    # print(f'\n==== input_bit, weight_bit, output_bit is {input_bit, weight_bit, output_bit} ====================')
    # main()
    # # ==============================INT:I8W4O8==========================
    # input_bit, weight_bit, output_bit = 8, 4, 8
    # print(f'\n==================== qn_on is {qn_on} ====================')
    # print(f'\n==== input_bit, weight_bit, output_bit is {input_bit, weight_bit, output_bit} ====================')
    # main()
    # # ==============================INT:I4W8O8==========================
    # input_bit, weight_bit, output_bit = 4, 8, 4
    # print(f'\n==================== qn_on is {qn_on} ====================')
    # print(f'\n==== input_bit, weight_bit, output_bit is {input_bit, weight_bit, output_bit} ====================')
    # main()
    # # ==============================INT:I4W4O4========================
    # input_bit, weight_bit, output_bit = 4, 4, 4
    # print(f'\n==================== qn_on is {qn_on} ====================')
    # print(f'\n==== input_bit, weight_bit, output_bit is {input_bit, weight_bit, output_bit} ====================')
    # main()
    # # ===============================fp8===========================
    # fp_on = 1
    # qn_on = 0
    # print(f'\n==================== fp_on is {fp_on} ====================')
    # main()
    # # ===============================fp8硬件参数=================================
    # fp_on = 2
    # print(f'\n==================== fp_on is {fp_on} ====================')
    for group_number in group_number_list:
        print(f'\n==================== group_number is {group_number} ====================')
        main()