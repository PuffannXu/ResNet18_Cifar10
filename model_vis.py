import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.my_utils import fp8_downcast, fp8_alignment, uint8_to_fp32

def main_graph(model_name):
    # 加载 .pth 文件
    # Load .pth file
    # model_name = f'ResNet18_fp8_wo_bn'
    model_weights = torch.load(f"checkpoint/{model_name}.pt")
    os.makedirs(f"weight_distribution_analysis/{model_name}",exist_ok=True)

    weight_bit = 8
    bin_num = 2 << weight_bit - 1

    # 遍历每一层的权重
    for layer_name, weights in model_weights.items():
        if isinstance(weights, torch.Tensor) and len(weights.shape) >= 2:
            # 转换为 NumPy 数组
            weights_np = weights.cpu().numpy()

            # 对称量化
            weight_max = torch.max(torch.abs(weights))
            weight_min = 0
            scaling_factor = weight_max / (1.875 * 2**(n_bits+left_shift_bit))
            weight_scale = weights / scaling_factor
            q_min = 0
            weight_n_scale = fp8_downcast(weight_scale, n_bits=3)
            co, ci, kx, ky = weight_n_scale.shape
            if quant_type == 'channel':
                weight_reshape = weights_np.reshape([co, -1])
                weight_reshape_temp = weight_n_scale.reshape([co, -1])
                weight_align, sign, e_max, m_sft = fp8_alignment(weight_reshape_temp, left_shift_bit)
            elif quant_type == 'layer':
                weight_reshape = weights_np.reshape([1, -1])
                weight_reshape_temp = weight_n_scale.reshape([1, -1])
                weight_align, sign, e_max, m_sft = fp8_alignment(weight_reshape_temp, left_shift_bit)
            elif quant_type == 'group':
                # 计算需要的填充数量
                total_elements = co * ci * kx * ky
                remainder = total_elements % group_number
                if remainder != 0:
                    padding_size = group_number - remainder
                else:
                    padding_size = 0
                # 用零填充张量
                if padding_size > 0:
                    # 创建一个与 weight_n 具有相同维度的 padding
                    padding_shape = list(weight_n_scale.shape)
                    padding_shape[0] = padding_size  # 只在最后一个维度上添加
                    padding = torch.zeros(padding_shape, device=weight_n_scale.device, dtype=weight_n_scale.dtype)
                    weight_n_scale = torch.cat((weight_n_scale, padding))
                    weights_n = weights_np.reshape([-1, 1])
                    padding_shape = list(weights_n.shape)
                    padding_shape[0] = padding_size
                    padding = np.zeros(padding_shape, dtype=weights_np.dtype)
                    weights_np = np.concatenate((weights_n, padding))
                weight_reshape = weights_np.reshape([-1, group_number])
                weight_reshape_temp = weight_n_scale.reshape([-1, group_number])
                weight_align, sign, e_max, m_sft = fp8_alignment(weight_reshape_temp, left_shift_bit)
            else:
                weight_align = weight_n_scale
            weight_align = weight_align.reshape([-1, ci, kx, ky])
            e_max = e_max.reshape([-1, ci, kx, ky])
            m_sft = m_sft.reshape([-1, ci, kx, ky])
            sign = sign.reshape([-1, ci, kx, ky])
            weight_align = weight_align[:co, :ci, :kx, :ky]
            e_max = e_max[:co, :ci, :kx, :ky]
            m_sft = m_sft[:co, :ci, :kx, :ky]
            sign = sign[:co, :ci, :kx, :ky]
            a = weight_scale
            weight_align_fp = uint8_to_fp32(weight_align, sign, e_max, m_sft, n_bits, left_shift_bit=left_shift_bit)
            weight_align_fp_out = (weight_align_fp - q_min) * scaling_factor + weight_min
            weight_align_fp = weight_align_fp.cpu().numpy()
            weight_align_fp_out = weight_align_fp_out.cpu().numpy()

            channel_number = weight_reshape.shape[0]

            # ====================== 绘制第一幅图 ======================
            # 计算统计信息
            max_val = np.max(weights_np)
            min_val = np.min(weights_np)
            mean_val = np.mean(weights_np)
            std_val = np.std(weights_np)

            print(f"Layer: {layer_name}")
            print(f"Max: {max_val}, Min: {min_val}, Mean: {mean_val}, Std: {std_val}")

            # 绘制整体权重直方图
            plt.hist(weights_np.flatten(), bins=bin_num, alpha=0.7, color='blue')
            plt.title(f'Overall Distribution of weights in {model_name}')
            plt.xlabel('Weight Value')
            plt.ylabel(f'Layer: {layer_name}')
            # 添加最大值和最小值的垂直线
            plt.axvline(max_val, color='red', linestyle='dashed', linewidth=1)
            plt.axvline(min_val, color='green', linestyle='dashed', linewidth=1)
            # 调整文字的位置和样式
            text_x = 0.95 * plt.xlim()[1]  # 设置文本的 x 坐标靠近图表的右边
            text_y = 0.95 * plt.ylim()[1]  # 设置文本的 y 坐标靠近图表的上边
            plt.text(text_x, text_y,
                     f"Max: {max_val:.3f}\nMin: {min_val:.3f}\nMean: {mean_val:.3f}\nStd: {std_val:.3f}",
                     fontsize=10, color='black', ha='right', va='top',
                     bbox=dict(facecolor='white', alpha=0.5))
            plt.grid(True)
            plt.savefig(f'weight_distribution_analysis/{model_name}/LayerDistribution_{model_name}_{layer_name}.png')
            plt.show()

            if quant_type == 'channel':
                weight_reshape = weights_np.reshape([co, -1])
                weight_align_fp = weight_align_fp.reshape([co, -1])
                weight_align_fp_out = weight_align_fp_out.reshape([co, -1])
            elif quant_type == 'layer':
                weight_reshape = weights_np.reshape([1, -1])
                weight_align_fp = weight_align_fp.reshape([1, -1])
                weight_align_fp_out = weight_align_fp_out.reshape([1, -1])
            elif quant_type == 'group':
                # 计算需要的填充数量
                total_elements = co * ci * kx * ky
                remainder = total_elements % group_number
                if remainder != 0:
                    padding_size = group_number - remainder
                else:
                    padding_size = 0
                # 用零填充张量
                if padding_size > 0:
                    # 创建一个与 weight_n 具有相同维度的 padding
                    weights_n = weights_np.reshape([-1, 1])
                    weight_align_fp = weight_align_fp.reshape([-1, 1])
                    weight_align_fp_out = weight_align_fp_out.reshape([-1, 1])
                    padding_shape = list(weights_n.shape)
                    padding_shape[0] = padding_size
                    padding = np.zeros(padding_shape, dtype=weights_np.dtype)
                    weights_np = np.concatenate((weights_n, padding))
                    weight_align_fp = np.concatenate((weight_align_fp, padding))
                    weight_align_fp_out = np.concatenate((weight_align_fp_out, padding))
                weight_reshape = weights_np.reshape([-1, group_number])
                weight_align_fp = weight_align_fp.reshape([-1, group_number])
                weight_align_fp_out = weight_align_fp_out.reshape([-1, group_number])
            # ====================== 绘制第二幅图 ======================
            for i in range(channel_number):
                channel_weights = weight_reshape[i].flatten()
                channel_weights_fp_temp = weight_align_fp[i].flatten()
                channel_weights_fp = weight_align_fp_out[i].flatten()
                # 计算每个通道的统计信息
                max_val_channel = np.max(channel_weights)
                min_val_channel = np.min(channel_weights)
                mean_val_channel = np.mean(channel_weights)
                std_val_channel = np.std(channel_weights)

                print(f"Layer: {layer_name}, {quant_type}: {i}")
                print(f"Max: {max_val_channel}, Min: {min_val_channel}, Mean: {mean_val_channel}, Std: {std_val_channel}")

                # 创建一个 3x1 的子图
                fig, axs = plt.subplots(3, 1, figsize=(10, 15))

                # 绘制权重直方图
                axs[0].hist(channel_weights, bins=bin_num, alpha=0.7, color='green')
                axs[0].set_title(f'Distribution of weights in layer: {layer_name}, {quant_type}: {i}')
                axs[0].set_xlabel('Weight Value')
                axs[0].set_ylabel('Frequency')
                axs[0].axvline(0, color='red', linestyle='dashed', linewidth=1)

                # 添加统计信息文本
                text_x = 0.95 * axs[0].get_xlim()[1]
                text_y = 0.95 * axs[0].get_ylim()[1]
                axs[0].text(text_x, text_y,
                            f"Max: {max_val_channel:.3f}\nMin: {min_val_channel:.3f}\nMean: {mean_val_channel:.3f}\nStd: {std_val_channel:.3f}",
                            fontsize=10, color='black', ha='right', va='top',
                            bbox=dict(facecolor='white', alpha=0.5))
                axs[0].grid(True)



                # 绘制 weight_align_fp
                axs[1].hist(channel_weights_fp_temp, bins=bin_num, alpha=0.7, color='blue')
                axs[1].set_title('Weight Align FP')
                axs[1].set_xlabel('FP Value')
                axs[1].set_ylabel('Frequency')
                axs[1].grid(True)

                # 绘制 weight_align_fp_out
                axs[2].hist(channel_weights_fp, bins=bin_num, alpha=0.7, color='orange')
                axs[2].set_title('Weight Align FP Out')
                axs[2].set_xlabel('FP Out Value')
                axs[2].set_ylabel('Frequency')
                axs[2].grid(True)

                # # 获取所有图的 y 轴范围
                # y_min = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0], axs[2].get_ylim()[0])
                # y_max = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1], axs[2].get_ylim()[1])
                #
                # # 设置所有子图的 y 轴范围
                # for ax in axs:
                #     ax.set_xlim(y_min, y_max)

                # 调整布局
                plt.tight_layout()
                plt.savefig(f'weight_distribution_analysis/{model_name}/{quant_type}Distribution_{model_name}_{layer_name}_{quant_type}{i}.png')
                plt.show()



            # # 如果是卷积层或线性层，按输出通道绘制
            # if len(weights_np.shape) >= 2:
            #     num_output_channels = weights_np.shape[0]
            #     for i in range(num_output_channels):
            #         channel_weights = weights_np[i].flatten()
            #
            #         # 计算每个通道的统计信息
            #         max_val_channel = np.max(channel_weights)
            #         min_val_channel = np.min(channel_weights)
            #         mean_val_channel = np.mean(channel_weights)
            #         std_val_channel = np.std(channel_weights)
            #
            #         print(f"Layer: {layer_name}, Channel: {i}")
            #         print(f"Max: {max_val_channel}, Min: {min_val_channel}, Mean: {mean_val_channel}, Std: {std_val_channel}")
            #
            #         # 绘制每个通道的权重直方图
            #         plt.figure(figsize=(10, 5))
            #         plt.hist(channel_weights, bins=bin_num, alpha=0.7, color='green')
            #         plt.title(f'Distribution of weights in layer: {layer_name}, Channel: {i}')
            #         plt.xlabel('Weight Value')
            #         plt.ylabel('Frequency')
            #         plt.grid(True)
            #         plt.show()

def main(group_number):
    # Load .pth file
    model_name = f'ResNet18_fp8_wo_bn'
    model_weights = torch.load(f"checkpoint/{model_name}.pt")
    weight_bit = 8
    bin_num = 2 << weight_bit - 1

    alpha_q = -(2 << (weight_bit - 1))
    beta_q = (2 << (weight_bit - 1)) - 1
    with pd.ExcelWriter(f'weight_distribution_analysis/{model_name}/model_statistics_.xlsx', engine='openpyxl') as writer:
        # Iterate over each layer's weights
        for layer_name, weights in model_weights.items():
            if isinstance(weights, torch.Tensor) and "conv" in layer_name:
                # Convert to NumPy array
                weights_np = weights.cpu().numpy()

                # Calculate statistics
                max_val = np.max(weights_np)
                min_val = np.min(weights_np)
                mean_val = np.mean(weights_np)
                std_val = np.std(weights_np)

                print(f"Layer: {layer_name}")
                print(f"Max: {max_val}, Min: {min_val}, Mean: {mean_val}, Std: {std_val}")

                # Store statistics in a DataFrame
                df = pd.DataFrame({
                    'Group Number': [group_number],
                    'Max': [max_val],
                    'Min': [min_val],
                    'Mean': [mean_val],
                    'Std': [std_val]
                })

                # Write to Excel, each layer in a different sheet
                sheet_name = layer_name.replace('.', '_')
                if sheet_name in writer.sheets:
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=writer.sheets[sheet_name].max_row)
                else:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == '__main__':
    # main_graph("ResNet18_fp8_hw_group9_epoch30")
    quant_type = "group"
    group_number = 72
    left_shift_bit = 0
    n_bits = 3
    group_number_list = [72, 288]
    print(f'\n==================== group ====================')
    for group_number in group_number_list:
        print(f'\n==================== group_number is {group_number} ====================')
        name = f"ResNet18_fp8_w_hw_group{group_number}_wo_be_epoch30"
        main_graph(name)
    print(f'\n==================== layer ====================')
    quant_type = "layer"
    name = f"ResNet18_fp8_w_hw_layer_epoch30"
    main_graph(name)
    print(f'\n==================== channel ====================')
    quant_type = "channel"
    name = f"ResNet18_fp8_w_hw_channel_epoch30"
    main_graph(name)