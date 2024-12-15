import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def main_graph(model_name, quant_type='channel', group_number=4):
    # 加载 .pth 文件
    model_weights = torch.load(f"checkpoint/{model_name}.pt")
    os.makedirs(f"weight_distribution_analysis/{model_name}", exist_ok=True)

    weight_bit = 8
    bin_num = 2 << weight_bit - 1

    # 遍历每一层的权重
    for layer_name, weights in model_weights.items():
        if isinstance(weights, torch.Tensor) and len(weights.shape) >= 2:
            # 转换为 NumPy 数组
            weights_np = weights.cpu().numpy()

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
            plt.axvline(max_val, color='red', linestyle='dashed', linewidth=1)
            plt.axvline(min_val, color='green', linestyle='dashed', linewidth=1)
            text_x = 0.95 * plt.xlim()[1]
            text_y = 0.95 * plt.ylim()[1]
            plt.text(text_x, text_y,
                     f"Max: {max_val:.3f}\nMin: {min_val:.3f}\nMean: {mean_val:.3f}\nStd: {std_val:.3f}",
                     fontsize=10, color='black', ha='right', va='top',
                     bbox=dict(facecolor='white', alpha=0.5))
            plt.grid(True)
            plt.savefig(f'weight_distribution_analysis/{model_name}/LayerDistribution_{model_name}_{layer_name}.png')
            plt.show()

            co, ci, kx, ky = weights_np.shape
            if quant_type == 'channel':
                weight_reshape = weights_np.reshape([co, -1])
                num_channels = co

                # 创建一个子图
                fig, axes = plt.subplots(num_channels, 1, figsize=(10, 5 * num_channels))
                if num_channels == 1:  # 如果只有一个通道，axes 不是数组
                    axes = [axes]

                for i in range(num_channels):
                    channel_weights = weight_reshape[i].flatten()
                    max_val_channel = np.max(channel_weights)
                    min_val_channel = np.min(channel_weights)
                    mean_val_channel = np.mean(channel_weights)
                    std_val_channel = np.std(channel_weights)

                    print(f"Layer: {layer_name}, {quant_type}: {i}")
                    print(f"Max: {max_val_channel}, Min: {min_val_channel}, Mean: {mean_val_channel}, Std: {std_val_channel}")

                    axes[i].hist(channel_weights, bins=bin_num, alpha=0.7, color='green')
                    axes[i].set_title(f'Distribution of weights in layer: {layer_name}, {quant_type}: {i}')
                    axes[i].set_xlabel('Weight Value')
                    axes[i].set_ylabel('Frequency')
                    axes[i].axvline(max_val_channel, color='red', linestyle='dashed', linewidth=1)
                    axes[i].axvline(min_val_channel, color='green', linestyle='dashed', linewidth=1)
                    text_x = 0.95 * axes[i].get_xlim()[1]
                    text_y = 0.95 * axes[i].get_ylim()[1]
                    axes[i].text(text_x, text_y,
                                  f"Max: {max_val_channel:.3f}\nMin: {min_val_channel:.3f}\nMean: {mean_val_channel:.3f}\nStd: {std_val_channel:.3f}",
                                  fontsize=10, color='black', ha='right', va='top',
                                  bbox=dict(facecolor='white', alpha=0.5))
                    axes[i].grid(True)

                plt.tight_layout()
                plt.savefig(f'weight_distribution_analysis/{model_name}/{quant_type}Distribution_{model_name}_{layer_name}.png')
                plt.show()


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
    group_number_list = [9, 72, 288]
    print(f'\n==================== group ====================')
    for group_number in group_number_list:
        print(f'\n==================== group_number is {group_number} ====================')
        name = f"ResNet18_fp8_hw_group{group_number}_epoch30"
        main_graph(name)
    print(f'\n==================== layer ====================')
    quant_type = "layer"
    name = f"ResNet18_fp8_w_hw_layer_epoch30"
    main_graph(name)
    print(f'\n==================== channel ====================')
    quant_type = "channel"
    name = f"ResNet18_fp8_w_hw_channel_epoch30"
    main_graph(name)