import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
quant_type = "group"
group_number = 72
group_number_list = [9, 18, 36, 72, 144]
def main_graph(model_name):
    # 加载 .pth 文件
    # Load .pth file
    # model_name = f'ResNet18_fp8_wo_bn'
    model_weights = torch.load(f"checkpoint/{model_name}.pt")

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
            plt.savefig(f'LayerDistribution_{model_name}_{layer_name}.png')
            plt.show()


            co, ci, kx, ky = weights_np.shape
            if quant_type == 'channel':
                weight_reshape = weights_np.reshape([co, -1])
            elif quant_type == 'layer':
                weight_reshape = weights_np.reshape([1, -1])
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
                    padding_shape = list(weights_n.shape)
                    padding_shape[0] = padding_size
                    padding = np.zeros(padding_shape, dtype=weights_np.dtype)
                    weights_np = np.concatenate((weights_n, padding))
                weight_reshape = weights_np.reshape([-1, group_number])
                for i in range(4):
                    channel_weights = weight_reshape[i].flatten()
                    # 计算每个通道的统计信息
                    max_val_channel = np.max(channel_weights)
                    min_val_channel = np.min(channel_weights)
                    mean_val_channel = np.mean(channel_weights)
                    std_val_channel = np.std(channel_weights)

                    print(f"Layer: {layer_name}, {quant_type}: {i}")
                    print(f"Max: {max_val_channel}, Min: {min_val_channel}, Mean: {mean_val_channel}, Std: {std_val_channel}")

                    # 绘制每个通道的权重直方图
                    plt.figure(figsize=(10, 5))
                    plt.hist(channel_weights, bins=bin_num, alpha=0.7, color='green')
                    plt.title(f'Distribution of weights in layer: {layer_name}, {quant_type}: {i}')
                    plt.xlabel('Weight Value')
                    plt.ylabel('Frequency')
                    # 添加最大值和最小值的垂直线
                    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
                    # plt.axvline(min_val_channel, color='green', linestyle='dashed', linewidth=1)
                    # 调整文字的位置和样式
                    text_x = 0.95 * plt.xlim()[1]  # 设置文本的 x 坐标靠近图表的右边
                    text_y = 0.95 * plt.ylim()[1]  # 设置文本的 y 坐标靠近图表的上边
                    plt.text(text_x, text_y,
                             f"Max: {max_val_channel:.3f}\nMin: {min_val_channel:.3f}\nMean: {mean_val_channel:.3f}\nStd: {std_val_channel:.3f}",
                             fontsize=10, color='black', ha='right', va='top',
                             bbox=dict(facecolor='white', alpha=0.5))
                    plt.grid(True)
                    plt.savefig(f'{quant_type}Distribution_{model_name}_{layer_name}_{quant_type}{i}.png')
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
    with pd.ExcelWriter(f'model_statistics_.xlsx', engine='openpyxl') as writer:
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
    main_graph("ResNet18_fp8_w_bn_w_sym_loss_min_max")

    # for group_number in group_number_list:
    #
    #     print(f'\n==================== group_number is {group_number} ====================')
    #     main_graph(group_number)
