import torch
import numpy as np
import pandas as pd

group_number_list = [1]#[1, 9, 18, 36, 72, 144]

def main(group_number, writer):
    # Load .pth file
    model_name = f'ResNet18_fp8_hw_group{group_number}.pt'
    model_weights = torch.load(f"checkpoint/ResNet18_fp8_wo_bn.pt.pt")
    weight_bit = 8
    bin_num = 2 << weight_bit - 1

    alpha_q = -(2 << (weight_bit - 1))
    beta_q = (2 << (weight_bit - 1)) - 1

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
    # Create a Pandas Excel writer using openpyxl
    with pd.ExcelWriter('model_statistics.xlsx', engine='openpyxl') as writer:
        for group_number in group_number_list:
            print(f'\n==================== group_number is {group_number} ====================')
            main(group_number, writer)
