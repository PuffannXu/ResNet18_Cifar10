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
fp_on = 1 #0:off 1:wo hw 2:hw
quant_type = "group" #"layer" "channel" "group"
group_number = 1
left_shift_bit = 0

SAVE_TB = False


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
n_epochs = 100
RELOAD_CHECKPOINT = 1
PATH_TO_PTH_CHECKPOINT = f'checkpoint/ResNet18_fp8_wo_bn_w_sym_loss.pt'
# PATH_TO_PTH_CHECKPOINT = f'checkpoint/{model_name}.pt'

def main():
    model_name = "ResNet18_fp8_wo_bn_w_sym_loss"#f'ResNet18_fp8_hw_{quant_type}{group_number}'
    print(f"current model name is {model_name}")
    valid_loss_min = np.Inf # track change in validation loss
    accuracy = []
    lr = 0.001
    counter = 0
    # Initialize lists to store losses and accuracies
    train_losses = []
    valid_losses = []
    accuracies = []
    lrs = []
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=========================== run on {device} ===========================")

    if SAVE_TB:
        # ======================================== #
        # 实例化SummaryWriter对象
        # ======================================== #
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(log_dir='/'.join(["output","tensorboard", model_name]))
    # 读数据

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
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
        model.load_state_dict(torch.load(PATH_TO_PTH_CHECKPOINT, map_location=device))

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
    for epoch in tqdm(range(1, n_epochs+1)):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        total_sample = 0
        right_sample = 0
        # 动态调整学习率
        if counter/10 ==1:
            counter = 0
            lr = lr*0.5
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        ###################
        # 训练集的模型 #
        ###################
        model.train() #作用是启用batch normalization和drop out
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            # clear the gradients of all optimized variables（清除梯度）
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            # (正向传递：通过向模型传递输入来计算预测输出)
            output = model(data)[0].to(device)
            a = model(data)[1] #（等价于output = model.forward(data).to(device) ）
            # calculate the batch loss（计算损失值）
            loss = criterion(output, target)
            # 计算对称性损失
            sym_loss = symmetry_loss_model(model)
            # 总损失
            total_loss = loss + 0.001 * sym_loss  # 0.01 是对称性损失的权重系数
            # backward pass: compute gradient of the loss with respect to model parameters
            # （反向传递：计算损失相对于模型参数的梯度）
            total_loss.backward()
            # perform a single optimization step (parameter update)
            # 执行单个优化步骤（参数更新）
            optimizer.step()
            # update training loss（更新损失）
            train_loss += loss.item()*data.size(0)

        ######################
        # 验证集的模型#
        ######################

        model.eval()  # 验证模型
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)[0].to(device)

            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class(将输出概率转换为预测类)
            _, pred = torch.max(output, 1)
            # compare predictions to true label(将预测与真实标签进行比较)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # correct = np.squeeze(correct_tensor.to(device).numpy())
            total_sample += batch_size
            for i in correct_tensor:
                if i:
                    right_sample += 1
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
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        a_b = a
        a = flatten_dict(a_b)
        tags = ["train_loss", "val_loss", "Accuracy"]
        if SAVE_TB is True:
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], valid_loss, epoch)
            tb_writer.add_scalar(tags[2], accuracy[-1], epoch)

            weights_keys = model.state_dict().keys()
            for key in weights_keys:
                weight_t = model.state_dict()[key].cpu()
                tb_writer.add_histogram(tag=key, values=weight_t, global_step=epoch)
                if weight_t.dim() == 4:
                    c_out, c_in, k_w, k_h = weight_t.shape
                    for o_idx in range(c_out):
                        kernel_idx = weight_t[o_idx, :, :, :].unsqueeze(1)
                        kernel_grid = torchvision.utils.make_grid(kernel_idx, nrow=8, normalize=True, scale_each=True)
                        tb_writer.add_image(f'{key}_split_cin', kernel_grid, global_step=o_idx)
                elif weight_t.dim() == 2:
                    weight_t = weight_t.unsqueeze(0)
                    tb_writer.add_image(f'{key}_split_cin', weight_t)
                elif weight_t.dim() == 1:
                    weight_t = weight_t.unsqueeze(0)
                    weight_t = weight_t.unsqueeze(0)
                    tb_writer.add_image(f'{key}_split_cin', weight_t)
                # kernel_all = weight_t.view(-1,c_in,k_h, k_w)
                # kernel_grid = torchvision.utils.make_grid(kernel_all, nrow=8, normalize=False, scale_each= False)
                # tb_writer.add_image(f'{key}_all', kernel_grid)

            for key in a:
                # im = np.squeeze(a[key].cpu().detach().numpy())
                im = a[key].cpu().detach()

                # [C, H, W] -> [H, W, C]
                # im = np.transpose(im, [1, 2, 0])
                tb_writer.add_histogram(tag=key,
                                        values=im,
                                        global_step=epoch)
                if im.dim() == 4:
                    c_out, c_in, k_w, k_h = im.shape
                    for o_idx in range(c_out):
                        im_idx = im[o_idx, :, :, :].unsqueeze(1)
                        im_grid = torchvision.utils.make_grid(im_idx, nrow=8, normalize=True, scale_each=True)
                        tb_writer.add_image(f'{key}_split_cin', im_grid, global_step=o_idx)
                elif im.dim() == 2:
                    im = im.unsqueeze(0)
                    tb_writer.add_image(f'{key}_split_cin', im)
                elif im.dim() == 1:
                    im = im.unsqueeze(0)
                    im = im.unsqueeze(0)
                    tb_writer.add_image(f'{key}_split_cin', im)


    # Plotting after training
    plt.figure(figsize=(15, 5))
    fig, (ax1,ax3) = plt.subplots(1, 2)
    # Plotting training and validation losses
    ax1.plot(train_losses, label='Training Loss', color='b')
    ax1.plot(valid_losses, label='Validation Loss', color='g')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss and Learning Rate over Epochs')
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
    ax3.set_title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # for group_number in [1,9,18,36,72,144,288,576]:#1,9,
    #     print(f'==================== group_number is {group_number} ====================')
    main()