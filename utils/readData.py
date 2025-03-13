import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from utils.cutout import Cutout
# 全局取消证书验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# number of subprocesses to use for data loading
num_workers = 0
# 每批加载图数量
batch_size = 16
# percentage of training set to use as validation
valid_size = 0.2
class AddChannel(object):
    """将图像的通道数从3增加到4。"""

    def __call__(self, img):
        # 在通道维度增加一个全零通道
        zero_channel = torch.zeros((1, img.size(1), img.size(2)))  # 创建一个全零通道
        img = torch.cat((img, zero_channel), dim=0)  # 在通道维度拼接
        return img
def read_dataset(batch_size=16,valid_size=0.2,num_workers=0,pic_path='dataset'):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #R,G,B每层的归一化用到的均值和方差
        AddChannel(),  # 添加新的通道
        # Cutout(n_holes=1, length=16),
    ])

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        AddChannel(),  # 添加新的通道
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        AddChannel(),  # 添加新的通道
    ])
    # 将数据转换为torch.FloatTensor，并标准化。
    train_data = datasets.CIFAR10(pic_path, train=True,
                                download=True, transform=transform_train)
    valid_data = datasets.CIFAR10(pic_path, train=True,
                                download=True, transform=transform_valid)
    test_data = datasets.CIFAR10(pic_path, train=True,
                                download=True, transform=transform_test)
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
        num_workers=num_workers)

    return train_loader,valid_loader,test_loader