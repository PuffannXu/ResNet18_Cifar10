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

class TinyImageNetDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        # 定义类名
        self.classes = self._load_classes()

        if self.split == 'train':
            self.data_dir = os.path.join(root, 'train')
            self.images = []
            self.labels = []
            for idx, class_dir in enumerate(self.classes):
                class_images = os.listdir(os.path.join(self.data_dir, class_dir, 'images'))
                self.images.extend([os.path.join(self.data_dir, class_dir, 'images', img) for img in class_images])
                self.labels.extend([idx] * len(class_images))
        elif self.split == 'val':
            self.data_dir = os.path.join(root, 'val')
            self.images = [os.path.join(self.data_dir, 'images', img) for img in os.listdir(os.path.join(self.data_dir, 'images'))]
            self.labels = []
            with open(os.path.join(self.data_dir, 'val_annotations.txt'), 'r') as f:
                val_annotations = f.readlines()
                val_dict = {line.split('\t')[0]: line.split('\t')[1] for line in val_annotations}
                self.labels = [self.classes.index(val_dict[os.path.basename(img)]) for img in self.images]
        elif self.split == 'test':
            self.data_dir = os.path.join(root, 'test')
            self.images = [os.path.join(self.data_dir, 'images', img) for img in os.listdir(os.path.join(self.data_dir, 'images'))]
            self.labels = [-1] * len(self.images)  # Test labels are not provided

    def _load_classes(self):
        with open(os.path.join(self.root, 'wnids.txt'), 'r') as f:
            classes = f.read().splitlines()
        return classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self.images)}")
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def read_dataset(batch_size=16,valid_size=0.2,num_workers=0,pic_path='dataset',dataset='CIFAR10'):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        # transforms.RandomCrop(224, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #R,G,B每层的归一化用到的均值和方差
        Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    if dataset == "CIFAR10":
        # 将数据转换为torch.FloatTensor，并标准化。
        train_data = datasets.CIFAR10(pic_path, train=True,
                                    download=True, transform=transform_train)
        valid_data = datasets.CIFAR10(pic_path, train=True,
                                    download=True, transform=transform_test)
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
    else:
        # 加载训练集
        train_data = TinyImageNetDataset(pic_path, split='train', transform=transform_train)
        valid_data = TinyImageNetDataset(pic_path, split='val', transform=transform_test)
        test_data = TinyImageNetDataset(pic_path, split='test', transform=transform_test)
        # # obtain training indices that will be used for validation
        # num_train = len(train_data)
        # indices = list(range(num_train))
        # # random indices
        # np.random.shuffle(indices)
        # # the ratio of split
        # split = int(np.floor(valid_size * num_train))
        # # divide data to radin_data and valid_data
        # train_idx, valid_idx = indices[split:], indices[:split]
        #
        # # define samplers for obtaining training and validation batches
        # # 无放回地按照给定的索引列表采样样本元素
        # train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)
        #
        # # prepare data loaders (combine dataset and sampler)
        # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        #                                            sampler=train_sampler, num_workers=num_workers)
        # valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
        #                                            sampler=valid_sampler, num_workers=num_workers)
        # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
        #                                           num_workers=num_workers)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader,valid_loader,test_loader