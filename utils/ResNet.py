import torch
import torch.nn as nn
from utils.my_utils import Conv2d_fp8, Conv2d_fp8_hw, Conv2d_quant
BN = True

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,qn_on: bool = False,
                 fp_on: int = 0,
                 weight_bit: int = 4,
                 output_bit: int = 8,
                 isint: int = 0,
                 clamp_std: int = 0,
                 quant_type: str ='None',
                 group_number: int =72, left_shift_bit: int = 0):
    """3x3 convolution with padding"""
    if (fp_on == 1):
        return Conv2d_fp8(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                          padding=dilation, groups=groups, bias=False, dilation=dilation)
    elif (fp_on == 2):
        return Conv2d_fp8_hw(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                             padding=dilation, groups=groups, bias=False, dilation=dilation,quant_type=quant_type, group_number=group_number, left_shift_bit=left_shift_bit)
    elif (qn_on):
        return Conv2d_quant(qn_on=qn_on, in_channels=in_planes, out_channels=out_planes,
                                kernel_size=3,
                                stride=stride, padding=dilation,
                                weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                bias=False)
    else:
        return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)




def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 qn_on: bool = False,
                 fp_on: int = 0,
                 weight_bit: int = 4,
                 output_bit: int = 8,
                 isint: int = 0,
                 clamp_std: int = 0,
                 quant_type: str ='None',
                 group_number: int =72,
                 left_shift_bit: int = 0):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes=inplanes, out_planes=planes, stride=stride, qn_on = qn_on,
                 fp_on=fp_on,
                 weight_bit=weight_bit,
                 output_bit=output_bit,
                 isint=isint, clamp_std=clamp_std,
                 quant_type=quant_type,
                 group_number=group_number, left_shift_bit=left_shift_bit)
        if BN:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, qn_on = qn_on,
                 fp_on=fp_on,
                 weight_bit=weight_bit,
                 output_bit=output_bit,
                 isint=isint, clamp_std=clamp_std,
                 quant_type=quant_type,
                 group_number=group_number, left_shift_bit=left_shift_bit)
        if BN:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a = {}

        identity = x
        a['identity'] = x
        out = self.conv1(x)
        a['conv1'] = out
        if BN:
            out = self.bn1(out)
            a['bn1'] = out
        out = self.relu(out)
        a['relu'] = out
        out = self.conv2(out)
        a['conv2'] = out
        if BN:
            out = self.bn2(out)
            a['bn2'] = out
        if self.downsample is not None:
            identity = self.downsample(x)
            a['downsample'] = out
        out = out.clone()  # 克隆 out
        out += identity
        out = self.relu(out)
        a['relu'] = out

        return out,a


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        if BN:
            self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if BN:
            self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        if BN:
            self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if BN:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if BN:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,
                 qn_on: bool = False,
                 fp_on: int = 0,
                 weight_bit: int = 4,
                 output_bit: int = 8,
                 isint: int = 0,
                 clamp_std: int = 0,
                 quant_type: str ='None',
                 group_number: int =72,
                 left_shift_bit: int=3):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if fp_on == 1:
            self.conv1 = Conv2d_fp8(3, self.inplanes, kernel_size=3, stride=2, padding=3,
                               bias=False)
        elif fp_on == 2:
            self.conv1 = Conv2d_fp8_hw(3, self.inplanes, kernel_size=3, stride=2, padding=3,
                               bias=False, quant_type=quant_type, group_number=group_number, left_shift_bit=left_shift_bit)
        elif (qn_on):
            self.conv1 = Conv2d_quant(qn_on=qn_on, in_channels=3, out_channels=self.inplanes,
                                      kernel_size=3,
                                      stride=2, padding=3,
                                      bias=False,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        if BN:
            self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block=block, planes=64, blocks=layers[0],qn_on = qn_on,
                 fp_on=fp_on,
                 weight_bit=weight_bit,
                 output_bit=output_bit,
                 isint=isint, clamp_std=clamp_std,
                 quant_type=quant_type,
                 group_number=group_number,left_shift_bit=left_shift_bit)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],qn_on = qn_on,
                 fp_on=fp_on,
                 weight_bit=weight_bit,
                 output_bit=output_bit,
                 isint=isint, clamp_std=clamp_std,
                 quant_type=quant_type,
                 group_number=group_number,left_shift_bit=left_shift_bit)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],qn_on = qn_on,
                 fp_on=fp_on,
                 weight_bit=weight_bit,
                 output_bit=output_bit,
                 isint=isint, clamp_std=clamp_std,
                 quant_type=quant_type,
                 group_number=group_number,left_shift_bit=left_shift_bit)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],qn_on = qn_on,
                 fp_on=fp_on,
                 weight_bit=weight_bit,
                 output_bit=output_bit,
                 isint=isint, clamp_std=clamp_std,
                 quant_type=quant_type,
                 group_number=group_number,left_shift_bit=left_shift_bit)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if BN and zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                 qn_on: bool = False,
                 fp_on: int = 0,
                 weight_bit: int = 4,
                 output_bit: int = 8,
                 isint: int = 0,
                 clamp_std: int = 0,
                 quant_type: str ='None',
                 group_number: int =72,
                 left_shift_bit: int=3):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,qn_on = qn_on,
                 fp_on=fp_on,
                 weight_bit=weight_bit,
                 output_bit=output_bit,
                 isint=isint, clamp_std=clamp_std,
                 quant_type=quant_type,
                 group_number=group_number,
                left_shift_bit  = left_shift_bit))


        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,qn_on = qn_on,
                 fp_on=fp_on,
                 weight_bit=weight_bit,
                 output_bit=output_bit,
                 isint=isint, clamp_std=clamp_std,
                 quant_type=quant_type,
                 group_number=group_number,
                left_shift_bit=left_shift_bit               ))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        a = {}
        # See note [TorchScript super()]
        a['in'] = x
        x = self.conv1(x)
        a['conv1'] = x
        if BN:
            x = self.bn1(x)
            a['bn1'] = x
        x = self.relu(x)
        a['relu1'] = x
        # x = self.maxpool(x)
        for i in range(0,len(self.layer1)):
            x, a[f'layer1_{i}'] = self.layer1[i](x)
        for i in range(0, len(self.layer2)):
            x, a[f'layer2_{i}'] = self.layer2[i](x)
        for i in range(0, len(self.layer3)):
            x, a[f'layer3_{i}'] = self.layer3[i](x)
        for i in range(0, len(self.layer4)):
            x, a[f'layer4_{i}'] = self.layer4[i](x)
        x = self.avgpool(x)
        a['avgpool'] = x
        x = torch.flatten(x, 1)
        a['flatten'] = x
        x = self.fc(x)
        a['fc'] = x

        return x, a

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def ResNet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2],**kwargs)


def ResNet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3],**kwargs)


def ResNet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3],**kwargs)


def ResNet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3],**kwargs)


def ResNet152(**kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3],**kwargs)
