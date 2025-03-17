"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2024/11/27 15:25
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""
import numpy as np
from math import log
import os
import struct
import torch
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: kidwz
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from torch.nn import init
from torch.nn.parameter import Parameter



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_quantization(data_float, half_level=15, scale=None,
                          isint=0, clamp_std=0, boundary_refine=True,
                          reg_shift_mode=False, reg_shift_bits=None):
    alpha_q = - (half_level + 1)
    beta_q = half_level
    # 计算统计信息
    max_val = data_float.max()
    min_val = data_float.min()
    scale = (max_val - min_val) / (beta_q - alpha_q)
    zero_bias = beta_q - max_val / scale
    data_quantized = (data_float / scale + zero_bias).round()
    quant_scale = 1 / scale
    return data_quantized, quant_scale

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def floor_pass(x):
    y = torch.floor(x)
    y_grad = x
    return (y - y_grad).detach() + y_grad


def data_quantization_sym(data_float, half_level=15, scale=None,
                          isint=0, clamp_std=0, boundary_refine=False,
                          reg_shift_mode=False, reg_shift_bits=None):
    # isint = 1 -> return quantized values as integer levels
    # isint = 0 -> return quantized values as float numbers with the same range as input
    # reg_shift_mode -> force half_level to be exponent of 2, i.e., half_level = 2^n (n is integer)

    # if torch.any(torch.isnan(data_float)):
    #     print("nan")
    if half_level <= 0:
        return data_float, 1

    if boundary_refine:
        half_level += 0.4999

    if clamp_std:
        std = data_float.std()
        data_float[data_float < (clamp_std * -std)] = (clamp_std * -std)
        data_float[data_float > (clamp_std * std)] = (clamp_std * std)

    if scale == None or scale == 0:
        scale = data_float.abs().max().detach()  # 动态计算但不参与梯度

    if scale == 0:
        return data_float, 1
    shift_bits = 0
    if reg_shift_mode:
        if reg_shift_bits != None:
            quant_scale = 2 ** reg_shift_bits
        else:
            shift_bits = np.floor(math.log(1 / scale * half_level, 2))
            quant_scale = 2 ** shift_bits
        if isinstance(data_float, torch.Tensor):
            data_quantized = floor_pass((data_float * quant_scale))
        else:
            data_quantized = np.floor((data_float * quant_scale))
        # print(f'quant_scale = {quant_scale}')
        # print(f'reg_shift_bits = {shift_bits}')
    else:
        data_quantized = round_pass(data_float / scale * half_level)
        quant_scale = 1 / scale * half_level
        shift_bits = 0

    if isint == 0:
        data_quantized = data_quantized * scale / half_level
        quant_scale =  half_level / scale

    return data_quantized, shift_bits


# Add noise to input data
def add_noise(weight, method='add', n_scale=0.074, n_range='max'):
    # weight -> input data, usually a weight
    # method ->
    #   'add' -> add a Gaussian noise to the weight, preferred method
    #   'mul' -> multiply a noise factor to the weight, rarely used
    # n_scale -> noise factor
    # n_range ->
    #   'max' -> use maximum range of weight times the n_scale as the noise std, preferred method
    #   'std' -> use weight std times the n_scale as the noise std, rarely used
    std = weight.std()
    w_range = weight.max() - weight.min()

    if n_range == 'max':
        factor = w_range
    if n_range == 'std':
        factor = std

    if method == 'add':
        w_noise = factor * n_scale * torch.randn_like(weight)
        weight_noise = weight + w_noise
    if method == 'mul':
        w_noise = torch.randn_like(weight) * n_scale + 1
        weight_noise = weight * w_noise
    return weight_noise


# ================================== #
# Autograd Functions
# ================================== #
class Round_Grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.round()
        # ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# ================================== #
# Quant Functions
# ================================== #
class Weight_Quant(torch.autograd.Function):
    # Number of inputs (excluding ctx, only weight, bias, half_level, isint, clamp_std, noise_scale)
    # for forward need to be the same as the number of return in def backward()
    # (return weight_grad, bias_grad, None, None, None, None)
    @staticmethod
    def forward(ctx, weight, bias, half_level, isint, clamp_std):
        # weight -> input weight
        # bias -> input bias
        # half_level -> quantization level
        # isint -> return int (will result in scaling) or float (same scale)
        # clamp_std -> clamp weight to [- std * clamp_std, std * clamp_std]
        # noise_scale -> noise scale, equantion can be found in add_noise()
        ctx.save_for_backward()

        std = weight.std()
        if clamp_std != 0:
            weight = torch.clamp(weight, min=-clamp_std * std, max=clamp_std * std)

        # log down the max scale for input weight
        scale_in = abs(weight).max()

        # log down the max scale for input weight
        weight, scale = data_quantization_sym(weight, half_level, scale=scale_in,
                                              isint=isint, clamp_std=0)

        # No need for bias quantization, since the bias is added to the feature map on CPU (or GPU)
        if bias != None:
            # bias = bias / scale
            bias, _ = data_quantization_sym(bias, 127,
                                            isint=isint, clamp_std=0)
            # bias = add_noise(bias, n_scale=noise_scale)

        return weight, bias

    # Use default gradiant to train the network
    # Number of inputs (excluding ctx, only weight_grad, bias_grad) for backward need to be the same as the
    # number of return in def forward() (return weight, bias)
    @staticmethod
    def backward(ctx, weight_grad, bias_grad):
        return weight_grad, bias_grad, None, None, None, None

class Feature_Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feature, half_level, isint, scale):
        # feature_q, _ = data_quantization_sym(feature, half_level, scale = None, isint = isint, clamp_std = 0)
        feature_q, shift_bits= data_quantization_sym(feature, half_level, isint=isint, scale=scale)
        # print(shift_bits)
        return feature_q

    @staticmethod
    def backward(ctx, feature_grad):
        return feature_grad, None, None



# A convolution layer which adds noise and quantize the weight and output feature map
class Conv2d_quant(nn.Conv2d):
    def __init__(self,
                 qn_on,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 weight_bit,
                 output_bit,
                 isint,
                 clamp_std,
                 bias,
                 ):
        # weight_bit -> bit level for weight
        # output_bit -> bit level for output feature map
        # isint, clamp_std, noise_scale -> same arguments as Weight_Quant_Noise()
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         bias=bias
                         )
        self.qn_on = qn_on
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.output_half_level = 2 ** output_bit / 2 - 1
        self.isint = isint
        self.clamp_std = clamp_std

    def forward(self, x):
        # quantize weight and add noise first
        if self.qn_on:
            # 先对输入进行量化，放缩到 127 大小
            x = x * 127.0
            # 然后进行round取整
            x = round_pass(x)
            # x = x.round()
            # 对权重进行量化，用整型表示
            weight_q, scale = data_quantization_sym(self.weight, self.weight_half_level, isint=1, clamp_std=self.clamp_std)
            # 正确做法：量化后的值用浮点表示
            weight_q = weight_q.to(torch.float32)  # 强制保留浮点类型
            # cout, cin, kernel_height, kernel_width = weight_q.shape
            # 6. 转换为二维形式 KxN
            # K = cin * kernel_height * kernel_width  # K = cin * kernel_height * kernel_width
            # N = cout  # N = cout
            # weights_2d = weight_q.permute(0, 3, 2, 1).reshape(N, K).transpose(1,0).cpu().detach().numpy()
            # np.savetxt(f'weight{weight_q.shape}_{weight_q[0][0][0][0]}.csv', weights_2d, delimiter=',')
            if self.bias != None:
                # bias = bias / scale
                bias_q, _ = data_quantization_sym(self.bias, 127, isint=1, clamp_std=0)
            else:
                bias_q = self.bias
            # calculate the convolution next
            # x_np0=x.cpu().detach().numpy()
            # x_2d0 = x_np0.reshape(N, -1).transpose(1,0)
            # np.savetxt(f'in{weight_q.shape}_{weight_q[0][0][0][0]}.csv', x_2d0, delimiter=',')
            x = self._conv_forward(x, weight_q, bias_q)
            # 对输出的整型结果进行移位
            aver = x.max() #注意不是abs
            if (aver > (127 - 1)):
                left_shift = 0
                post_shift = int(np.ceil(math.log2(aver / (127 + 1))))
            else:
                left_shift = 1
                post_shift = int(np.ceil(math.log2((127 + 1) / (0.1 + aver))))
            # # x_2d1 = x.reshape(N, -1).transpose(1,0).cpu().detach().numpy()
            # # np.savetxt(f'x_before{weight_q.shape}_{weight_q[0][0][0][0]}.csv', x_2d1, delimiter=',')
            x = floor_pass(x/(2 ** post_shift))
            # # x = torch.floor(x/(2 ** post_shift))

            # x_2d = x.reshape(N, -1).transpose(1,0).cpu().detach().numpy()
            # np.savetxt(f'x{weight_q.shape}_{weight_q[0][0][0][0]}.csv', x_2d, delimiter=',')
            x = x / 127.0
        else:
            x = self._conv_forward(x, self.weight, self.bias)
        return x

class Linear_quant_noise(nn.Linear):
    def __init__(self, qn_on, in_features, out_features,
                 weight_bit,
                 output_bit,
                 isint,
                 clamp_std,
                 noise_scale,
                 bias=False, ):
        super().__init__(in_features, out_features, bias)
        self.qn_on = qn_on
        self.weight_bit = weight_bit
        self.output_bit = output_bit
        self.isint = isint
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.output_half_level = 2 ** output_bit / 2 - 1

    def forward(self, x):
        if self.qn_on:
            # 先对输入进行量化，放缩到 127 大小
            x = x * 127.0
            # 然后进行round取整
            x = round_pass(x)

            # 对权重进行量化，用整型表示
            weight_q, scale = data_quantization_sym(self.weight, self.weight_half_level, isint=1, clamp_std=self.clamp_std)
            # cout, cin = weight_q.shape
            # N = cout  # N = cout
            # weights_2d = weight_q.cpu().detach().numpy()
            #
            # np.savetxt(f'weight{weight_q.shape}_{weight_q[0][0]}.csv', weights_2d, delimiter=',')

            if self.bias != None:
                # bias = bias / scale
                bias_q, _ = data_quantization_sym(self.bias, 127, isint=1, clamp_std=0)
            else:
                bias_q = self.bias
            # calculate the convolution next

            # x_2d0 = x.cpu().detach().numpy()
            # np.savetxt(f'in{weight_q.shape}_{weight_q[0][0]}.csv', x_2d0, delimiter=',')

            x = F.linear(x, weight_q, bias_q)
            # 对输出的整型结果进行移位
            aver = x.max()  # 注意不是abs
            if (aver > (127 - 1)):
                left_shift = 0
                post_shift = int(np.ceil(math.log2(aver / (127 + 1))))
            else:
                left_shift = 1
                post_shift = int(np.ceil(math.log2((127 + 1) / (0.1 + aver))))
            # x_2d1 = x.cpu().detach().numpy()
            # np.savetxt(f'x_before{weight_q.shape}_{weight_q[0][0]}.csv', x_2d1, delimiter=',')
            x = floor_pass(x/(2 ** post_shift))
            # x_2d = x.cpu().detach().numpy()
            # np.savetxt(f'x{weight_q.shape}_{weight_q[0][0]}.csv', x_2d, delimiter=',')

            x = x / 127.0
            # x = Feature_Quant.apply(x, self.output_half_level, self.isint, None)
        else:
            x = F.linear(x, self.weight, self.bias)

        return x

class Conv2d_quant_lsq(nn.Conv2d):
    def __init__(self,
                 qn_on,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 weight_bit,
                 output_bit,
                 isint,
                 clamp_std,
                 bias,
                 symmetric=False
                 ):
        # weight_bit -> bit level for weight
        # output_bit -> bit level for output feature map
        # isint, clamp_std, noise_scale -> same arguments as Weight_Quant_Noise()
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         bias=bias
                         )
        self.qn_on = qn_on
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.output_half_level = 2 ** output_bit / 2 - 1
        self.isint = isint
        self.clamp_std = clamp_std

        if symmetric:
            # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
            self.thd_neg_w = - 2 ** (weight_bit - 1) + 1
            self.thd_pos_w = 2 ** (weight_bit - 1) - 1

            self.thd_neg_o = - 2 ** (output_bit - 1) + 1
            self.thd_pos_o = 2 ** (output_bit - 1) - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.thd_neg_w = - 2 ** (weight_bit - 1)
            self.thd_pos_w = 2 ** (weight_bit - 1) - 1

            self.thd_neg_o = - 2 ** (output_bit - 1)
            self.thd_pos_o = 2 ** (output_bit - 1) - 1
        self.s = nn.Parameter(torch.ones(1) * 0.1260)

    def forward(self, x):
        # quantize weight and add noise first
        if self.qn_on:
            # weight量化
            s_grad_scale = 1.0 / ((self.thd_pos_w * self.weight.numel()) ** 0.5)
            s = nn.Parameter(self.weight.detach().abs().mean() * 2 / (self.thd_pos_w ** 0.5))
            s_scale = grad_scale(s, s_grad_scale)

            quantized_weight = self.weight / s_scale
            quantized_weight = torch.clamp(quantized_weight, self.thd_neg_o, self.thd_pos_o)
            quantized_weight_int = round_pass(quantized_weight)
            quantized_weight = quantized_weight_int * s_scale
            # x量化
            s_grad_scale = 1.0 / ((self.thd_pos_o * x[0].numel()) ** 0.5)
            s = nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos_o ** 0.5))
            s_scale = grad_scale(s, s_grad_scale)

            quantized_act = x / s_scale
            quantized_act = torch.clamp(quantized_act, self.thd_neg_o, self.thd_pos_o)
            quantized_act_int = round_pass(quantized_act)
            quantized_act = quantized_act_int * s_scale

            # quantized_weight = self.quan_w_fn(self.weight)
            # quantized_act = self.quan_a_fn(x)
            x = self._conv_forward(quantized_act, quantized_weight, self.bias)
        #     # 先对输入进行量化，放缩到 127 大小
        #     x = x * 127
        #     # 然后进行round取整
        #     x = x.round()
        #     # 对权重进行量化，用整型表示
        #     weight_q, scale = data_quantization_sym(self.weight, self.weight_half_level, isint=1, clamp_std=self.clamp_std)
        #     # cout, cin, kernel_height, kernel_width = weight_q.shape
        #     # 6. 转换为二维形式 KxN
        #     # K = cin * kernel_height * kernel_width  # K = cin * kernel_height * kernel_width
        #     # N = cout  # N = cout
        #     # weights_2d = weight_q.permute(0, 3, 2, 1).reshape(N, K).transpose(1,0).cpu().detach().numpy()
        #     # np.savetxt(f'weight{weight_q.shape}_{weight_q[0][0][0][0]}.csv', weights_2d, delimiter=',')
        #     if self.bias != None:
        #         # bias = bias / scale
        #         bias_q, _ = data_quantization_sym(self.bias, 127, isint=1, clamp_std=0)
        #     else:
        #         bias_q = self.bias
        #     # calculate the convolution next
        #     # x_np0=x.cpu().detach().numpy()
        #     # x_2d0 = x_np0.reshape(N, -1).transpose(1,0)
        #     # np.savetxt(f'in{weight_q.shape}_{weight_q[0][0][0][0]}.csv', x_2d0, delimiter=',')
        #     x = self._conv_forward(x, weight_q, bias_q)
        #     # 对输出的整型结果进行移位
        #     aver = x.max() #注意不是abs
        #     if (aver > (127 - 1)):
        #         left_shift = 0
        #         post_shift = int(np.ceil(math.log2(aver / (127 + 1))))
        #     else:
        #         left_shift = 1
        #         post_shift = int(np.ceil(math.log2((127 + 1) / (0.1 + aver))))
        #     # x_2d1 = x.reshape(N, -1).transpose(1,0).cpu().detach().numpy()
        #     # np.savetxt(f'x_before{weight_q.shape}_{weight_q[0][0][0][0]}.csv', x_2d1, delimiter=',')
        #     x = torch.floor(x/(2 ** post_shift))
        #     # x_2d = x.reshape(N, -1).transpose(1,0).cpu().detach().numpy()
        #     # np.savetxt(f'x{weight_q.shape}_{weight_q[0][0][0][0]}.csv', x_2d, delimiter=',')
        #     x = x / 127
        else:
            x = self._conv_forward(x, self.weight, self.bias)
        return x
# ================================== #
# Quant Noise Functions
# ================================== #
# Quantize weight and add noise
class Weight_Quant_Noise(torch.autograd.Function):
    # Number of inputs (excluding ctx, only weight, bias, half_level, isint, clamp_std, noise_scale)
    # for forward need to be the same as the number of return in def backward()
    # (return weight_grad, bias_grad, None, None, None, None)
    @staticmethod
    def forward(ctx, weight, bias, half_level, isint, clamp_std, noise_scale):
        # weight -> input weight
        # bias -> input bias
        # half_level -> quantization level
        # isint -> return int (will result in scaling) or float (same scale)
        # clamp_std -> clamp weight to [- std * clamp_std, std * clamp_std]
        # noise_scale -> noise scale, equantion can be found in add_noise()
        ctx.save_for_backward()

        std = weight.std()
        if clamp_std != 0:
            weight = torch.clamp(weight, min=-clamp_std * std, max=clamp_std * std)

        # log down the max scale for input weight
        scale_in = abs(weight).max()

        # log down the max scale for input weight
        weight, scale = data_quantization_sym(weight, half_level, scale=scale_in,
                                              isint=isint, clamp_std=0)
        # add noise to weight
        weight = add_noise(weight, n_scale=noise_scale)

        # No need for bias quantization, since the bias is added to the feature map on CPU (or GPU)
        if bias != None:
            # bias = bias / scale
            bias, _ = data_quantization_sym(bias, 127,
                                            isint=isint, clamp_std=0)
            # bias = add_noise(bias, n_scale=noise_scale)

        return weight, bias

    # Use default gradiant to train the network
    # Number of inputs (excluding ctx, only weight_grad, bias_grad) for backward need to be the same as the
    # number of return in def forward() (return weight, bias)
    @staticmethod
    def backward(ctx, weight_grad, bias_grad):
        return weight_grad, bias_grad, None, None, None, None


class Feature_Quant_noise(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feature, half_level, scale, isint, noise_scale):
        # feature_q, _ = data_quantization_sym(feature, half_level, scale = None, isint = isint, clamp_std = 0)
        feature_q = add_noise(feature, n_scale=noise_scale)
        feature_q, _ = data_quantization_sym(feature_q, half_level=half_level, scale = scale, isint = isint)

        return feature_q

    @staticmethod
    def backward(ctx, feature_grad):
        return feature_grad, None, None, None

# A convolution layer which adds noise and quantize the weight and output feature map
class Conv2d_quant_noise(nn.Conv2d):
    def __init__(self,
                 qn_on,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 weight_bit,
                 output_bit,
                 isint,
                 clamp_std,
                 noise_scale,
                 bias,
                 ):
        # weight_bit -> bit level for weight
        # output_bit -> bit level for output feature map
        # isint, clamp_std, noise_scale -> same arguments as Weight_Quant_Noise()
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         bias=bias
                         )
        self.qn_on = qn_on
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.output_half_level = 2 ** output_bit / 2 - 1
        self.isint = isint
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale

    def forward(self, x):
        # quantize weight and add noise first
        if self.qn_on:
            weight_q, bias_q = Weight_Quant_Noise.apply(self.weight, self.bias,
                                                        self.weight_half_level, self.isint, self.clamp_std,
                                                        self.noise_scale)
            # calculate the convolution next
            x = self._conv_forward(x, weight_q, bias_q)

            # quantize the output feature map at last
            x = Feature_Quant.apply(x, self.output_half_level, self.isint)
        else:
            x = self._conv_forward(x, self.weight, self.bias)

        return x


def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)


def reshape_to_bias(input):
    return input.reshape(-1)


# ====================================================================== #
# Customized nn.Module layers for quantization and noise adding
# ====================================================================== #
# A quantization layer
class Layer_Quant(nn.Module):
    def __init__(self, bit_level, isint, clamp_std):
        super().__init__()
        self.isint = isint
        self.output_half_level = 2 ** bit_level / 2 - 1
        self.clamp_std = clamp_std

    def forward(self, x):
        x = Feature_Quant.apply(x, self.output_half_level, self.isint)
        return x


class Layer_Quant_noise(nn.Module):
    def __init__(self, bit_level, isint, clamp_std, noise_scale):
        super().__init__()
        self.isint = isint
        self.output_half_level = 2 ** bit_level / 2 - 1
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale

    def forward(self, x):
        x = Feature_Quant.apply(x, self.output_half_level, self.isint)
        return x


# BN融合
class BNFold_Conv2d_Q(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight_bit,
            output_bit,
            isint,
            clamp_std,
            noise_scale,
            bias,
            eps=1e-5,
            momentum=0.01, ):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         bias=True,
                         )
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.output_half_level = 2 ** output_bit / 2 - 1
        self.isint = isint
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale

        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('first_bn', torch.zeros(1))
        init.uniform_(self.gamma)
        init.zeros_(self.beta)

    def forward(self, input):
        # 训练态
        output = self._conv_forward(input, self.weight, self.bias)
        # 先做普通卷积得到A，以取得BN参数

        # 更新BN统计参数（batch和running）
        dims = [dim for dim in range(4) if dim != 1]
        batch_mean = torch.mean(output, dim=dims)
        batch_var = torch.var(output, dim=dims)
        with torch.no_grad():
            if self.first_bn == 0:
                self.first_bn.add_(1)
                self.running_mean.add_(batch_mean)
                self.running_var.add_(batch_var)
            else:
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
        # BN融合
        if self.bias is not None:
            bias = reshape_to_bias(
                self.beta + (self.bias - batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
        else:
            bias = reshape_to_bias(
                self.beta - batch_mean * (self.gamma / torch.sqrt(batch_var + self.eps)))  # b融batch
        weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running

        # 量化A和bn融合后的W
        if qn_on:
            weight_q, bias_q = Weight_Quant_Noise.apply(weight, bias,
                                                        self.weight_half_level, self.isint, self.clamp_std,
                                                        self.noise_scale)
        else:
            weight_q = weight
            bias_q = bias
        # 量化卷积
        output = self._conv_forward(input, weight_q, bias_q)
        # output = F.conv2d(
        #     input=input,
        #     weight=weight_q,
        #     bias=self.bias,  # 注意，这里不加bias（self.bias为None）
        #     stride=self.stride,
        #     padding=self.padding,
        #     dilation=self.dilation,
        #     groups=self.groups
        # )
        # # # （这里将训练态下，卷积中w融合running参数的效果转为融合batch参数的效果）running ——> batch
        # output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
        # output += reshape_to_activation(bias_q)
        # 量化输出
        if qnon:
            output = Feature_Quant.apply(output, self.output_half_level, self.isint)

        return output


# A fully connected layer which adds noise and quantize the weight and output feature map
# See notes in Conv2d_quant_noise



# ================================== #
# Other Functions, rarely used
# ================================== #
def plt_weight_dist(weight, name, bins):
    num_ele = weight.numel()
    weight_np = weight.cpu().numpy().reshape(num_ele, -1).squeeze()
    plt.figure()
    plt.hist(weight_np, density=True, bins=bins)
    plot_name = f"saved_best_examples/weight_dist_{name}.png"
    plt.savefig(plot_name)
    plt.close()


# Similar to Conv2d_quant_noise, only add noise without quantization
class Conv2d_noise(nn.Conv2d):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 bias=False,
                 ):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         bias=False,
                         )

    def forward(self, x):
        weight_n = add_noise(self.weight)
        x = self._conv_forward(x, weight_n, self.bias)
        return x

IEEE_BIT_FORMATS = {
    torch.float32: {'exponent': 8, 'mantissa': 23, 'workable': torch.int32},
    torch.float16: {'exponent': 5, 'mantissa': 10, 'workable': torch.int32},
    torch.bfloat16: {'exponent': 8, 'mantissa': 7, 'workable': torch.int32}
}


def print_float32(val: float):
    """ Print Float32 in a binary form """
    m = struct.unpack('I', struct.pack('f', val))[0]
    return format(m, 'b').zfill(32)


# print_float32(0.15625)


def print_float16(val: float):
    """ Print Float16 in a binary form """
    m = struct.unpack('H', struct.pack('e', np.float16(val)))[0]
    return format(m, 'b').zfill(16)

# def print_float8(val: float):
#     """ Print Float16 in a binary form """
#     m = struct.unpack('H', struct.pack('e', np.float16(val)))[0]
#     return format(m, 'b').zfill(16)
# # print_float16(3.14)

def ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=8, mant_len=23):
    """ Convert binary data into the floating point value """
    sign_mult = -1 if sign == 1 else 1
    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
    mant_mult = 1
    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (mant_len - b))

    return sign_mult * (2 ** exponent) * mant_mult


# ieee_754_conversion(0b0, 0b01111100, 0b01000000000000000000000)

def print_bits(x: torch.Tensor, bits: int):
    """Prints the bits of a tensor

    Args:
        x (torch.Tensor): The tensor to print
        bits (int): The number of bits to print

    Returns:
        ByteTensor : The bits of the tensor
    """
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    bit = x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
    return bit


def shift_base(exponent_bits: int):
    """Computes the bias for a given number of exponent bits

    Args:
        exponent_bits (int): The number of exponent bits

    Returns:
        int : The bias for the given number of exponent bits
    """
    return 2 ** (exponent_bits - 1) - 1


def shift_right(t, shift_by: int):
    """Shifts a tensor to the right by a given number of bits

    Args:
        t (torch.Tensor): The tensor to shift
        shift_by (int): The number of bits to shift by

    Returns:
        torch.Tensor : The shifted tensor
    """
    return t >> shift_by


def shift_left(t, shift_by: int):
    """Shifts a tensor to the left by a given number of bits

    Args:
        t (torch.Tensor): The tensor to shift
        shift_by (int): The number of bits to shift by

    Returns:
        torch.Tensor : The shifted tensor
    """
    return t << shift_by


def fp8_downcast(source_tensor, n_bits: int):
    """Downcasts a tensor to an 8 bit float

    Args:
        source_tensor (torch.Tensor): The tensor to downcast
        n_bits (int): The number of bits to use for the mantissa

    Raises:
        ValueError: If the mantissa is too large for an 8 bit float

    Returns:
        ByteTensor : The downcasted tensor
    """
    target_m_nbits = n_bits
    target_e_nbits = 7 - n_bits

    if target_e_nbits + target_m_nbits + 1 > 8:
        raise ValueError("Mantissa is too large for an 8 bit float")

    source_m_nbits = IEEE_BIT_FORMATS[source_tensor.dtype]['mantissa']
    source_e_nbits = IEEE_BIT_FORMATS[source_tensor.dtype]['exponent']
    source_all_nbits = 1 + source_m_nbits + source_e_nbits
    int_type = torch.int32 if source_all_nbits == 32 else torch.int16

    # Extract the sign
    sign = shift_right(source_tensor.view(int_type), source_all_nbits - 1).to(torch.uint8)
    sign = torch.bitwise_and(sign, torch.ones_like(sign, dtype=torch.uint8))

    # Zero out the sign bit
    bit_tensor = torch.abs(source_tensor)

    # Shift the base to the right of the buffer to make it an int
    base = shift_right(bit_tensor.view(int_type), source_m_nbits)
    # Shift the base back into position and xor it with bit tensor to get the mantissa by itself
    mantissa = torch.bitwise_xor(shift_left(base, source_m_nbits), bit_tensor.view(int_type))
    maskbase = torch.where(base>0,1,0)
    maskmantissa = torch.where(mantissa <= 0,1,0)
    maskadd = maskbase*maskmantissa

    # Shift the mantissa left by the target mantissa bits then use modulo to zero out anything outside of the mantissa
    # t1 = (shift_left(mantissa, target_m_nbits) % (2 ** source_m_nbits))
    # # Create a tensor of fp32 1's and convert them to int32
    # t2 = torch.ones_like(source_tensor).view(int_type)
    # Use bitwise or to combine the 1-floats with the shifted mantissa to get the probabilities + 1 and then subtract 1
    # expectations = (torch.bitwise_or(t1, t2).view(source_tensor.dtype) - 1)

    # Stochastic rounding
    # torch.ceil doesnt work on float16 tensors
    # https://github.com/pytorch/pytorch/issues/51199
    # r = torch.rand_like(expectations, dtype=torch.float32)
    # ones = torch.ceil(expectations.to(torch.float32) - r).type(torch.uint8)

    # Shift the sign, base, and mantissa into position
    target_sign = shift_left(sign, 7)
    target_base = base.type(torch.int16) - shift_base(source_e_nbits) + shift_base(target_e_nbits)# + maskadd
    mask_b = torch.where(target_base<0, 0, 1)
    target_base = (shift_left(target_base, target_m_nbits) * mask_b).to(torch.uint8)
    target_mantissa = shift_right(mantissa, source_m_nbits - target_m_nbits).to(torch.uint8)

    # target_mantissa = [[ele if sign[i][j]==0 else 2**target_m_nbits - ele for j, ele in enumerate(row)] for i, row in enumerate(target_mantissa)]

    fp8_as_uint8 = target_sign + target_base + target_mantissa
    mask = torch.where(source_tensor == 0 , 0, 1)
    maskc = torch.where(fp8_as_uint8 == 128, 0, 1)

    fp8_as_uint8 = fp8_as_uint8 * mask*mask_b*maskc
    return fp8_as_uint8 #+ ones * mask*mask_b

# def uint8_to_fp32(source_tensor: torch.ShortTensor, n_bits: int, left_shift_bit: int):
#     """Converts a uint8 tensor to a fp16 tensor
#
#     Args:
#         source_tensor (torch.ByteTensor): The tensor to convert
#         n_bits (int): The number of bits to use for the mantissa
#
#     Returns:
#         _type_: The converted tensor
#     """
#     if source_tensor.dtype != torch.uint8:
#         source_tensor = source_tensor.clip(0, 2**(8+n_bits))
#     source_tensor = source_tensor.clone().detach().to(torch.int16)
#     mask = torch.where(source_tensor == 0, 0, 1)
#
#     source_m_nbits = n_bits
#     source_e_nbits = 7 - n_bits
#
#     # Extract sign as int16
#     sign = shift_right(source_tensor, 7 + left_shift_bit)
#     # shifted_sign = shift_left(sign.type(torch.int16), 15)
#     m_8bit = source_tensor % (1<< (n_bits+left_shift_bit))
#     & 0b00000111
#     # Extract base as int16 and adjust the bias accordingly
#     base_mantissa = shift_left(source_tensor, 1 + 24 - left_shift_bit)
#     base = shift_right(base_mantissa, source_m_nbits + 1 + 24 - left_shift_bit) - shift_base(source_e_nbits)
#     base = base.type(torch.int16) + shift_base(5)
#     shifted_base = shift_left(base, 10)
#
#     # Extract mantissa as int16
#     mantissa = shift_left(base_mantissa, source_e_nbits)
#     shifted_mantissa = shift_left(mantissa.type(torch.int16), 2)
#     recover_m = shifted_mantissa.view(torch.float16).float() / (1<<left_shift_bit)
#     out = mask*(shifted_base.view(torch.float16).float() + shifted_sign.view(torch.float16).float() + recover_m)
#     return out

def uint8_to_fp32(source_tensor: torch.ShortTensor, sign=None, e_max=None, m_sft=None, n_bits: int=3, left_shift_bit: int=0):
    """Converts a uint8 tensor to a fp16 tensor

    Args:
        source_tensor (torch.ByteTensor): The tensor to convert
        n_bits (int): The number of bits to use for the mantissa

    Returns:
        _type_: The converted tensor
    """
    source_tensor = source_tensor.clone().detach().to(torch.int16)
    mask = torch.where(source_tensor == 0, 0, 1)

    if sign is None or e_max is None or m_sft is None:
        if left_shift_bit==3:
            sign = (source_tensor & 0b0000_0100_0000_0000)>>10
            # shifted_sign = shift_left(sign.type(torch.int16), 15)
            m_8bit = source_tensor % (1<< (n_bits+left_shift_bit))
            m_8bit = (source_tensor & 0b0000_0000_0011_1111)
            e_4bit = (source_tensor & 0b0000_0011_1100_0000)>>6
        else:
            sign = (source_tensor & 0b0000_0000_1000_0000)>>7
            # shifted_sign = shift_left(sign.type(torch.int16), 15)
            m_8bit = source_tensor % (1 << (n_bits + left_shift_bit))
            m_8bit = (source_tensor & 0b0000_0000_0000_0111)
            e_4bit = (source_tensor & 0b0000_0000_0111_1000)>>3
    else:
        e_4bit = e_max
        m_8bit = m_sft
    m_float = (m_8bit.float() / (1<< (n_bits+left_shift_bit)))+1
    e_float = (2 ** (e_4bit-7)).float()
    out = (-1)**sign*e_float*m_float*mask
    return out

def fp8_alignment(ifm_uint8, left_shift_bit=3):
    device = ifm_uint8.device  # 获取输入张量的设备
    mask = torch.where(ifm_uint8 == 0, 0, 1).to(device)
    # 最高位和低3位
    # m = (torch.ones(ifm_uint8.shape, dtype=torch.uint8, device=device) & 0b00000001) << 3 | (ifm_uint8 & 0b00000111)
    m = ifm_uint8 & 0b00000111
    m = m * mask

    s = (ifm_uint8 & 0b10000000) >> 7
    # 低4~7位
    e = (ifm_uint8 & 0b01111000) >> 3
    # 每行的最大值
    pre_macro_data_e_max, _ = torch.max(e, dim=1)
    e_delta = pre_macro_data_e_max.unsqueeze(1) - e
    m_shifted = (m << left_shift_bit) >> e_delta
    m_shifted_signed = (-1)**s * m_shifted
    e_max = (e + e_delta) * mask
    result = (s << (7+left_shift_bit)) + (e_max << (3+left_shift_bit)) + m_shifted

    return result, s, e_max, m_shifted

# 示例用法
# ifm_uint8 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint8)
# result = fp8_alignment(ifm_uint8)
# print(result)

# a = torch.tensor([[-0.0397, -0.0279,  0.0117], [-0.0397, -0.0279,  0.0117]])
# b = fp8_downcast(a, 3)
# c = fp8_alignment(b, 0)
# d = uint8_to_fp32(c, 3)

def fp8_add(floatA, floatB):
    exponentA = (floatA & 0b01111000)>>3
    exponentB = (floatB & 0b01111000)>>3
    fractionA = (floatA & 0b00000111) + 8
    fractionB = (floatB & 0b00000111) + 8
    exponent = exponentA
    if floatA == 0 or floatA == 128:
        sum = floatB
    elif floatB == 0 or floatB == 128:
        sum = floatA
    elif ((floatA & 0b01111111) == (floatB & 0b01111111)) and ((floatA & 0b10000000) != (floatB & 0b10000000)):
        sum = 0
    else:
        if exponentB > exponentA:
            shiftAmount = exponentB - exponentA
            fractionA = (fractionA >> (shiftAmount))
            exponent = exponentB
        elif exponentA > exponentB:
            shiftAmount = exponentA - exponentB
            fractionB = (fractionB >> (shiftAmount))
            exponent = exponentA

        if (floatA & 0b10000000) == (floatB & 0b10000000):
            fraction = fractionA + fractionB
            if fraction >= 2**4:
                fraction = fraction >> 1
                exponent = exponent + 1
            sign = floatA & 0b10000000
        else:
            if floatA & 0b10000000:
                fraction = fractionB - fractionA
            else:
                fraction = fractionA - fractionB

            sign = (fraction<0)*128
            if sign:
                fraction =(-fraction)

        if fraction & 0b00001000 == 0:
            if fraction & 0b00000100:
                fraction = (fraction << 1)
                exponent = exponent - 1
            elif fraction & 0b00000010:
                fraction = (fraction << 2)
                exponent = exponent - 2
            elif fraction & 0b00000001:
                fraction = (fraction << 3)
                exponent = exponent - 3

        mantissa = fraction & 0b00000111
        if (exponent < 0):
            sum = 0
        elif (exponent >= 16):
            sum = sign + 127
        elif (((exponent & 0b00001111) + mantissa)==0):
            sum = 0
        else:
            sum = sign + ((exponent & 0b00001111)<<3) + mantissa
    if sum == 128:
        sum = 0
    return sum

    # t = np.array([[-32,1],[-1,32]])

class Weight_fp(torch.autograd.Function):
    # Number of inputs (excluding ctx, only weight, bias, half_level, isint, clamp_std, noise_scale)
    # for forward need to be the same as the number of return in def backward()
    # (return weight_grad, bias_grad, None, None, None, None)
    @staticmethod
    def forward(ctx, weight, n_bits):
        ctx.save_for_backward()
        # 首先计算scale_factor
        weight_max = torch.max(torch.abs(weight))
        scaling_factor = weight_max/(1.875 * 2**8)
        weight_scale = weight / scaling_factor
        weight_n_scale = fp8_downcast(weight_scale, n_bits)
        weight_n_scale = uint8_to_fp32(weight_n_scale, n_bits=n_bits, left_shift_bit=0)
        weight_n = weight_n_scale * scaling_factor

        return weight_n

    # Use default gradiant to train the network
    # Number of inputs (excluding ctx, only weight_grad, bias_grad) for backward need to be the same as the
    # number of return in def forward() (return weight, bias)
    @staticmethod
    def backward(ctx, weight_grad):
        return weight_grad, None, None, None, None

class Feature_fp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feature, n_bits):
        ctx.save_for_backward()
      # 首先计算scale_factor
        feature_max = torch.max(torch.abs(feature))
        scaling_factor = feature_max / (1.875 * 2**8)
        feature_scale = feature / scaling_factor

        feature_n_scale = fp8_downcast(feature_scale, n_bits)
        feature_n_scale = uint8_to_fp32(feature_n_scale, n_bits=n_bits, left_shift_bit=0)
        feature_n = feature_n_scale * scaling_factor

        return feature_n


    @staticmethod
    def backward(ctx, feature_grad):
        return feature_grad, None, None, None



DBG = 0

# class Weight_fp_hw(torch.autograd.Function):
#     # Number of inputs (excluding ctx, only weight, bias, half_level, isint, clamp_std, noise_scale)
#     # for forward need to be the same as the number of return in def backward()
#     # (return weight_grad, bias_grad, None, None, None, None)
#     @staticmethod
#     def forward(ctx, weight, n_bits, quant_type, group_number,left_shift_bit=0):
#         # quant type can be none, layer, channel, group
#         ctx.save_for_backward()
#         #对称量化
#         # weight_max = torch.max(torch.abs(weight))
#         # scaling_factor = weight_max/(1.875 * 2**(n_bits+left_shift_bit))
#         # weight_scale = weight / scaling_factor
#
#
#         #非对称量化
#         weight_max = torch.max(weight)
#         weight_min = torch.min(weight)
#         scaling_factor = (weight_max-weight_min) / (1.875 * 2**(n_bits+left_shift_bit))/2
#         weight_temp = (weight - weight_min) / scaling_factor - (1.875 * 2**(n_bits+left_shift_bit))
#         weight_n_scale = fp8_downcast(weight_temp, n_bits)
#         if DBG:
#             print(f"\nweight_max:{weight_max},weight_min:{weight_min},scaling_factor:{scaling_factor},weight_temp.max():{weight_temp.max()},weight_temp.min():{weight_temp.min()}")
#         co, ci, kx, ky = weight_n_scale.shape
#         if quant_type == 'channel':
#             weight_reshape = weight_n_scale.reshape([co,-1])
#             weight_align, e_max, m_sft = fp8_alignment(weight_reshape, left_shift_bit)
#         elif quant_type == 'layer':
#             weight_reshape = weight_n_scale.reshape([1,-1])
#             weight_align, e_max, m_sft = fp8_alignment(weight_reshape, left_shift_bit)
#         elif quant_type == 'group':
#             # 计算需要的填充数量
#             total_elements = co * ci * kx * ky
#             remainder = total_elements % group_number
#             if remainder != 0:
#                 padding_size = group_number - remainder
#             else:
#                 padding_size = 0
#             # 用零填充张量
#             if padding_size > 0:
#                 # 创建一个与 weight_n 具有相同维度的 padding
#                 padding_shape = list(weight_n_scale.shape)
#                 padding_shape[0] = padding_size  # 只在最后一个维度上添加
#                 padding = torch.zeros(padding_shape, device=weight_n_scale.device, dtype=weight_n_scale.dtype)
#                 weight_n = torch.cat((weight_n_scale, padding))
#             weight_reshape = weight_n_scale.reshape([-1, group_number])
#             weight_align, sign, e_max, m_sft = fp8_alignment(weight_reshape, left_shift_bit)
#         else:
#             weight_align = weight_n_scale
#         weight_align = weight_align.reshape([-1, ci, kx, ky])
#         e_max = e_max.reshape([-1, ci, kx, ky])
#         m_sft = m_sft.reshape([-1, ci, kx, ky])
#         sign = sign.reshape([-1, ci, kx, ky])
#         weight_align = weight_align[:co, :ci, :kx, :ky]
#         e_max = e_max[:co, :ci, :kx, :ky]
#         m_sft = m_sft[:co, :ci, :kx, :ky]
#         sign = sign[:co, :ci, :kx, :ky]
#         a = weight_temp
#         weight_align_fp = uint8_to_fp32(weight_align, sign, e_max, m_sft, n_bits, left_shift_bit=left_shift_bit)
#         weight_align_fp_out = (weight_align_fp +(1.875 * 2**(n_bits+left_shift_bit))) * scaling_factor + weight_min
#         b = weight
#         if DBG:
#             # 计算绝对误差
#             absolute_error = torch.abs(weight_align_fp_out - weight)
#             # 避免除以零的情况
#             epsilon = 1e-10
#             # 计算误差百分比
#             zero_mask = (weight != 0.0)
#             error_percentage = (absolute_error / (torch.abs(weight) + epsilon)) * 100 * zero_mask
#             error_percentage_max = torch.max(error_percentage)
#             max_index = torch.argmax(error_percentage)
#             d0, d1, d2, d3 = error_percentage.shape
#
#             i = max_index // (d1 * d2 * d3)
#             j = (max_index % (d1 * d2 * d3)) // (d2 * d3)
#             k = (max_index % (d2 * d3)) // d3
#             l = max_index % d3
#             print(error_percentage[i,j,k,l], weight_align_fp_out[i,j,k,l], weight[i,j,k,l])
#             # 计算平均误差百分比
#             mean_error_percentage = torch.mean(error_percentage).item()
#             # print(f'平均误差百分比-lfs{left_shift_bit}: {mean_error_percentage:.2f}%')
#             max_count = torch.sum(error_percentage == error_percentage_max)
#             # 计算总元素个数
#             total_elements = error_percentage.numel()
#             # 计算最大值的占比
#             max_ratio = max_count.float() / total_elements
#         return weight_align_fp_out
#
#     # Use default gradiant to train the network
#     # Number of inputs (excluding ctx, only weight_grad, bias_grad) for backward need to be the same as the
#     # number of return in def forward() (return weight, bias)
#     @staticmethod
#     def backward(ctx, weight_grad):
#         return weight_grad, None, None, None, None
class Weight_fp_hw(torch.autograd.Function):
    # Number of inputs (excluding ctx, only weight, bias, half_level, isint, clamp_std, noise_scale)
    # for forward need to be the same as the number of return in def backward()
    # (return weight_grad, bias_grad, None, None, None, None)
    @staticmethod
    def forward(ctx, weight, n_bits, quant_type, group_number,left_shift_bit=0):
        # quant type can be none, layer, channel, group
        ctx.save_for_backward()
        #对称量化
        weight_max = torch.max(torch.abs(weight))
        weight_min=0
        scaling_factor = weight_max/(1.875 * 2**(n_bits+left_shift_bit))
        weight_scale = weight / scaling_factor
        q_min=0
        #非对称量化
        # weight_max = torch.max(weight)
        # weight_min = torch.min(weight)
        # scaling_factor = (weight_max-weight_min) / (1.875 * 2**(n_bits+left_shift_bit-3))/2
        # weight_scale = (weight - weight_min) / scaling_factor - (1.875 * 2**(n_bits+left_shift_bit-3))
        # q_min = -488
        weight_n_scale = fp8_downcast(weight_scale, n_bits)
        if DBG:
            print(f"\nweight_max:{weight_max},weight_min:{weight_min},scaling_factor:{scaling_factor},weight_temp.max():{weight_scale.max()},weight_temp.min():{weight_scale.min()}")
        co, ci, kx, ky = weight_n_scale.shape
        total_elements = co * ci * kx * ky
        if quant_type == 'channel':
            weight_reshape = weight_n_scale.reshape([co,-1])
            weight_align, sign, e_max, m_sft = fp8_alignment(weight_reshape, left_shift_bit)
            # weight_align = weight_align.transpose(1,0)
            # sign=sign.transpose(1,0)
            # e_max=e_max.transpose(1,0)
            # m_sft =m_sft.transpose(1,0)
        elif quant_type == 'layer':
            weight_reshape = weight_n_scale.reshape([1,-1])
            weight_align, sign, e_max, m_sft = fp8_alignment(weight_reshape, left_shift_bit)
            # weight_align = weight_align.transpose(1,0)
            # sign=sign.transpose(1,0)
            # e_max=e_max.transpose(1,0)
            # m_sft =m_sft.transpose(1,0)
        elif quant_type == 'group':
            # 计算需要的填充数量
            remainder = total_elements % group_number
            if remainder != 0:
                padding_size = group_number - remainder
            else:
                padding_size = 0
            # 用零填充张量
            if padding_size > 0:
                # 创建一个与 weight_n 具有相同维度的 padding
                weight_n_scale = weight_n_scale.reshape([-1, 1])
                padding_shape = list(weight_n_scale.shape)
                padding_shape[0] = padding_size  # 只在最后一个维度上添加
                padding = torch.zeros(padding_shape, device=weight_n_scale.device, dtype=weight_n_scale.dtype)
                weight_n_scale = torch.cat((weight_n_scale, padding))
            weight_reshape = weight_n_scale.reshape([-1, group_number])
            weight_align, sign, e_max, m_sft = fp8_alignment(weight_reshape, left_shift_bit)
        else:
            weight_reshape = weight_n_scale.reshape([-1, 1])
            weight_align, sign, e_max, m_sft = fp8_alignment(weight_reshape, left_shift_bit)

        weight_align = weight_align.reshape([-1, 1])
        e_max = e_max.reshape([-1, 1])
        m_sft = m_sft.reshape([-1, 1])
        sign = sign.reshape([-1, 1])
        #
        weight_align = weight_align[:total_elements,:]
        e_max = e_max[:total_elements, :]
        m_sft = m_sft[:total_elements, :]
        sign = sign[:total_elements, :]

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
        b = weight
        # print(f"weight_align_fp_out={torch.max(torch.abs(weight_align_fp_out))},weight_max={weight_max}")
        # 计算绝对误差
        absolute_error = torch.abs(weight_align_fp_out - weight)
        # 避免除以零的情况
        epsilon = 1e-10
        # 计算误差百分比
        zero_mask = (weight != 0.0)
        error_percentage = (absolute_error / (torch.abs(weight) + epsilon)) * 100 * zero_mask
        error_percentage_max = torch.max(error_percentage)
        max_index = torch.argmax(error_percentage)
        d0, d1, d2, d3 = error_percentage.shape

        i = torch.div(max_index, (d1 * d2 * d3), rounding_mode='floor')
        j = torch.div(max_index % (d1 * d2 * d3), (d2 * d3), rounding_mode='floor')
        k = torch.div(max_index % (d2 * d3), d3, rounding_mode='floor')
        l = max_index % d3
        wm = weight_align_fp_out[i, j, k, l],
        wmr= weight[i, j, k, l]
        # 计算平均误差百分比
        mean_error_percentage = torch.mean(error_percentage).item()
        wmax = weight.max()
        wmin = weight.min()
        max_count = torch.sum(error_percentage == error_percentage_max)
        # 计算总元素个数
        total_elements = error_percentage.numel()
        # 计算最大值的占比
        max_ratio = max_count.float() / total_elements


        if DBG:
            print(f'平均误差百分比-lfs{left_shift_bit}: {mean_error_percentage:.2f}%')
            print(error_percentage[i, j, k, l], wm,wmr)
        return weight_align_fp_out

    # Use default gradiant to train the network
    # Number of inputs (excluding ctx, only weight_grad, bias_grad) for backward need to be the same as the
    # number of return in def forward() (return weight, bias)
    @staticmethod
    def backward(ctx, weight_grad):
        return weight_grad, None, None, None, None
class Feature_fp_hw(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feature, n_bits, quant_type, group_number,left_shift_bit=3):
        # quant type can be none, layer, channel, group
        ctx.save_for_backward()
        feature_max = torch.max(torch.abs(feature))
        scaling_factor = feature_max / (1.875 * 2**(n_bits+left_shift_bit))
        feature_scale = feature / scaling_factor

        feature_n = fp8_downcast(feature_scale, n_bits)
        co, ci, kx, ky = feature_n.shape
        total_elements = co * ci * kx * ky
        if quant_type == 'channel':
            feature_reshape = feature_n.reshape([co,-1])
            feature_align, sign, e_max, m_sft = fp8_alignment(feature_reshape, left_shift_bit=left_shift_bit)

        elif quant_type == 'layer':
            feature_reshape = feature_n.reshape([1,-1])
            feature_align, sign, e_max, m_sft = fp8_alignment(feature_reshape, left_shift_bit=left_shift_bit)

        elif quant_type == 'group':
            # 计算需要的填充数量

            remainder = total_elements % group_number
            if remainder != 0:
                padding_size = group_number - remainder
            else:
                padding_size = 0
            # 用零填充张量
            if padding_size > 0:
                # 创建一个与 weight_n 具有相同维度的 padding
                feature_n = feature_n.reshape([-1, 1])
                padding_shape = list(feature_n.shape)
                padding_shape[0] = padding_size  # 只在最后一个维度上添加
                padding = torch.zeros(padding_shape, device=feature_n.device, dtype=feature_n.dtype)
                feature_n = torch.cat((feature_n, padding))
            feature_reshape = feature_n.reshape([-1, group_number])
            feature_align, sign, e_max, m_sft = fp8_alignment(feature_reshape, left_shift_bit)
        else:
            feature_reshape = feature_n.reshape([-1, 1])
            feature_align, sign, e_max, m_sft = fp8_alignment(feature_reshape, left_shift_bit)
        # feature_align = feature_align.reshape([-1, 1])
        # e_max = e_max.reshape([-1, 1])
        # m_sft = m_sft.reshape([-1, 1])
        # sign = sign.reshape([-1, 1])
        #
        # feature_align = feature_align[:total_elements,:]
        # e_max = e_max[:total_elements, :]
        # m_sft = m_sft[:total_elements, :]
        # sign = sign[:total_elements, :]`

        # feature_align = feature_align.reshape([-1, ci, kx, ky])
        # e_max = e_max.reshape([-1, ci, kx, ky])
        # m_sft = m_sft.reshape([-1, ci, kx, ky])
        # sign = sign.reshape([-1, ci, kx, ky])
        feature_align = feature_align.reshape([co, ci, kx, ky])
        e_max = e_max.reshape([co, ci, kx, ky])
        m_sft = m_sft.reshape([co, ci, kx, ky])
        sign = sign.reshape([co, ci, kx, ky])
        feature_align_fp = uint8_to_fp32(feature_align, sign, e_max, m_sft, n_bits, left_shift_bit=left_shift_bit)
        # print("\n",feature[:,0,0,0])
        # # print(feature_n)
        # # print(feature_align)
        # print(feature_align_fp[:,0,0,0])
        feature_align_fp_out = feature_align_fp * scaling_factor

        # print(f'feature_align_fp_out max = {torch.max(torch.abs(feature_align_fp_out))},feature_max = {torch.max(torch.abs(feature))}')

        if torch.isnan(feature_align_fp_out).any():
            print("Nan in feature_align_fp_out")
        # 计算绝对误差
        absolute_error = torch.abs(feature_align_fp_out - feature)
        # 避免除以零的情况
        epsilon = 1e-10
        # 计算误差百分比
        zero_mask = (feature != 0.0)
        error_percentage = (absolute_error / (torch.abs(feature) + epsilon)) * 100 * zero_mask
        error_percentage_max = torch.max(error_percentage)
        max_index = torch.argmax(error_percentage)
        d0, d1, d2, d3 = error_percentage.shape

        i = torch.div(max_index, (d1 * d2 * d3), rounding_mode='floor')
        j = torch.div(max_index % (d1 * d2 * d3), (d2 * d3), rounding_mode='floor')
        k = torch.div(max_index % (d2 * d3), d3, rounding_mode='floor')
        l = max_index % d3
        # print(error_percentage[i, j, k, l], feature_align_fp_out[i, j, k, l], feature[i, j, k, l])
        # 计算平均误差百分比
        mean_error_percentage = torch.mean(error_percentage).item()
        # print(f'对称平均误差百分比-lfs{left_shift_bit}: {mean_error_percentage:.2f}%')
        max_count = torch.sum(error_percentage == error_percentage_max)
        # 计算总元素个数
        total_elements = error_percentage.numel()
        # 计算最大值的占比
        max_ratio = max_count.float() / total_elements
        return feature_align_fp_out

    @staticmethod
    def backward(ctx, feature_grad):
        return feature_grad, None, None, None, None

# for i in range(3):
#     tensor1 = torch.randn(4, 3, 3, 3)
#
#     # tensor1 = torch.tensor([[[[-9.0432e-03,  1.2039e-02, -3.9068e-02],
#     #       [ 1.5672e-02, -8.7397e-02,  8.9242e-03],
#     #       [-1.9430e-02,  4.3638e-02, -2.9295e-02]],
#     #
#     #      [[ 1.3471e-02,  3.2389e-02, -2.3290e-02],
#     #       [ 2.7648e-02, -7.7760e-02,  2.0374e-02],
#     #       [-4.5371e-03,  5.8875e-02, -2.1869e-02]],
#     #
#     #      [[ 3.1740e-04,  4.4886e-02, -1.4928e-02],
#     #       [ 8.7579e-03, -6.2884e-02,  2.9567e-02],
#     #       [-3.3912e-02,  4.3335e-02, -1.7217e-02]]],
#     #
#     #     [[[-4.0797e-02, -6.5671e-02, -3.4777e-02],
#     #       [-7.7855e-02,  6.3777e-02, -5.5809e-02],
#     #       [-2.3450e-02, -5.9316e-02,  2.1160e-02]],
#     #
#     #      [[-3.8100e-02, -7.5460e-02, -3.3063e-02],
#     #       [-1.4496e-01, -4.6420e-02, -1.2613e-01],
#     #       [-9.5986e-02, -2.0046e-01, -7.6887e-02]],
#     #
#     #      [[ 7.2325e-02,  4.6303e-02,  4.2261e-02],
#     #       [ 3.8953e-02,  1.7025e-01,  3.2282e-02],
#     #       [ 4.4703e-02, -1.1234e-02,  5.3663e-02]]]])
#     b = Weight_fp_hw.apply(tensor1, 3, "group", 1, 3)
#     c = Weight_fp_hw.apply(tensor1, 3, "group", 1, 0)
#     d = Feature_fp_hw.apply(tensor1, 3, "group", 1, 3)
#     e = Feature_fp_hw.apply(tensor1, 3, "group", 1, 0)
# print()

class Conv2d_fp8(nn.Conv2d):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation = 1,
                 groups: int = 1,
                 bias=False,
                 n_bits=3,

                 ):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation = 1,
                         groups = 1,
                         bias=False,
                         )
        self.n_bits = n_bits

    def forward(self, x):
        weight_n = Weight_fp.apply(self.weight, self.n_bits)
        # if torch.isnan(weight_n).any():
        #     print("Nan in weight")
        x_old= x
        x = self._conv_forward(x, weight_n, self.bias)
        x_for = x
        x = Feature_fp.apply(x, self.n_bits)
        if torch.isnan(x).any():
            print("Nan in feature")
        return x

class Conv2d_fp8_hw(nn.Conv2d):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 dilation = 1,
                 groups: int = 1,
                 bias=False,
                 n_bits=3,
                 quant_type = 'None',
                 group_number = 72,
                 left_shift_bit=0
                 ):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation = 1,
                         groups = 1,
                         bias=False,
                         )
        self.n_bits = n_bits
        self.quant_type = quant_type
        self.group_number = group_number
        self.left_shift_bit = left_shift_bit

    def forward(self, x):
        # weight_n_t = Weight_fp.apply(self.weight, self.n_bits)
        # if torch.isnan(weight_n_t).any():
        #     print("Nan in weight")
        # x_old_t = x
        # x_t = self._conv_forward(x_old_t, weight_n_t, self.bias)
        # x_for_t = x_t
        # x_t = Feature_fp.apply(x_t, self.n_bits)
        # if torch.isnan(x_t).any():
        #     print("Nan in feature")

        weight_n = Weight_fp_hw.apply(self.weight, self.n_bits, self.quant_type, self.group_number, self.left_shift_bit)
        x_init = x
        # x = Feature_fp_hw.apply(x, self.n_bits, self.quant_type, self.group_number)
        # if torch.isnan(weight_n).any():
        #     print("Nan in weight")
        x = self._conv_forward(x, weight_n, self.bias)
        x = Feature_fp_hw.apply(x, self.n_bits, "none", 1, self.left_shift_bit)
        # if torch.isnan(x).any():
        #     print("Nan in feature")

        # if not torch.equal(weight_n_t, weight_n):
        #     # 找到不相等的元素
        #     unequal_elements = torch.ne(weight_n_t, weight_n)
        #
        #     # 获取不相等元素的位置
        #     unequal_positions = unequal_elements.nonzero(as_tuple=True)
        #
        #     # 获取不相等元素的个数
        #     num_unequal_elements = unequal_elements.sum().item()
        #
        #     print(f"Number of unequal elements: {num_unequal_elements}")
        #     print(f"Positions of unequal elements: {unequal_positions}")
        #
        # if not torch.equal(x_t,x):
        #     # 找到不相等的元素
        #     unequal_elements = torch.ne(x_t, x)
        #
        #     # 获取不相等元素的位置
        #     unequal_positions = unequal_elements.nonzero(as_tuple=True)
        #
        #     # 获取不相等元素的个数
        #     num_unequal_elements = unequal_elements.sum().item()
        #
        #     print(f"Number of unequal elements: {num_unequal_elements}")
        #     print(f"Positions of unequal elements: {unequal_positions}")

        return x


# logger_config.py
import logging
import sys

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():  # 忽略空行
            self.level(message.strip())

    def flush(self):
        pass

def setup_logger(name='FP8Logger', log_file='fp8_alignment.log', level=logging.DEBUG):
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # 创建格式化器并添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 如果没有处理器，添加它们
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # 重定向 print 到 logger
    sys.stdout = LoggerWriter(logger.info)

    return logger

def initialize_weights_fp8e4m3(shape, min_val=-448, max_val=448):
    scale = (max_val - min_val) / 255
    weights = torch.rand(shape) * (max_val - min_val) + min_val
    quantized_weights = torch.round(torch.log2(weights - min_val + 1) / scale) * scale + min_val
    return quantized_weights