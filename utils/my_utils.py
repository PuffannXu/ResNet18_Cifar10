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




def data_quantization_sym(data_float, half_level=15, scale=None,
                          isint=0, clamp_std=0, boundary_refine=True,
                          reg_shift_mode=False, reg_shift_bits=None):
    # isint = 1 -> return quantized values as integer levels
    # isint = 0 -> return quantized values as float numbers with the same range as input
    # reg_shift_mode -> force half_level to be exponent of 2, i.e., half_level = 2^n (n is integer)

    if half_level <= 0:
        return data_float, 1

    if boundary_refine:
        half_level += 0.4999

    if clamp_std:
        std = data_float.std()
        data_float[data_float < (clamp_std * -std)] = (clamp_std * -std)
        data_float[data_float > (clamp_std * std)] = (clamp_std * std)

    if scale == None or scale == 0:
        scale = abs(data_float).max()

    if scale == 0:
        return data_float, 1

    if reg_shift_mode:
        if reg_shift_bits != None:
            quant_scale = 2 ** reg_shift_bits
        else:
            shift_bits = round(math.log(1 / scale * half_level, 2))
            quant_scale = 2 ** shift_bits
        data_quantized = (data_float * quant_scale).round()
        #print(f'quant_scale = {quant_scale}')
        #print(f'reg_shift_bits = {reg_shift_bits}')
    else:
        data_quantized = (data_float / scale * half_level).round()
        quant_scale = 1 / scale * half_level

    if isint == 0:
        data_quantized = data_quantized * scale / half_level
        quant_scale = 1

    return data_quantized, quant_scale


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
    def forward(ctx, feature, half_level, isint):
        # feature_q, _ = data_quantization_sym(feature, half_level, scale = None, isint = isint, clamp_std = 0)
        feature_q, _ = data_quantization_sym(feature, half_level,  isint=isint)
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
            weight_q, bias_q = Weight_Quant.apply(self.weight, self.bias,
                                                        self.weight_half_level, self.isint, self.clamp_std)
            # calculate the convolution next
            x = self._conv_forward(x, weight_q, bias_q)

            # quantize the output feature map at last
            x = Feature_Quant.apply(x, self.output_half_level, self.isint)
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
            weight_q, bias_q = Weight_Quant_Noise.apply(self.weight, self.bias,
                                                        self.weight_half_level, self.isint, self.clamp_std,
                                                        self.noise_scale)
            x = F.linear(x, weight_q, bias_q)
            x = Feature_Quant.apply(x, self.output_half_level, self.isint)
        else:
            x = F.linear(x, self.weight, self.bias)

        return x


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


def uint8_to_fp32(source_tensor: torch.ByteTensor, n_bits: int):
    """Converts a uint8 tensor to a fp16 tensor

    Args:
        source_tensor (torch.ByteTensor): The tensor to convert
        n_bits (int): The number of bits to use for the mantissa

    Returns:
        _type_: The converted tensor
    """
    if source_tensor.dtype != torch.uint8:
        source_tensor = source_tensor.clip(-480, 480)
    source_tensor = source_tensor.clone().detach().to(torch.uint8)
    mask = torch.where(source_tensor == 0, 0, 1)

    source_m_nbits = n_bits
    source_e_nbits = 7 - n_bits

    # Extract sign as int16
    sign = shift_right(source_tensor, 7)
    shifted_sign = shift_left(sign.type(torch.int16), 15)

    # Extract base as int16 and adjust the bias accordingly
    base_mantissa = shift_left(source_tensor, 1)
    base = shift_right(base_mantissa, source_m_nbits + 1) - shift_base(source_e_nbits)
    base = base.type(torch.int16) + shift_base(5)
    shifted_base = shift_left(base, 10)

    # Extract mantissa as int16
    mantissa = shift_left(base_mantissa, source_e_nbits)
    shifted_mantissa = shift_left(mantissa.type(torch.int16), 2)
    out = mask*(shifted_base + shifted_sign + shifted_mantissa).view(torch.float16).float()
    return out

class Weight_fp(torch.autograd.Function):
    # Number of inputs (excluding ctx, only weight, bias, half_level, isint, clamp_std, noise_scale)
    # for forward need to be the same as the number of return in def backward()
    # (return weight_grad, bias_grad, None, None, None, None)
    @staticmethod
    def forward(ctx, weight, n_bits):
        ctx.save_for_backward()

        weight_n = fp8_downcast(weight, n_bits)
        weight_n = uint8_to_fp32(weight_n, n_bits)

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
        # feature_q, _ = data_quantization_sym(feature, half_level, scale = None, isint = isint, clamp_std = 0)
        feature_q = fp8_downcast(feature, n_bits)
        feature_q = uint8_to_fp32(feature_q, n_bits)
        return feature_q

    @staticmethod
    def backward(ctx, feature_grad):
        return feature_grad, None, None, None
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
        if torch.isnan(weight_n).any():
            print("Nan in weight")
        x = self._conv_forward(x, weight_n, self.bias)
        x = Feature_fp.apply(x, self.n_bits)
        if torch.isnan(x).any():
            print("Nan in feature")
        return x