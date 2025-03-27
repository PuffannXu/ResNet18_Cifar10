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
                          reg_shift_mode=True, reg_shift_bits=None):
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
        scale = data_float.abs().max().detach() + 1e-8 # 动态计算但不参与梯度

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
                 shift_mode = True
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
        self.shift_mode = shift_mode

    def forward(self, x):
        # quantize weight and add noise first
        if self.qn_on:
            # 先对输入进行量化，放缩到 127 大小
            x = x * self.output_half_level + torch.randn_like(x)
            # 然后进行round取整
            x = round_pass(x)
            # x = x.round()
            # 对权重进行量化，用整型表示
            weight_q, scale = data_quantization_sym(self.weight, self.weight_half_level, isint=1, clamp_std=self.clamp_std,reg_shift_mode=self.shift_mode)
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
                bias_q, _ = data_quantization_sym(self.bias, self.weight_half_level, isint=1, clamp_std=0,reg_shift_mode=self.shift_mode)
            else:
                bias_q = self.bias
            # calculate the convolution next
            # x_np0=x.cpu().detach().numpy()
            # x_2d0 = x_np0.reshape(N, -1).transpose(1,0)
            # np.savetxt(f'in{weight_q.shape}_{weight_q[0][0][0][0]}.csv', x_2d0, delimiter=',')
            x = self._conv_forward(x, weight_q, bias_q)
            # 对输出的整型结果进行移位
            if self.shift_mode:
                aver = x.max() #注意不是abs
                post_shift = int(np.ceil(math.log2(aver / (self.output_half_level + 1))))
                # # x_2d1 = x.reshape(N, -1).transpose(1,0).cpu().detach().numpy()
                # # np.savetxt(f'x_before{weight_q.shape}_{weight_q[0][0][0][0]}.csv', x_2d1, delimiter=',')
                x = floor_pass(x/(2 ** post_shift))
            else:
                scale = x.abs().max().detach()
                x = round_pass(x / scale * self.output_half_level)

            x = x / self.output_half_level
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
                 bias=False,
                 shift_mode=True):
        super().__init__(in_features, out_features, bias)
        self.qn_on = qn_on
        self.weight_bit = weight_bit
        self.output_bit = output_bit
        self.isint = isint
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.output_half_level = 2 ** output_bit / 2 - 1
        self.shift_mode = shift_mode

    def forward(self, x):
        if self.qn_on:
            # 先对输入进行量化，放缩到 127 大小
            x = x * self.output_half_level
            # 然后进行round取整
            x = round_pass(x)

            # 对权重进行量化，用整型表示
            weight_q, scale = data_quantization_sym(self.weight, self.weight_half_level, isint=1, clamp_std=self.clamp_std, reg_shift_mode=self.shift_mode)
            # cout, cin = weight_q.shape
            # N = cout  # N = cout
            # weights_2d = weight_q.cpu().detach().numpy()
            #
            # np.savetxt(f'weight{weight_q.shape}_{weight_q[0][0]}.csv', weights_2d, delimiter=',')

            if self.bias != None:
                # bias = bias / scale
                bias_q, _ = data_quantization_sym(self.bias, self.weight_half_level, isint=1, clamp_std=0,reg_shift_mode=self.shift_mode)
            else:
                bias_q = self.bias
            # calculate the convolution next

            # x_2d0 = x.cpu().detach().numpy()
            # np.savetxt(f'in{weight_q.shape}_{weight_q[0][0]}.csv', x_2d0, delimiter=',')

            x = F.linear(x, weight_q, bias_q)

            if self.shift_mode:
                aver = x.max()  # 注意不是abs
                if (aver > (self.output_half_level - 1)):
                    left_shift = 0
                    post_shift = int(np.ceil(math.log2(aver / (self.output_half_level + 1))))
                else:
                    left_shift = 1
                    post_shift = int(np.ceil(math.log2((self.output_half_level + 1) / (0.1 + aver))))
                # # x_2d1 = x.reshape(N, -1).transpose(1,0).cpu().detach().numpy()
                # # np.savetxt(f'x_before{weight_q.shape}_{weight_q[0][0][0][0]}.csv', x_2d1, delimiter=',')
                x = floor_pass(x / (2 ** post_shift))
            else:
                scale = x.abs().max().detach()
                x = round_pass(x / scale * self.output_half_level)

            x = x / self.output_half_level
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
    e_float = (2 ** (e_4bit-7.0)).float()
    out = (-1)**sign*e_float*m_float*mask
    return out

def fp8_alignment(ifm_uint8, left_shift_bit=3):
    device = ifm_uint8.device  # 获取输入张量的设备
    mask = torch.where(ifm_uint8 == 0, 0, 1).to(device)
    # 最高位和低3位
    m_plus = (torch.ones(ifm_uint8.shape, dtype=torch.uint8, device=device) & 0b00000001) << 3 | (ifm_uint8 & 0b00000111)
    m_plus = m_plus * mask
    m = ifm_uint8 & 0b00000111
    m = m * mask

    s = (ifm_uint8 & 0b10000000) >> 7
    # 低4~7位
    e = (ifm_uint8 & 0b01111000) >> 3
    # 每行的最大值
    pre_macro_data_e_max, _ = torch.max(e, dim=1)
    e_delta = pre_macro_data_e_max.unsqueeze(1) - e
    m_shifted = (m << left_shift_bit) >> e_delta
    m_plus_shifted = (m_plus << left_shift_bit) >> e_delta
    m_shifted_signed = (-1)**s * m_plus_shifted
    e_max = (e + e_delta) * mask
    result = (s << (7+left_shift_bit)) + (e_max << (3+left_shift_bit)) + m_shifted

    return result, s, e_max, m_shifted, m_shifted_signed, pre_macro_data_e_max

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
        scaling_factor = weight_max/448#(1.875 * 2**(n_bits+left_shift_bit))
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
        K = ci * kx * ky
        total_elements = co * ci * kx * ky
        cout, cin, kernel_height, kernel_width = weight.shape

        # 6. 转换为二维形式 KxN
        K = cin * kernel_height * kernel_width  # K = cin * kernel_height * kernel_width
        N = cout  # N = cout
        if quant_type == 'channel':
            weight_reshape = weight_n_scale.reshape([co,-1])
            weight_align, sign, e_max, m_sft, m_plus, e = fp8_alignment(weight_reshape, left_shift_bit)
            # weight_align = weight_align.transpose(1,0)
            # sign=sign.transpose(1,0)
            # e_max=e_max.transpose(1,0)
            # m_sft =m_sft.transpose(1,0)
        elif quant_type == 'layer':
            weight_reshape = weight_n_scale.reshape([1,-1])
            weight_align, sign, e_max, m_sft, m_plus, e = fp8_alignment(weight_reshape, left_shift_bit)
            # weight_align = weight_align.transpose(1,0)
            # sign=sign.transpose(1,0)
            # e_max=e_max.transpose(1,0)
            # m_sft =m_sft.transpose(1,0)
        elif quant_type == 'group':
                        # 创建二维数组
            weights_2d_uint8 = weight_n_scale.permute(0, 3, 2, 1).reshape(N, K)

            # 判断前两个维度并进行填充
            if np.mod(weights_2d_uint8.shape[-1], 72)!=0:
                # weight_reshape = weights_2d_uint8.reshape(-1, K)
                padding_0 = (weights_2d_uint8.shape[-1] // 72 + 1) * 72 - weights_2d_uint8.shape[-1]
                weights_2d_uint8 = F.pad(input=weights_2d_uint8, pad=(0, padding_0, 0, 0), mode='constant')
            weight_reshape = weights_2d_uint8.reshape(-1, 72)
            weight_align, sign, e_max, m_sft, m_plus, e = fp8_alignment(weight_reshape, left_shift_bit)
        else:
            weight_reshape = weight_n_scale.reshape([-1, 1])
            weight_align, sign, e_max, m_sft, m_plus, e = fp8_alignment(weight_reshape, left_shift_bit)

        # weight_align = weight_align.reshape([co, -1])
        # e_max = e_max.reshape([co, -1])
        # m_sft = m_sft.reshape([co, -1])
        # sign = sign.reshape([co, -1])
        # m_plus = m_plus.reshape([co, -1])
        # e = e.reshape([co, -1])
        e_expanded = e.unsqueeze(1)
        fp_weight = torch.cat((m_plus, e_expanded), dim=1)
        fp_weight = fp_weight.reshape(N, -1)
        A_len = 73
        K_add = int(np.ceil(K / 73))
        macro_output_len = min(N, int(np.ceil(64 / K_add)))
        B_len = K_add * macro_output_len
        R_len = min(int(np.ceil(N / macro_output_len)), 128)
        if kernel_height == 1:
            C_len = int(np.ceil(R_len / 128))
        else:
            C_len = int(np.ceil(int(np.ceil(N / macro_output_len)) / 128))
        if([C_len, R_len, macro_output_len, K_add, A_len]==[1,128,2,32,73]):
            print()

        quantized_weights_reshape = fp_weight.reshape(C_len, R_len, macro_output_len, K_add, A_len)

        # 判断前两个维度并进行填充
        if quantized_weights_reshape.shape[-1] < 73:
            padding_0 = 73 - quantized_weights_reshape.shape[-1]
            quantized_weights_reshape = np.pad(quantized_weights_reshape, ((0, 0), (0, 0), (0, 0), (0, 0), (0, padding_0)), mode='constant')
        if B_len < 64:
            padding_1 = 64 // macro_output_len - quantized_weights_reshape.shape[-2]
            quantized_weights_reshape = np.pad(quantized_weights_reshape, ((0, 0), (0, 0), (0, 0), (0, padding_1), (0, 0)), mode='constant')
        # else:
        #     if B_len < 64:
        #         padding_1 = 64 // K_add - quantized_weights_reshape.shape[-3]
        #         quantized_weights_reshape = np.pad(quantized_weights_reshape, ((0, 0), (0, 0), (0, padding_1), (0, 0), (0, 0)), mode='constant')

        ww = quantized_weights_reshape.reshape((C_len, R_len, 64, 73)).permute(1, 2, 0, 3)
        ww = ww.reshape(R_len, 64, C_len * 73)
        ww = ww.permute(0, 2, 1)
        # 使用列表推导式得到73的倍数的列索引和非73倍数的列索引
        indices_73 = [i - 1 for i in range(1, ww.shape[1] + 1) if (i % 73) == 0]
        indices_not_73 = [i - 1 for i in range(1, ww.shape[1] + 1) if (i % 73) != 0]
        # 使用列索引取出对应的列
        e_fp8_r = ww[:, indices_73, :].type(torch.uint8)
        m_fp8_r = ww[:, indices_not_73, :]
        e_fp8 = e_fp8_r[:, 0, :]
        m_fp8 = m_fp8_r[:, 0:72, :]
        for i in range(e_fp8_r.shape[1] - 1):
            e_fp8 = torch.cat((e_fp8, e_fp8_r[:, i + 1, :]), dim=0)
            m_fp8 = torch.cat((m_fp8, m_fp8_r[:, (i + 1) * 72:(i + 2) * 72, :]), dim=0)
        # e_fp8 = e_fp8_r.reshape(-1, 64)
        # e_fp8 = e_fp8_r.reshape(weight_row_end_grp - weight_row_start_grp + 1, -1, 1, 64 // K_add, K_add)
        # e_fp8 = e_fp8.transpose(1, 0, 3, 4, 2).squeeze(-1)  # K_add as the last
        # e_fp8_max = np.zeros((e_fp8.shape[0], e_fp8.shape[1], e_fp8.shape[2])).astype(np.uint8)
        # delta_fp8_max = np.zeros((e_fp8.shape[0], e_fp8.shape[1], e_fp8.shape[2], e_fp8.shape[3])).astype(np.uint8)
        # m_signed_uint8 = torch.tensor([[[x if x < 16 * 8 else x - 256 for x in col] for col in row] for row in m_fp8])

        weight_align_fp = uint8_to_fp32(weight_align, sign, e_max, m_sft, n_bits, left_shift_bit=left_shift_bit)
        weight_align_fp_out = (weight_align_fp - q_min) * scaling_factor + weight_min

        return weight_align_fp_out,weight_align_fp, m_fp8, e_fp8

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
        scaling_factor = 1#feature_max / 448#(1.875 * 2**(n_bits+left_shift_bit))
        feature_scale = feature / scaling_factor

        feature_n = fp8_downcast(feature_scale, n_bits)
        co, ci, kx, ky = feature_n.shape
        total_elements = co * ci * kx * ky
        if quant_type == 'channel':
            feature_reshape = feature_n.reshape([co,-1])
            feature_align, sign, e_max, m_sft, m_plus, e = fp8_alignment(feature_reshape, left_shift_bit=left_shift_bit)

        elif quant_type == 'layer':
            feature_reshape = feature_n.reshape([1,-1])
            feature_align, sign, e_max, m_sft, m_plus, e = fp8_alignment(feature_reshape, left_shift_bit=left_shift_bit)

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
            feature_align, sign, e_max, m_sft, m_plus, e = fp8_alignment(feature_reshape, left_shift_bit)
        else:
            feature_reshape = feature_n.reshape([-1, 1])
            feature_align, sign, e_max, m_sft, m_plus, e = fp8_alignment(feature_reshape, left_shift_bit)
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
        # # 计算绝对误差
        # absolute_error = torch.abs(feature_align_fp_out - feature)
        # # 避免除以零的情况
        # epsilon = 1e-10
        # # 计算误差百分比
        # zero_mask = (feature != 0.0)
        # error_percentage = (absolute_error / (torch.abs(feature) + epsilon)) * 100 * zero_mask
        # error_percentage_max = torch.max(error_percentage)
        # max_index = torch.argmax(error_percentage)
        # d0, d1, d2, d3 = error_percentage.shape
        #
        # i = torch.div(max_index, (d1 * d2 * d3), rounding_mode='floor')
        # j = torch.div(max_index % (d1 * d2 * d3), (d2 * d3), rounding_mode='floor')
        # k = torch.div(max_index % (d2 * d3), d3, rounding_mode='floor')
        # l = max_index % d3
        # # print(error_percentage[i, j, k, l], feature_align_fp_out[i, j, k, l], feature[i, j, k, l])
        # # 计算平均误差百分比
        # mean_error_percentage = torch.mean(error_percentage).item()
        # # print(f'对称平均误差百分比-lfs{left_shift_bit}: {mean_error_percentage:.2f}%')
        # max_count = torch.sum(error_percentage == error_percentage_max)
        # # 计算总元素个数
        # total_elements = error_percentage.numel()
        # # 计算最大值的占比
        # max_ratio = max_count.float() / total_elements
        return feature_align_fp_out,feature_align_fp, m_plus, e

    @staticmethod
    def backward(ctx, feature_grad):
        return feature_grad, None, None, None, None

import torch
import torch.nn.functional as F

def im2col_tensor(input_data, kernel, stride=1, pad=0):
    """
    将输入张量转换为二维展开的列形式，以便进行卷积运算。

    Parameters
    ----------
    input_data : torch.Tensor
        输入张量，形状为 (N, C, W, H)，其中N是批大小，C是通道数，W是宽度，H是高度。
    kernel : int
        卷积核的尺寸（假设为正方形）。
    stride : int, optional
        卷积步幅，默认为1。
    pad : int, optional
        输入周围的填充大小，默认为0。

    Returns
    -------
    torch.Tensor
        二维张量，形状为 (N*out_h*out_w, C*kernel*kernel)，每一行对应一个卷积窗口的展平数据。
    """
    N, C, W, H = input_data.shape
    out_h = (H + 2 * pad - kernel) // stride + 1
    out_w = (W + 2 * pad - kernel) // stride + 1

    # 对输入进行填充，处理宽度和高度维度
    img = F.pad(input_data, (pad, pad, pad, pad), mode='constant', value=0)

    # 初始化展开后的张量
    col = torch.zeros((N, C, kernel, kernel, out_w, out_h),
                      dtype=input_data.dtype, device=input_data.device)

    # 填充每个卷积核位置的数据
    for y in range(kernel):
        y_max = y + stride * out_h
        for x in range(kernel):
            x_max = x + stride * out_w
            col[:, :, x, y, :, :] = img[:, :, x:x_max:stride, y:y_max:stride]

    # 调整维度顺序并展平
    col_permuted = col.permute(0, 5, 4, 2, 3, 1)  # 维度变为 (N, out_h, out_w, kernel, kernel, C)
    col_out = col_permuted.reshape(N * out_h * out_w, -1)  # 展平为二维张量

    return col_out

class Feature_in_fp_hw(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feature, n_bits, channel_out_size, kernel_size, stride, padding_size, left_shift_bit=3):
        # quant type can be none, layer, channel, group
        ctx.save_for_backward()
        _, channel_in_size, col_max, row_max = feature.shape
        K = kernel_size ** 2 * (channel_in_size)
        N = int(channel_out_size)
        K_add = int(np.ceil(K / 72))
        N_max = int(64 / K_add)
        macro_data_len = min(N, N_max)
        macro_output_time = int(N / macro_data_len)
        # feature_pre
        feature_pre_data_fp8 = im2col_tensor(feature, kernel_size, stride, padding_size)  # 输入展开成矩阵
        feature_pre_data_uint8 = fp8_downcast(feature_pre_data_fp8, n_bits)
        feature_align, sign, e_max, m_sft, pre_macro_data_uint8, pre_macro_data_e_max = fp8_alignment(feature_pre_data_uint8, left_shift_bit)
        # pre_macro_data_sign_uint8 = np.array([[x if x < 16 * 8 else 16 * 8 - x for x in row] for row in pre_macro_data_uint8])

        # feature_align = feature_align.reshape([channel_out_size, channel_in_size, col_max, row_max])
        # e_max = e_max.reshape([channel_out_size, channel_in_size, col_max, row_max])
        # m_sft = m_sft.reshape([channel_out_size, channel_in_size, col_max, row_max])
        # sign = sign.reshape([channel_out_size, channel_in_size, col_max, row_max])
        feature_align_fp = uint8_to_fp32(feature_align, sign, e_max, m_sft, n_bits, left_shift_bit=left_shift_bit)
        a = feature_align_fp.cpu().detach().numpy()
        # feature_align_fp_out = feature_align_fp * scaling_factor

        if torch.isnan(feature_align_fp).any():
            print("Nan in feature_align_fp_out")

        return feature_align_fp, pre_macro_data_uint8, pre_macro_data_e_max

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
        self.channel_out_size = out_channels
        self.n_bits = n_bits
        self.quant_type = quant_type
        self.group_number = group_number
        self.left_shift_bit = left_shift_bit
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_size = padding

        self.K = kernel_size ** 2 * (in_channels)
        self.N = int(out_channels)
        self.K_add = int(np.ceil(self.K / 72))
        self.N_max = int(64 / self.K_add)
        self.macro_data_len = min(self.N, self.N_max)
        self.macro_output_time = int(self.N / self.macro_data_len)
    def forward(self, x):
        # 输入由fp变成m+e
        _, _, col_max, row_max = x.shape
        col_max_w = int((col_max+self.padding_size*2-self.kernel_size+1)/self.stride)
        row_max_w = int((col_max+self.padding_size*2-self.kernel_size+1)/self.stride)
        xin, xinm_plus, xine = Feature_in_fp_hw.apply(x, self.n_bits, self.channel_out_size, self.kernel_size, self.stride, self.padding_size, self.left_shift_bit)
        # pre_macro_data_sign_uint8 = torch.tensor([[x if x < 16 * 8 else 16 * 8 - x for x in row] for row in xinm_plus])
        pre_macro_data_sign_uint8 = xinm_plus
        pre_macro_data_sign_uint8_np = pre_macro_data_sign_uint8.cpu().detach().numpy()
        weight_n, weight_align_fp, wm_plus, we = Weight_fp_hw.apply(self.weight, self.n_bits, self.quant_type, self.group_number, self.left_shift_bit)
        # xx = self._conv_forward(xin, weight_n, self.bias)
        # xinm_plus = xinm_plus.float()
        # wm_plus = wm_plus.float()
        # xx_m = torch.mm(xinm_plus, wm_plus.t())
        # tree_data = xx_m
        # tree_e_max = we.t().repeat(xinm_plus.shape[0], 1)
        # 对 pre_macro_data_sign_uint8 进行填充，使其最后一个维度可以被72整除
        padding_width = (-pre_macro_data_sign_uint8.shape[-1]) % 72
        if padding_width > 0:
            pre_macro_data_sign_uint8 = F.pad(pre_macro_data_sign_uint8, (0, padding_width), 'constant', 0)
        # 根据形状调整 pre_macro_data_sign_uint8
        if pre_macro_data_sign_uint8.shape[-1] > 72:
            pre_macro_data_sign_uint8 = pre_macro_data_sign_uint8.view(col_max_w * row_max_w, self.K_add, 72)
        else:
            pre_macro_data_sign_uint8 = pre_macro_data_sign_uint8.view(col_max_w * row_max_w, self.K_add, -1)
            pad_size = 72 - pre_macro_data_sign_uint8.shape[-1]
            if pad_size > 0:
                pre_macro_data_sign_uint8 = F.pad(pre_macro_data_sign_uint8, (0, pad_size), 'constant', 0)
        bank_data = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wm_plus = wm_plus.to(device).type(torch.float)
        pre_macro_data_sign_uint8 = pre_macro_data_sign_uint8.to(device).type(torch.float)
        for i0, c0 in enumerate(wm_plus):
            for i1, c1 in enumerate(pre_macro_data_sign_uint8):
                bank_data.append([torch.dot(c1[i % (c1.shape[0])], c0[:, i]) for i in range(64)])
        # 获取尺寸
        # N0 = wm_plus.size(0)
        # N1 = pre_macro_data_sign_uint8.size(0)
        # K_add = pre_macro_data_sign_uint8.size(1)
        # # 创建索引
        # indices = torch.arange(64, device=pre_macro_data_sign_uint8.device) % K_add  # (64,)
        # indices_expanded = indices.unsqueeze(0).expand(N1, -1)  # (N1, 64)
        # # 选择 c1
        # c1_selected = pre_macro_data_sign_uint8.gather(1, indices_expanded.unsqueeze(-1).expand(-1, -1, 72))  # (N1, 64, 72)
        # # 转置 c0
        # c0_transposed = wm_plus.permute(0, 2, 1)  # (N0, 64, 72)
        # # 扩展维度以进行广播
        # c0_broadcasted = c0_transposed.unsqueeze(1).float()  # (N0, 1, 64, 72)
        # c1_broadcasted = c1_selected.unsqueeze(0).float()  # (1, N1, 64, 72)
        # # 计算点积
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # c0_broadcasted = c0_broadcasted.to(device)
        # c1_broadcasted = c1_broadcasted.to(device)
        # bank_data = (c0_broadcasted * c1_broadcasted).sum(-1)  # (N0, N1, 64)
        # 得到 tree_data
        tree_data = torch.tensor(bank_data).to(device)
        # tree_data_np = tree_data.cpu().detach().numpy()# 根据需要转换数据类型
        # 处理 tree_e_max
        tree_e_max = we.repeat(pre_macro_data_sign_uint8.size(0), *([1] * (we.dim() - 1))).clone()
        # tree_e_max = we.expand(N1, -1).clone()

        loop_cnt = 0
        while True:
            # 定义当前的上限 U 和下限 L
            U = 2 ** (22 + loop_cnt) - 1
            L = -2 ** (22 + loop_cnt)

            # 初始元素条件调整
            mask_zero = (tree_data == 0)
            tree_e_max[mask_zero] = 0

            mask_nonzero = ~mask_zero
            s = tree_data[mask_nonzero]
            e = tree_e_max[mask_nonzero]

            abs_s = torch.abs(s)
            # 合并运算步骤（保持梯度传播）
            k = torch.floor(torch.log2(U / abs_s)) - 1
            # 将 min 转换为与 e 同类型的张量
            min_tensor = torch.tensor(0, dtype=e.dtype, device=e.device)
            k = k.to(device)
            k = k.clamp(min=min_tensor, max=e)

            # 类型转换优化（兼容PyTorch语法）
            k = k.to(dtype=torch.uint8)  # 替代 .astype()

            # 更新 tree_e_max 和 tree_data
            tree_e_max[mask_nonzero] -= k
            tree_data[mask_nonzero] *= 2 ** k

            loop_cnt += 1

            # 主循环退出条件
            if tree_data.shape[1] <= 64 // self.K_add:
                break

            # 计算 tree_e_max_new，取相邻元素的最大值
            tree_e_max_even = tree_e_max[:, ::2]
            tree_e_max_odd = tree_e_max[:, 1::2]
            tree_e_max_new = torch.maximum(tree_e_max_even, tree_e_max_odd)

            # 计算 delta_e，用于缩放 tree_data
            delta_e_even = tree_e_max_new - tree_e_max_even
            delta_e_odd = tree_e_max_new - tree_e_max_odd
            delta_e_even = delta_e_even.to(dtype=torch.float)
            delta_e_odd = delta_e_odd.to(dtype=torch.float)
            # 获取相邻的 tree_data 对
            tree_data_even = tree_data[:, ::2]
            tree_data_odd = tree_data[:, 1::2]

            # 缩放并取整
            adj_tree_data_even = torch.floor(tree_data_even / (2 ** delta_e_even))
            adj_tree_data_odd = torch.floor(tree_data_odd / (2 ** delta_e_odd))

            # 计算新的 tree_data
            tree_data_new = adj_tree_data_even + adj_tree_data_odd

            # 更新 tree_data 和 tree_e_max
            tree_data = tree_data_new
            tree_e_max = tree_e_max_new

            # 元素条件调整，与初始调整类似
            mask_zero = (tree_data == 0)
            tree_e_max[mask_zero] = 0

            mask_nonzero = ~mask_zero
            s = tree_data[mask_nonzero]
            e = tree_e_max[mask_nonzero]

            abs_s = torch.abs(s)
            # 合并运算步骤（保持梯度传播）
            k = torch.floor(torch.log2(U / abs_s)) - 1
            # 将 min 转换为与 e 同类型的张量
            min_tensor = torch.tensor(0, dtype=e.dtype, device=e.device)
            k = k.clamp(min=min_tensor, max=e)

            # 类型转换优化（兼容PyTorch语法）
            k = k.to(dtype=torch.uint8)  # 替代 .astype()

            tree_e_max[mask_nonzero] -= k
            tree_data[mask_nonzero] *= 2 ** k

            loop_cnt += 1

        macro_post_data_reshape = tree_data.to(dtype=torch.int32)
        macro_post_e_max_reshape = tree_e_max.to(dtype=torch.uint8)
        macro_post_e_max = macro_post_e_max_reshape.reshape(-1, self.macro_data_len)
        pre_macro_e_max_reshape = xine.reshape(1, -1).repeat(self.macro_data_len * wm_plus.shape[0] , 1).reshape(self.macro_data_len, -1).transpose(1, 0)
        pre_macro_e_max = pre_macro_e_max_reshape.reshape(-1, self.macro_data_len)
        e_max_new = (pre_macro_e_max + macro_post_e_max).to(dtype=torch.int)

        # # macro_post_data_reshape = macro_post_data_signed.reshape(pre_macro_data_sign_uint8.shape[0], macro_output_time, macro_data_len)
        # macro_post_data_reshape = macro_post_data_align.reshape(-1, macro_data_len)
        # # macro_post_data_reshape = macro_post_data_uint8.reshape(macro_ou`tput_time * (col_max_w + 1) * (row_max_w + 1), macro_data_len)
        macro_post_data_align = macro_post_data_reshape.reshape(-1, self.macro_data_len)
        macro_post_data_signed = macro_post_data_align

        shape = macro_post_data_align.shape
        device = macro_post_data_align.device
        post_feature_data_e = torch.zeros(shape, dtype=torch.int32, device=device)
        post_feature_data_m = torch.zeros(shape, dtype=torch.int32, device=device)
        post_feature_data_s = torch.zeros(shape, dtype=torch.int32, device=device)
        first_1_bit = torch.full(shape, -1, dtype=torch.int32, device=device)

        pos_mask = (macro_post_data_align > 0)  # True/False 张量
        abs_val = macro_post_data_align[pos_mask].abs()  # 只取>0部分做绝对值

        # 计算满足 2^bit > abs_val 的最小 bit
        # 即 bit = ceil(log2(abs_val+1))，并限制在 [0,31] 内
        bit = torch.ceil(torch.log2(abs_val.float() + 1)).clamp(min=0, max=31).to(torch.int)

        # first_1_bit = bit - 1
        # 因为原始逻辑里是从 range(0,31) 循环，一旦 2^bit > abs(ele) 就 break，对应到此处
        first_1 = bit - 1

        # 原代码中：
        #  post_feature_data_e[i][j] = e_max_new[i][j] + (bit - 14 - 6)
        #  post_feature_data_m[i][j] = (abs(ele_signed) >> (bit - 4)) - 8
        #  bit - 7 + 3 = bit - 4
        e_pos = e_max_new[pos_mask] + (bit - 14 - 6)
        m_pos = (abs_val >> (bit - 4)) - 8
        # 原始逻辑: post_feature_data_s[i][j] = ele_signed<0
        # 但 pos_mask 已经是 >0，所以这里其实全为0
        # 如果需要区分真正的正/负，可以拆分出 negative_mask，再做相应处理

        # 将这些值回填进完整张量中
        post_feature_data_e[pos_mask] = e_pos
        post_feature_data_m[pos_mask] = m_pos
        post_feature_data_s[pos_mask] = 0  # (macro_post_data_align[pos_mask] < 0).to(torch.int)
        first_1_bit[pos_mask] = first_1

        # ----------------------------------------------------------------------
        #  2) 计算 exponent 的 left_shift / post_shift， 并将其映射到有效范围
        # ----------------------------------------------------------------------

        # 计算 aver = np.max(np.abs(post_feature_data_e)) (原文用 numpy，这里用 torch)
        aver_val = torch.max(torch.abs(post_feature_data_e)).item()
        if aver_val > 15:
            left_shift = False
            post_shift = int(aver_val - 15)
        else:
            left_shift = True
            post_shift = int(15 - aver_val)

        # 准备一个 e_f_shifted，用于存放 shift 后的 e
        post_feature_data_e_f_shifted = torch.zeros_like(post_feature_data_e, dtype=torch.int32)

        # 不想写二重循环时，可以先对“(e==0, m==0)就置 e_f_shifted=0”做一个布尔 mask
        mask_zero = (post_feature_data_e == 0) & (post_feature_data_m == 0)

        # 对非零元做 shift
        if left_shift:
            e_shifted = post_feature_data_e + post_shift
        else:
            e_shifted = post_feature_data_e - post_shift

        # 先整体替换，再把本来应该强制为0的地方复原为0
        post_feature_data_e_f_shifted = e_shifted.clone()
        post_feature_data_e_f_shifted[mask_zero] = 0

        # 接著做截断：若 <0 则 e=0,m=0；若 >15 则 e=15,m=7
        lt0_mask = (post_feature_data_e_f_shifted < 0)
        gt15_mask = (post_feature_data_e_f_shifted > 15)

        post_feature_data_e_f_shifted[lt0_mask] = 0
        post_feature_data_m[lt0_mask] = 0

        post_feature_data_e_f_shifted[gt15_mask] = 15
        post_feature_data_m[gt15_mask] = 7

        # 也可以再做一次 clamp
        post_feature_data_e_f_shifted = post_feature_data_e_f_shifted.clamp(0, 15)

        # ----------------------------------------------------------------------
        #  3) 将 exponent / mantissa / sign 打包成 uint8
        #     原代码: (post_feature_data_m < 0) * 128 + (post_feature_data_e_f_shifted) * 8 + post_feature_data_m
        #     这里可直接以 torch 形式合并
        # ----------------------------------------------------------------------
        sign_bit = (post_feature_data_m < 0).to(torch.int)  # 需要的“符号”信息
        post_feature_data_uint8 = sign_bit * 128 + (post_feature_data_e_f_shifted * 8) + post_feature_data_m
        post_feature_data_uint8 = post_feature_data_uint8.clamp(0, 255).to(torch.uint8)

        # 将其转换为最终的 FP16（或 FP32）
        post_feature_data_fp16 = uint8_to_fp32(post_feature_data_uint8, 3)  # 依赖原环境中的函数

        # 最后 reshape + transpose 回到原先需要的维度
        x = post_feature_data_fp16.reshape((1, row_max_w, col_max_w, self.channel_out_size)).permute(0, 3, 2, 1)


        # xx = torch.mm(xin, weight_n.t())
        # xxx = xx.reshape(self.macro_output_time, col_max_w,row_max_w, self.macro_data_len).permute(0, 3, 2, 1).reshape(
        #     (1, self.channel_out_size, col_max_w, row_max_w))
        # x, x_align_fp, xm_plus, xe = Feature_fp_hw.apply(xxx, self.n_bits, "none", 1, self.left_shift_bit)
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