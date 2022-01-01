#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""
import random
import copy
import numpy as np
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# This part implement the quantization and dequantization operations.
# The output of the encoder must be the bitstream.
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out

        
class MyLoss(nn.Module):
    def __init__(self, reduce=True):
        super().__init__()
        self.eps = 1e-6
        self.reduce = reduce
    
    def forward(self, pre, label):
        pre = pre - 0.5
        label = label - 0.5 
        dif = (pre - label).pow(2).sum(3).sum(2).sum(1)
        base = label.pow(2).sum(3).sum(2).sum(1)
        # base = torch.clamp(base, self.eps)
        loss = dif / base
        if self.reduce:
            loss = loss.mean()
        return loss
        

class MLP(nn.Module):
    def __init__(self, indim, hidden, act=nn.GELU):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(indim, hidden),
            act(),
            nn.Linear(hidden, indim),
        )
        
    def forward(self, x):
        return self.fc(x)
        
        
class MixerBlock(nn.Module):
    def __init__(self, token_dim, dim, ratio=(1., 1.),
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp1 = MLP(token_dim, int(token_dim * ratio[0]))
        self.mlp2 = MLP(dim, int(dim * ratio[1]))
        
    def forward(self, x):
        x = x + self.mlp1(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.mlp2(self.norm2(x))
        return x
        
        
class ChannelLinear(nn.Module):
    def __init__(self, c, dim, outdim):
        super().__init__()
        self.fc = nn.Linear(dim, outdim)
        self.c = c
        
    def forward(self, x):
        if self.c != len(x.shape):
            x = x.transpose(self.c, -1)
        x = self.fc(x)
        if self.c != len(x.shape):
            x = x.transpose(self.c, -1).contiguous()
        return x
        

class Encoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits):
        super(Encoder, self).__init__()
        self.encoder_blocks = nn.Sequential(
            nn.Linear(256, 128),
            MixerBlock(126, 128, (4., 4.)),
            ChannelLinear(1, 126, 56),
            MixerBlock(56, 128, (4., 4.)),
            MixerBlock(56, 128, (4., 4.)),
            nn.Linear(128, 32),
        )
        self.fc = nn.Linear(56 * 32, int(feedback_bits // self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)

    def forward(self, x):
        b = x.size(0)
        x = x - 0.5
        x = x.permute(0, 2, 3, 1).contiguous().view(b, 126, 256)
        x = self.encoder_blocks(x)
        x = x.view(b, -1)
        out = self.fc(x)
        out = self.sig(out)
        out = self.quantize(out)
        return out


class Decoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        self.fc = nn.Linear(int(feedback_bits // self.B), 56*32)
        self.decoder_blocks = nn.Sequential(
            nn.Linear(32, 256),
            MixerBlock(56, 256, (4., 4.)),
            ChannelLinear(1, 56, 126),
            MixerBlock(126, 256, (4., 4.)),
            MixerBlock(126, 256, (4., 4.)),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        
    def forward(self, x):
        b = x.size(0)
        out = self.dequantize(x)
        out = out.view(-1, int(self.feedback_bits // self.B))
        out = self.fc(out)
        out = out.view(-1, 56, 32)
        out = self.decoder_blocks(out)
        out = out * 0.5 + 0.5
        out = out.view(b, 126, 128, 2).permute(0, 3, 1, 2).contiguous()
        return out
        

class DatasetFolderTrain(Dataset):

    def __init__(self, matData, training=True):
        self.matdata = matData
        self.training = training

    def __getitem__(self, index):
        if self.training:
            data = copy.deepcopy(self.matdata[index])
            if random.random() < 0.2:
                data = 1 - data
            if random.random() < 0.2:
                data = data.reshape(2, 126, 2, 64)
                data = np.concatenate([data[:, :, 1], data[:, :, 0]], -1)
            if random.random() < 0.2:
                data = np.concatenate([data[1:2], data[0:1]], 0)
            return data
            
        return self.matdata[index]

    def __len__(self):
        return self.matdata.shape[0]


# Note: Do not modify following class and keep it in your submission.
# feedback_bits is 512 by default.
class AutoEncoder(nn.Module):

    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


def Score(NMSE):
    score = 1 - NMSE
    return score


# dataLoader
class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __getitem__(self, index):
        return self.matdata[index]

    def __len__(self):
        return self.matdata.shape[0]
