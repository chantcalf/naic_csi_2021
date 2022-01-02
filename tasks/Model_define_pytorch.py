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
import copy
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class NumBiter(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.b = b
        base = 2 ** torch.arange(b)
        self.register_buffer("base", base.unsqueeze(0).unsqueeze(1).long())

    def to_bit(self, x):
        b, n = x.shape
        x = x.unsqueeze(-1) // self.base % 2
        return x.view(b, -1)

    def to_num(self, x):
        b = x.size(0)
        x = x.view(b, -1, self.b)
        x = x * self.base
        x = x.sum(-1)
        return x


class FloatBiter(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.b = b
        int_base = 2 ** torch.arange(2)
        dot_base = 2 ** torch.arange(1, b - 1)
        self.register_buffer("int_base", int_base.unsqueeze(0).long())
        self.register_buffer("dot_base", dot_base.unsqueeze(0).float())

    @torch.no_grad()
    def to_bit(self, x):
        x = x.unsqueeze(-1)
        int_x = torch.log2(x).long()
        int_code = int_x // self.int_base % 2
        float_x = x / (2 ** int_x).float() - 1
        float_code = (float_x * self.dot_base).long() % 2
        return torch.cat([int_code, float_code], -1)

    @torch.no_grad()
    def to_float(self, x):
        int_code, float_code = x[:, :2], x[:, 2:]
        int_x = (int_code * self.int_base).sum(-1)
        int_x = (2 ** int_x).float()
        float_x = (float_code.float() / self.dot_base).sum(-1)
        return (float_x + 1) * int_x

    def forward(self, x):
        bits = self.to_bit(x)
        x1 = self.to_float(bits)
        return (x1 - x).detach() + x


class VQ(nn.Module):
    def __init__(self, b, dim):
        super().__init__()
        k = 2 ** b
        self.k = k
        self.dim = dim
        embed = torch.randn(k, dim)
        self.embed = nn.Parameter(embed)
        self.num_biter = NumBiter(b)

    @torch.no_grad()
    def quant(self, x):
        dist = x.unsqueeze(-2) - self.embed.unsqueeze(0).unsqueeze(1)
        _, ind = dist.pow(2).sum(-1).min(-1)
        bits = self.num_biter.to_bit(ind)
        return bits

    @torch.no_grad()
    def dequant(self, x):
        x = self.num_biter.to_num(x)
        return F.embedding(x, self.embed)

    def forward(self, x):
        # x: (b, n, dim)
        dist = x.unsqueeze(-2) - self.embed.unsqueeze(0).unsqueeze(1)
        _, ind = dist.pow(2).sum(-1).min(-1)  # (b, n)
        qx = F.embedding(ind, self.embed)  # (b, n, dim)

        loss1 = (qx - x.detach()).pow(2).mean()
        loss2 = (qx.detach() - x).pow(2).mean()
        return (qx - x).detach() + x, loss1, loss2


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


class VQVAE(nn.Module):
    def __init__(self, vq_b=9, vq_dim=32, vq_len=56, s_bit=8):
        super().__init__()
        self.s_bit = s_bit
        self.sq = VQ(vq_b, vq_dim)
        self.f16 = FloatBiter(s_bit)

        self.encoder_blocks = nn.Sequential(
            nn.Linear(256, 128),
            MixerBlock(126, 128, (4., 4.)),
            ChannelLinear(1, 126, vq_len),
            MixerBlock(vq_len, 128, (4., 4.)),
            MixerBlock(vq_len, 128, (4., 4.)),
            nn.Linear(128, vq_dim),
        )

        self.decoder_blocks = nn.Sequential(
            nn.Linear(vq_dim, 256),
            MixerBlock(vq_len, 256, (4., 4.)),
            ChannelLinear(1, vq_len, 126),
            MixerBlock(126, 256, (4., 4.)),
            MixerBlock(126, 256, (4., 4.)),
            nn.Linear(256, 256),
            nn.Tanh()
        )

    def forward(self, x):
        b = x.size(0)
        x = x - 0.5
        x = x.permute(0, 2, 3, 1).contiguous().view(b, 126, 256)
        s = torch.norm(x.view(b, -1), dim=1)
        s = torch.clamp(s, 1., 7.99)
        s = self.f16(s)
        s = s.unsqueeze(-1).unsqueeze(-1)
        x = x / s
        x = self.encoder_blocks(x)
        qx, loss1, loss2 = self.sq(x)
        out = self.decoder_blocks(qx)
        out = out * s * 0.5 + 0.5
        out = out.view(b, 126, 128, 2).permute(0, 3, 1, 2).contiguous()
        if self.training:
            return out, loss1, loss2
        return out


class Encoder(nn.Module):

    def __init__(self, feedback_bits, vq_b=9, vq_dim=32, vq_len=56, s_bit=8):
        super(Encoder, self).__init__()
        self.s_bit = s_bit
        self.sq = VQ(vq_b, vq_dim)
        self.f16 = FloatBiter(s_bit)

        self.encoder_blocks = nn.Sequential(
            nn.Linear(256, 128),
            MixerBlock(126, 128, (4., 4.)),
            ChannelLinear(1, 126, vq_len),
            MixerBlock(vq_len, 128, (4., 4.)),
            MixerBlock(vq_len, 128, (4., 4.)),
            nn.Linear(128, vq_dim),
        )

    def quantize(self, x, s):
        bits_x = self.sq.quant(x)
        bits_s = self.f16.to_bit(s)
        return torch.cat([bits_s, bits_x], -1)

    def forward(self, x):
        b = x.size(0)
        x = x - 0.5
        x = x.permute(0, 2, 3, 1).contiguous().view(b, 126, 256)
        s = torch.norm(x.view(b, -1), dim=1)
        s = torch.clamp(s, 1., 7.99)
        s = self.f16(s)
        x = x / s.unsqueeze(-1).unsqueeze(-1)
        x = self.encoder_blocks(x)
        return self.quantize(x, s)


class Decoder(nn.Module):

    def __init__(self, feedback_bits, vq_b=9, vq_dim=32, vq_len=56, s_bit=8):
        super(Decoder, self).__init__()
        self.s_bit = s_bit
        self.sq = VQ(vq_b, vq_dim)
        self.f16 = FloatBiter(s_bit)

        self.decoder_blocks = nn.Sequential(
            nn.Linear(vq_dim, 256),
            MixerBlock(vq_len, 256, (4., 4.)),
            ChannelLinear(1, vq_len, 126),
            MixerBlock(126, 256, (4., 4.)),
            MixerBlock(126, 256, (4., 4.)),
            nn.Linear(256, 256),
            nn.Tanh()
        )

    def dequantize(self, x):
        s = self.f16.to_float(x[:, :self.s_bit])
        out = self.sq.dequant(x[:, self.s_bit:])
        return out, s

    def forward(self, x):
        b = x.size(0)
        out, s = self.dequantize(x)
        s = s.unsqueeze(-1).unsqueeze(-1)
        out = self.decoder_blocks(out)
        out = out * s * 0.5 + 0.5
        out = torch.clamp(out, 0., 1.)
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
