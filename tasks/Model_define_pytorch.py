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
        dot_base = 2 ** torch.arange(0, b)
        self.register_buffer("dot_base", dot_base.unsqueeze(0).float())

    @torch.no_grad()
    def to_bit(self, x):
        x = x.clamp(1., 7.38)
        x = x.unsqueeze(-1)
        x = torch.log(x)
        x = (x * self.dot_base).long() % 2
        return x

    @torch.no_grad()
    def to_float(self, x):
        x = (x.float() / self.dot_base).sum(-1)
        return torch.exp(x)

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
        dist = self.dist(x.unsqueeze(-2), self.embed.unsqueeze(0).unsqueeze(1))
        _, ind = dist.min(-1)  # (b, n)
        qx = F.embedding(ind, self.embed)  # (b, n, dim)

        loss1 = self.dist(qx, x.detach()).mean()
        loss2 = self.dist(qx.detach(), x).mean()
        return (qx - x).detach() + x, loss1, loss2

    @staticmethod
    def dist(x, y):
        return (x - y).pow(2).mean(-1)


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, pre, label):
        # pre = pre - 0.5
        # label = label - 0.5
        b = pre.size(0)
        pre = pre.view(b, -1)
        label = label.view(b, -1)
        dif = (pre - label).pow(2).sum(-1)
        base = label.pow(2).sum(-1)
        # base = torch.clamp(base, self.eps)
        loss = dif / base
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
    def __init__(self, dim, outdim):
        super().__init__()
        self.fc = nn.Linear(dim, outdim)

    def forward(self, x):
        x = self.fc(x.transpose(1, 2)).transpose(1, 2)
        return x


class EncoderModel(nn.Module):
    def __init__(self, vq_dim=8, vq_len=63):
        super().__init__()
        self.encoder_blocks = nn.Sequential(
            nn.Linear(256, 128),
            MixerBlock(126, 128, (4., 4.)),
            ChannelLinear(126, vq_len),
            MixerBlock(vq_len, 128, (4., 4.)),
            MixerBlock(vq_len, 128, (4., 4.)),
            MixerBlock(vq_len, 128, (4., 4.)),
            nn.Linear(128, vq_dim),
        )

    def forward(self, x):
        return self.encoder_blocks(x)


class DecoderModel(nn.Module):
    def __init__(self, vq_b=8, vq_dim=8, vq_len=63):
        super().__init__()
        self.sq = VQ(vq_b, vq_dim)
        self.decoder_blocks = nn.Sequential(
            nn.Linear(vq_dim, 256),
            MixerBlock(vq_len, 256, (4., 4.)),
            ChannelLinear(vq_len, 126),
            MixerBlock(126, 256, (4., 4.)),
            MixerBlock(126, 256, (4., 4.)),
            MixerBlock(126, 256, (4., 4.)),
            MixerBlock(126, 256, (4., 4.)),
            nn.Linear(256, 256),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder_blocks(x) - 0.5


class VQVAE(nn.Module):
    def __init__(self, vq_b=8, vq_dim=8, vq_len=63, s_bit=8):
        super().__init__()
        self.s_bit = s_bit
        self.f16 = FloatBiter(s_bit)
        self.encoder = EncoderModel(vq_dim, vq_len)
        self.decoder = DecoderModel(vq_b, vq_dim, vq_len)
        self.criterion = MyLoss()

    def preprocess(self, x):
        b = x.size(0)
        x = x - 0.5
        x = x.permute(0, 2, 3, 1).contiguous().view(b, 126, 256)
        s = torch.norm(x.view(b, -1), dim=1)
        return x, s

    def forward_train(self, x):
        x, s = self.preprocess(x)
        s = self.f16(s)
        s = s.unsqueeze(-1).unsqueeze(-1)
        x = x / s
        y = self.encoder(x)
        qx, loss1, loss2 = self.decoder.sq(y)
        y = self.decoder(qx)
        loss = self.criterion(y, x).mean()
        return loss, loss1, loss2

    def forward_pre(self, x, s):
        bs = self.f16.to_bit(s)
        s = self.f16.to_float(bs)
        s = s.unsqueeze(-1).unsqueeze(-1)
        x = x / s
        y = self.encoder(x)
        bx = self.decoder.sq.quant(y)
        qx = self.decoder.sq.dequant(bx)
        y = self.decoder(qx)
        loss = self.criterion(y, x).mean()
        return torch.cat([bs, bx], -1), loss

    def forward(self, x):
        x, s = self.preprocess(x)
        s = self.f16.to_bit(s)
        s = self.f16.to_float(s)
        s = s.unsqueeze(-1).unsqueeze(-1)
        x = x / s
        y = self.encoder(x)
        y = self.decoder.sq.quant(y)
        y = self.decoder.sq.dequant(y)
        y = self.decoder(y)
        y = (y * s + 0.5).clamp(0., 1.)
        y = y.view(x.size(0), 126, 128, 2).permute(0, 3, 1, 2).contiguous()
        return y


class Encoder(nn.Module):

    def __init__(self, feedback_bits, vq_b=8, vq_dim=8, vq_len=63, s_bit=8):
        super(Encoder, self).__init__()
        self.s_bit = s_bit
        self.model = VQVAE(vq_b, vq_dim, vq_len, s_bit)

    def forward(self, x):
        x, s = self.model.preprocess(x)
        best_bx, best_loss = self.model.forward_pre(x, s)
        for r in [0.9, 0.95, 1.05, 1.1]:
            si = s * r
            bx, loss = self.model.forward_pre(x, si)
            mask = (best_loss <= loss).long().unsqueeze(-1)
            best_bx = best_bx * mask + bx * (1 - mask)
            mask = mask.squeeze(-1).float()
            best_loss = best_loss * mask + loss * (1 - mask)
        return best_bx


class Decoder(nn.Module):

    def __init__(self, feedback_bits, vq_b=8, vq_dim=8, vq_len=63, s_bit=8):
        super(Decoder, self).__init__()
        self.s_bit = s_bit
        self.decoder = DecoderModel(vq_b, vq_dim, vq_len)
        self.f16 = FloatBiter(s_bit)

    def dequantize(self, x):
        s = self.f16.to_float(x[:, :self.s_bit])
        out = self.decoder.sq.dequant(x[:, self.s_bit:])
        return out, s

    def forward(self, x):
        b = x.size(0)
        out, s = self.dequantize(x)
        s = s.unsqueeze(-1).unsqueeze(-1)
        out = self.decoder(out)
        out = (out * s + 0.5).clamp(0., 1.)
        out = out.view(b, 126, 128, 2).permute(0, 3, 1, 2).contiguous()
        return out


class DatasetFolderTrain(Dataset):

    def __init__(self, matData, training=True):
        self.matdata = matData
        self.training = training

    def __getitem__(self, index):
        if self.training:
            data = copy.deepcopy(self.matdata[index])
            if random.random() < 0.25:
                data = 1 - data
            if random.random() < 0.25:
                data = data.reshape(2, 126, 2, 64)
                data = np.concatenate([data[:, :, 1], data[:, :, 0]], -1)
            if random.random() < 0.25:
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
