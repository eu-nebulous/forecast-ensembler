"""SCript for deep learning helpers fucnctions"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def base_linear_module(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f), nn.LeakyReLU(), nn.BatchNorm1d(num_features=out_f)
    )


def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps / masked_sums


def calculate_matching_padding(i, o, k, d, s):
    p = ((o - 1) * s + (k - 1) * (d - 1) - i + k) / 2
    return int(p)


def base_conv_layer(n_inputs, n_outputs, kernel_size, stride=1, padding=1, dilation=1):
    """Base conv 1D layer with group norm"""
    return nn.Sequential(
        nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        ),
        nn.GroupNorm(1, n_outputs),
        nn.ReLU(inplace=True),
    )


def attention_module(embed_dim=4):
    """Self attention module"""
    return nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=embed_dim, dropout=0.5, batch_first=True
    )


def calculate_output_shape(i, p, k, d, s):
    """Output shape (series len) for the given padding, kernel, dilation, stride"""
    o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
    return int(o)
