# MIT License
# 
# Copyright (c) 2024 ID R&D, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
import functools
import numpy as np
import torch.nn as nn
from typing import List
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Iterable, Optional
from redimnet2.layers.layernorm import LayerNorm

#------------------------------------------
#           ConvNeXtV2 block
#------------------------------------------

MaxPoolNd = {
    1 : nn.MaxPool1d,
    2 : nn.MaxPool2d
}

ConvNd = {
    1 : nn.Conv1d,
    2 : nn.Conv2d
}

BatchNormNd = {
    1 : nn.BatchNorm1d,
    2 : nn.BatchNorm2d
}


class _WSConvMixin:
    def _standardize(self):
        w = self.weight
        orig_dtype = w.dtype
        w32 = w.to(torch.float32)
        mean_dims = list(range(1, w32.dim()))
        w32 = w32 - w32.mean(dim=mean_dims, keepdim=True)
        std = w32.std(dim=mean_dims, keepdim=True) + 1e-5
        return (w32 / std).to(orig_dtype)


class WSConv2d(nn.Conv2d, _WSConvMixin):
    def forward(self, x):
        w = self._standardize()
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class WSConv1d(nn.Conv1d, _WSConvMixin):
    def forward(self, x):
        w = self._standardize()
        return F.conv1d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


WSConvNd = {
    1 : WSConv1d,
    2 : WSConv2d
}


# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
class ConvNeXtLikeBlock(nn.Module):
    def __init__(self, C, dim=2, kernel_sizes=[(3,3),], Gdiv=1, padding='same',
                 activation='gelu', dilation=1, norm='bn',
                 norm_placement='mid', ws=False):
        super().__init__()
        if norm_placement not in ('pre', 'mid', 'post'):
            raise ValueError(
                f"norm_placement must be 'pre', 'mid', or 'post', got {norm_placement!r}"
            )
        self.norm_placement = norm_placement
        self.ws = ws

        Conv = WSConvNd[dim] if ws else ConvNd[dim]
        self.dwconvs = nn.ModuleList(modules=[
            Conv(C, C, kernel_size=ks, dilation=dilation,
                 padding=padding, groups=C//Gdiv if Gdiv is not None else 1)
            for ks in kernel_sizes
        ])

        norm_C = C * len(kernel_sizes) if norm_placement == 'mid' else C
        if norm == 'bn':
            self.norm = BatchNormNd[dim](norm_C)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_C, data_format='channels_first')
        else:
            raise NotImplementedError(f"Unknown norm: {norm!r}")

        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            raise NotImplementedError(f"Unknown activation: {activation!r}")

        self.pwconv1 = Conv(C * len(kernel_sizes), C, 1) # pointwise/1x1 convs, implemented with linear layers

    def forward(self, x):
        skip = x
        if self.norm_placement == 'pre':
            x = self.norm(x)
        x = torch.cat([dwconv(x) for dwconv in self.dwconvs],dim=1)
        if self.norm_placement == 'mid':
            x = self.norm(x)
        x = self.act(x)
        x = self.pwconv1(x)
        if self.norm_placement == 'post':
            x = self.norm(x)
        x = skip + x
        return x
