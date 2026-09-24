import torch
import torch.nn as nn
from redimnet2.layers.layernorm import LayerNorm
from redimnet2.layers.resblocks import ResBasicBlock
from redimnet2.layers.convnext import ConvNeXtLikeBlock
from redimnet2.layers.attention import (
    TransformerEncoderLayer,
    StableTransformerEncoderLayer,
    TransformerEncoderLayerV2,
)

#------------------------------------------
#              Main blocks
#------------------------------------------

class ConvBlock2d(nn.Module):
    _CONVNEXT_VARIANTS = {
        "convnext_like":             dict(norm='bn', norm_placement='mid', ws=False, activation='gelu'),
        "convnext_like_ln":          dict(norm='ln', norm_placement='mid', ws=False, activation='gelu'),
        "convnext_like_relu":        dict(norm='bn', norm_placement='mid', ws=False, activation='relu'),
        "convnext_like_ws":          dict(norm='bn', norm_placement='mid', ws=True,  activation='gelu'),
        "convnext_like_ws_ln":       dict(norm='ln', norm_placement='mid', ws=True,  activation='gelu'),
        "convnext_like_post":        dict(norm='bn', norm_placement='post', ws=False, activation='gelu'),
        "convnext_like_post_ln":     dict(norm='ln', norm_placement='post', ws=False, activation='gelu'),
        "convnext_like_post_relu":   dict(norm='bn', norm_placement='post', ws=False, activation='relu'),
        "convnext_like_post_ws":     dict(norm='bn', norm_placement='post', ws=True,  activation='gelu'),
        "convnext_like_post_ws_ln":  dict(norm='ln', norm_placement='post', ws=True,  activation='gelu'),
        "convnext_like_pre":         dict(norm='bn', norm_placement='pre',  ws=False, activation='gelu'),
        "convnext_like_pre_ln":      dict(norm='ln', norm_placement='pre',  ws=False, activation='gelu'),
        "convnext_like_pre_ws":      dict(norm='bn', norm_placement='pre',  ws=True,  activation='gelu'),
        "convnext_like_pre_ws_ln":   dict(norm='ln', norm_placement='pre',  ws=True,  activation='gelu'),
    }

    def __init__(self, c, f, block_type="convnext_like", Gdiv=1, kernel_sizes=None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [(3, 3)]
        if block_type in self._CONVNEXT_VARIANTS:
            self.conv_block = ConvNeXtLikeBlock(c, dim=2, kernel_sizes=kernel_sizes,
                                                Gdiv=Gdiv, padding='same',
                                                **self._CONVNEXT_VARIANTS[block_type])
        elif block_type == "basic_resnet":
            self.conv_block = ResBasicBlock(c, c, f, stride=1, se_channels=min(64,max(c,32)), Gdiv=Gdiv, use_fwSE=False)
        elif block_type == "basic_resnet_fwse":
            self.conv_block = ResBasicBlock(c, c, f, stride=1, se_channels=min(64,max(c,32)), Gdiv=Gdiv, use_fwSE=True)
        else:
            raise NotImplementedError(f"Unknown block_type: {block_type!r}")

    def forward(self, x):
        return self.conv_block(x)

#------------------------------------------
#                1D block
#------------------------------------------

class PosEncConv(nn.Module):
    def __init__(self, C, ks, groups=None):
        super().__init__()
        assert ks % 2 == 1
        self.conv = nn.Conv1d(C,C,ks,
                              padding=ks//2,
                              groups=C if groups is None else groups)
        self.norm = LayerNorm(C, eps=1e-6, data_format="channels_first")

    def forward(self,x):        
        return x + self.norm(self.conv(x))


def build_internal_1d_tcm_block(hC, block_type, pos_ker_sz, **kwargs):
    if block_type == 'fc':
        tcm = nn.Sequential(
            nn.Conv1d(hC,hC*2,1),
            LayerNorm(hC*2, eps=1e-6,
                      data_format="channels_first"),
            nn.GELU(),
            nn.Conv1d(hC*2,hC,1)
        )
    elif block_type == 'conv':
        tcm = nn.Sequential(*[ConvNeXtLikeBlock(
            hC, dim=1, kernel_sizes=[7, 15, 31], Gdiv=1, padding='same'
        ) for i in range(4)])
    elif block_type == 'convx2':
        tcm = nn.Sequential(*[ConvNeXtLikeBlock(
            hC, dim=1, kernel_sizes=[7, 15], Gdiv=1, padding='same'
        ) for i in range(3)])
    elif block_type == 'att':
        tcm = nn.Sequential(
            PosEncConv(hC, ks=pos_ker_sz, groups=hC),
            TransformerEncoderLayer(
                n_state=hC,
                n_mlp=hC*2,
                n_head=4,
                **kwargs
            )
        )
    elif block_type == 'att-rope':
        tcm = TransformerEncoderLayerV2(
            n_state=hC,
            n_mlp=hC,
            n_head=4,
            **kwargs,
        )
    elif block_type == 'conv+att':
        tcm = nn.Sequential(
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7], Gdiv=1, padding='same'),
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[19], Gdiv=1, padding='same'),
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[31], Gdiv=1, padding='same'),
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[59], Gdiv=1, padding='same'),
            TransformerEncoderLayer(
                n_state=hC,
                n_mlp=hC,
                n_head=4,
                **kwargs
            )
        )
    elif block_type == 'conv+att-rope':
        tcm = nn.Sequential(
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7,19,31], Gdiv=1,
                              norm='bn', norm_placement='post', padding='same'),
            TransformerEncoderLayerV2(
                n_state=hC,
                n_mlp=hC,
                n_head=4,
                **kwargs,
            ),
        )
    elif block_type == 'conv_large':
        tcm = nn.Sequential(
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7], Gdiv=1, padding='same'),
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[19], Gdiv=1, padding='same'),
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[31], Gdiv=1, padding='same'),
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[59], Gdiv=1, padding='same'),
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[15], Gdiv=1, padding='same', dilation=7),
        )
    elif block_type == 'stable_att':
        tcm = nn.Sequential(
            PosEncConv(hC, ks=pos_ker_sz, groups=hC),
            StableTransformerEncoderLayer(
                n_state=hC,
                n_mlp=hC*2,
                n_head=4,
                **kwargs
            )
        )
    elif block_type == 'conv+stable_att':
        tcm = nn.Sequential(
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7], Gdiv=1, padding='same'),
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[19], Gdiv=1, padding='same'),
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[31], Gdiv=1, padding='same'),
            ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[59], Gdiv=1, padding='same'),
            StableTransformerEncoderLayer(
                n_state=hC,
                n_mlp=hC,
                n_head=4,
                **kwargs
            )
        )
    else:
        raise NotImplementedError(f"Unknown 1D block_type: {block_type!r}")
    return tcm


class TimeContextBlock1d(nn.Module):
    def __init__(self, 
        C, 
        hC,
        pos_ker_sz = 59,
        block_type = 'att',
        red_dim_conv = None,
        exp_dim_conv = None,
        **kwargs
    ):
        super().__init__()
        assert pos_ker_sz 
        
        self.red_dim_conv = nn.Sequential(
            nn.Conv1d(C,hC,1),
            LayerNorm(hC, eps=1e-6, data_format="channels_first")
        )
        self.tcm = build_internal_1d_tcm_block(hC, block_type, pos_ker_sz, **kwargs)
        self.exp_dim_conv = nn.Conv1d(hC,C,1)

    def forward(self,x):
        skip = x
        x = self.red_dim_conv(x)
        x = self.tcm(x)
        x = self.exp_dim_conv(x)
        return skip + x


class SymmetricalDWMixer(nn.Module):
    def __init__(self, C: int, F: int, hC: int, block_type='att', **kwargs):
        super().__init__()
        self.C = C
        self.F = F
        self.hC = hC

        self.channel_reduce = nn.Conv2d(in_channels=C, out_channels=hC, kernel_size=1, bias=False)
        self.freq_reduce_weight = nn.Parameter(torch.empty(hC, F))
        self.freq_reduce_bias = nn.Parameter(torch.zeros(hC, 1))
        self.reduce_norm = LayerNorm(hC, eps=1e-6, data_format="channels_first")

        self.time_context_block = build_internal_1d_tcm_block(
            hC=hC, block_type=block_type, pos_ker_sz=59, **kwargs)
        self.post_tcm_norm = LayerNorm(hC, eps=1e-6, data_format="channels_first")

        self.freq_expand_weight = nn.Parameter(torch.empty(hC, F))
        self.freq_expand_bias = nn.Parameter(torch.zeros(hC, F))
        self.channel_expand = nn.Conv2d(in_channels=hC, out_channels=C, kernel_size=1, bias=False)
        self.expand_norm = LayerNorm(C, eps=1e-6, data_format="channels_first")

        self._init_weights()

    def _init_weights(self):
        import math
        for matrix in [self.freq_reduce_weight, self.freq_expand_weight]:
            hC, F = matrix.shape
            with torch.no_grad():
                for c in range(hC):
                    frequency_scale = 1.0 / (10000 ** (2 * (c // 2) / hC))
                    for f in range(F):
                        if c % 2 == 0:
                            matrix[c, f] = math.sin(f * frequency_scale)
                        else:
                            matrix[c, f] = math.cos(f * frequency_scale)
                matrix.mul_(0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.channel_reduce(x)
        x = torch.einsum('b c f t, c f -> b c t', x, self.freq_reduce_weight)
        x = x + self.freq_reduce_bias
        x = self.reduce_norm(x)
        x = self.time_context_block(x)
        x = self.post_tcm_norm(x)
        x = torch.einsum('b c t, c f -> b c f t', x, self.freq_expand_weight)
        x = x + self.freq_expand_bias.unsqueeze(-1)
        x = self.channel_expand(x)
        x = self.expand_norm(x)
        return x + residual


_V2_BASE_KERNELS = (7, 19, 31, 59)


def _to_odd(k: int) -> int:
    k = max(3, int(round(k)))
    if k % 2 == 0:
        k += 1
    return k


def _scale_kernels_by_stride(time_stride, base_kernels=_V2_BASE_KERNELS):
    return [_to_odd(k / max(1, time_stride)) for k in base_kernels]


class TimeContextBlock1dV2(nn.Module):
    def __init__(self,
        C,
        hC,
        block_type='conv+att',
        time_stride: int = 1,
        conv_kernel_sizes=None,
        red_dim_conv=None,
        exp_dim_conv=None,
        **kwargs
    ):
        super().__init__()
        self.block_type = block_type
        self.time_stride = time_stride

        self.red_dim_conv = nn.Sequential(
            nn.Conv1d(C, hC, 1),
            LayerNorm(hC, eps=1e-6, data_format="channels_first"),
        )

        if conv_kernel_sizes is None:
            conv_kernel_sizes = _scale_kernels_by_stride(time_stride)
        self.conv_kernel_sizes = list(conv_kernel_sizes)

        if block_type == 'fc':
            self.tcm = nn.Sequential(
                nn.Conv1d(hC, hC * 2, 1),
                LayerNorm(hC * 2, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
                nn.Conv1d(hC * 2, hC, 1),
            )
        elif block_type == 'conv':
            ks = self.conv_kernel_sizes
            self.tcm = nn.Sequential(*[
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=ks, Gdiv=1, padding='same')
                for _ in range(4)
            ])
        elif block_type == 'att':
            self.tcm = nn.Sequential(
                TransformerEncoderLayerV2(
                    n_state=hC,
                    n_mlp=hC * 2,
                    n_head=4,
                    **kwargs,
                )
            )
        elif block_type == 'conv+att':
            conv_layers = [
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[k], Gdiv=1, padding='same')
                for k in self.conv_kernel_sizes
            ]
            self.tcm = nn.Sequential(
                *conv_layers,
                TransformerEncoderLayerV2(
                    n_state=hC,
                    n_mlp=hC,
                    n_head=4,
                    **kwargs,
                ),
            )
        else:
            raise NotImplementedError(
                f"TimeContextBlock1dV2 does not support block_type={block_type!r}"
            )

        self.exp_dim_conv = nn.Conv1d(hC, C, 1)

    def extra_repr(self) -> str:
        return (
            f"block_type={self.block_type}, time_stride={self.time_stride}, "
            f"conv_kernels={self.conv_kernel_sizes}"
        )

    def forward(self, x):
        skip = x
        x = self.red_dim_conv(x)
        x = self.tcm(x)
        x = self.exp_dim_conv(x)
        return skip + x
