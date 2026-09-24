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

#-------------------------------------------------------------
# Rotary positional embeddings
#-------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        assert head_dim % 2 == 0, f"RoPE requires even head_dim, got {head_dim}"
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cache", emb.cos(), persistent=False)
        self.register_buffer("_sin_cache", emb.sin(), persistent=False)
        self._cached_len = seq_len

    def forward(self, seq_len: int, device, dtype):
        if seq_len > self._cached_len:
            self._build_cache(seq_len)
        cos = self._cos_cache[:seq_len].to(device=device, dtype=dtype)
        sin = self._sin_cache[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


#-------------------------------------------------------------
# Copy multi-head attention module from hugginface wav2vec2
#-------------------------------------------------------------

# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/wav2vec2/modeling_wav2vec2.py
# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Wav2Vec2
class MultiHeadAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        qk_rope: bool = False,
        rope_base: float = 10000.0,
        rope_max_seq_len: int = 4096,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.qk_norm = qk_norm
        self.qk_rope = qk_rope

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if self.qk_rope:
            self.rope = RotaryEmbedding(
                head_dim=self.head_dim,
                base=rope_base,
                max_seq_len=rope_max_seq_len,
            )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,hidden_states: torch.Tensor) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        if self.qk_rope:
            cos, sin = self.rope(tgt_len, device=query_states.device, dtype=query_states.dtype)
            query_states = _apply_rope(query_states, cos, sin)
            key_states = _apply_rope(key_states, cos, sin)

        if self.qk_norm:
            query_states = F.normalize(query_states, dim=-1)
            key_states = F.normalize(key_states, dim=-1)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output
    
#-------------------------------------------------------------
# Import activations from hugginface
#-------------------------------------------------------------
# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/activations.py

def gelu(x):
    """This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class NewGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class GELUActivation(nn.Module):
    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = F.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)

class ClippedGELUActivation(nn.Module):
    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)

# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/activations.py
class FastGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))

# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/activations.py
class QuickGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)

# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/activations.py
class SiLUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input)
    
class MishActivation(nn.Module):
    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9.0"):
            self.act = self._mish_python
        else:
            self.act = F.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        return input * torch.tanh(F.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)

class LinearActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)
    
ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLUActivation,
    "swish": SiLUActivation,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)

#-------------------------------------------------------------
#  Copy FeedForward & Encoder from hugginface wav2vec2
#-------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, 
            hidden_size:int,
            intermediate_size:int,
            hidden_act:str='gelu_new',
            activation_dropout:float=0.0,
            hidden_dropout:float=0.0
        ):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(activation_dropout)
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN[hidden_act]
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
            n_state: int, 
            n_mlp : int, 
            n_head: int,
            channel_last: bool = False,
            act: str = 'gelu_new',
            act_do:float = 0.0,
            att_do:float = 0.0,
            hid_do:float = 0.0,
            ln_eps:float=1e-6
        ):
        
        hidden_size=n_state
        num_attention_heads=n_head
        intermediate_size=n_mlp
        hidden_act=act
        activation_dropout=act_do
        attention_dropout=att_do
        hidden_dropout=hid_do
        layer_norm_eps=ln_eps
        
        super().__init__()
        self.channel_last = channel_last
        self.attention = MultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            intermediate_size=intermediate_size,
            activation_dropout=activation_dropout,
            hidden_dropout=hidden_dropout
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        if not self.channel_last:
            hidden_states = hidden_states.permute(0, 2, 1)
        attn_residual = hidden_states
        hidden_states = self.attention(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = hidden_states
        if not self.channel_last:
            outputs = outputs.permute(0, 2, 1)
        return outputs


class StableTransformerEncoderLayer(nn.Module):
    def __init__(self,
            n_state: int,
            n_mlp : int,
            n_head: int,
            channel_last: bool = False,
            act: str = 'gelu_new',
            act_do:float = 0.0,
            att_do:float = 0.02,
            hid_do:float = 0.0,
            ln_eps:float=1e-6
        ):
        super().__init__()
        self.channel_last = channel_last
        self.attention = MultiHeadAttention(
            embed_dim=n_state,
            num_heads=n_head,
            dropout=att_do,
            qk_norm=True,
        )
        self.layer_norm = nn.LayerNorm(n_state, eps=ln_eps)
        self.feed_forward = FeedForward(
            hidden_size=n_state,
            hidden_act=act,
            intermediate_size=n_mlp,
            activation_dropout=act_do,
            hidden_dropout=hid_do,
        )
        self.final_layer_norm = nn.LayerNorm(n_state, eps=ln_eps)

    def forward(self, hidden_states):
        if not self.channel_last:
            hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = hidden_states + self.attention(self.layer_norm(hidden_states))
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = hidden_states
        if not self.channel_last:
            outputs = outputs.permute(0, 2, 1)
        return outputs


class TransformerEncoderLayerV2(nn.Module):
    def __init__(self,
            n_state: int,
            n_mlp: int,
            n_head: int,
            channel_last: bool = False,
            act: str = 'gelu_new',
            act_do: float = 0.0,
            att_do: float = 0.0,
            hid_do: float = 0.0,
            ln_eps: float = 1e-6,
            qk_norm: bool = False,
            rope_max_seq_len: int = 4096,
            rope_base: float = 10000.0,
        ):
        super().__init__()
        self.channel_last = channel_last
        self.attention = MultiHeadAttention(
            embed_dim=n_state,
            num_heads=n_head,
            dropout=att_do,
            qk_norm=qk_norm,
            qk_rope=True,
            rope_base=rope_base,
            rope_max_seq_len=rope_max_seq_len,
        )
        self.layer_norm = nn.LayerNorm(n_state, eps=ln_eps)
        self.feed_forward = FeedForward(
            hidden_size=n_state,
            hidden_act=act,
            intermediate_size=n_mlp,
            activation_dropout=act_do,
            hidden_dropout=hid_do,
        )
        self.final_layer_norm = nn.LayerNorm(n_state, eps=ln_eps)

    def forward(self, hidden_states):
        if not self.channel_last:
            hidden_states = hidden_states.permute(0, 2, 1)

        attn_residual = hidden_states
        hidden_states = self.attention(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)

        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = hidden_states
        if not self.channel_last:
            outputs = outputs.permute(0, 2, 1)
        return outputs
