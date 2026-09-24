import math
import torch
import functools
import numpy as np
import torch.nn as nn

import redimnet2.layers.features_tf as features_tf
import redimnet2.layers.poolings as pooling_layers
from redimnet2.layers.layernorm import LayerNorm
from redimnet2.layers.blocks import (
    ConvBlock2d,
    TimeContextBlock1d,
    SymmetricalDWMixer,
    TimeContextBlock1dV2,
)
from redimnet2.layers.redim_structural import to1d, to2d, to1d_tfopt, to2d_tfopt, weigth1d

# class ShapeLogger(nn.Module):
#     def __init__(self, module):
#         super().__init__()
#         self.module = module

#     def forward(self,x):
#         cls_name = self.module.__class__.__name__
#         in_shape = tuple(x.size())
#         h = self.module(x)
#         out_shape = tuple(h.size())
#         print(f"{cls_name} : {in_shape} -> {out_shape}")
#         return h

ShapeLogger = lambda x : x

#------------------------------------------

class FreqEncoder(nn.Module):
    def __init__(self,c,bins):
        super().__init__()
        self.freq_embedder = nn.Embedding(
            num_embeddings=bins, 
            embedding_dim=c)

    def forward(self,x):
        b, c, f, t = x.size()
        freqs = torch.range(start=0,end=f-1, step=1, dtype=torch.long)
        freqs = freqs.unsqueeze(0).repeat(b,1).to(x.device) # [bs,f]
        fe = self.freq_embedder(freqs).permute(0,2,1).unsqueeze(-1) # [bs, freq_emb_dim, f, 1]
        fe = fe.repeat(1,1,1,t)
        x = x + fe
        return x

import collections
from itertools import repeat
from typing import Any
# https://github.com/pytorch/pytorch/blob/dc8692b0eb093d5af150ae0f3a29a0957c3e4c0d/torch/nn/modules/utils.py#L10
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse
    
_pair = _ntuple(2, "_pair")


def _parse_agg_gnorm(spec, C, F):
    if spec is False or spec is None:
        return 0
    if spec is True:
        groups = int(C)
    elif isinstance(spec, int):
        if spec <= 0:
            return 0
        groups = int(spec)
    else:
        raise TypeError(f"agg_gnorm must be bool, int, or None; got {spec!r}")

    num_channels = int(C * F)
    if num_channels % groups != 0:
        raise ValueError(f"agg_gnorm groups={groups} must divide C*F={num_channels}")
    return groups


def _make_agg_norm(groups, num_channels):
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)

#------------------------------------------
#                 UReDimNet
#------------------------------------------
class ReDimNet2(nn.Module):
    # UNet-like ReDimNet
    def __init__(self,
        F = 72,
        C = 24,
        spec_in_channels = 1, # Phase + Magnitude
        causal = 'none',
        out_channels = None,
        block_1d_type = 'conv+att',
        block_2d_type = "basic_resnet",
        return_2d_output = False,
        fm_weigthing_type = 'NC',
        use_freq_pos_enc = False,
        compress_tconvs = True,
        stages_setup = [
            # Encoder part:
            ((1,1),2,4,[(3,3)],None), # 16
            ((2,1),3,3,[(3,3)],None), # 32

            ((1,2),4,2,[(3,3)],None), # 64,
            ((2,1),5,1,[(3,3)],48), # 128

            ((1,2),4,1,[(3,3)],64), # 128
            ((2,1),3,1,[(3,3)],96), # 128
        ],
        group_divisor = 1,
        dual_agg = False,
        agg_gnorm = False,
        att_dos = None,
        #-----------------------
        #     Subnet stuff
        #-----------------------
        return_all_outputs = False,
        offset_fm_weights = 0,
        is_subnet = False,
    ):
        super().__init__()
        self.F = F
        self.C = C

        if causal == 'full':
            block_1d_type = block_1d_type + '-causal'
            block_2d_type = block_2d_type + '-causal'
            self.causal = True
        elif causal == 'only_1d':
            block_1d_type = block_1d_type + '-causal'
            self.causal = True
        elif causal == 'none':
            self.causal = False
        else:
            raise NotImplementedError()

        self.tcm_v2 = block_1d_type.startswith('v2_')
        if self.tcm_v2:
            block_1d_type = block_1d_type[len('v2_'):]

        self.use_sdwm = block_1d_type.startswith('sdwm_')
        if self.use_sdwm:
            block_1d_type = block_1d_type[len('sdwm_'):]

        self.block_1d_type = block_1d_type
        self.block_2d_type = block_2d_type

        self.stages_setup = stages_setup
        self.fm_weigthing_type = fm_weigthing_type
        self.dual_agg = dual_agg
        self.agg_gnorm = _parse_agg_gnorm(agg_gnorm, C, F)
        self.att_dos = {} if att_dos is None else att_dos

        # Subnet stuff
        self.is_subnet = is_subnet
        self.offset_fm_weights = offset_fm_weights
        self.return_all_outputs = return_all_outputs

        self.build(F,C,spec_in_channels,out_channels,stages_setup,group_divisor,
                   compress_tconvs,return_2d_output,use_freq_pos_enc)
        
    def build(self,F,C,spec_in_channels,out_channels,stages_setup,group_divisor,
              compress_tconvs,return_2d_output,use_freq_pos_enc):
        self.F = F
        self.C = C
        
        c = C
        f = F
        
        stt = 1
        sft = 1

        max_stt = stt
        
        self.num_stages = len(stages_setup)

        append_to1d_before_tcm = not self.use_sdwm
        if self.use_sdwm:
            _sdwm_partial = functools.partial(
                SymmetricalDWMixer,
                block_type=self.block_1d_type,
                **self.att_dos,
            )

            def Block1d(*args, C=None, F=None, hC=None, _stt_val=1, **kw):
                return _sdwm_partial(C=C, F=F, hC=hC)
        elif self.tcm_v2:
            _v2_partial = functools.partial(
                TimeContextBlock1dV2,
                block_type=self.block_1d_type,
                **self.att_dos,
            )

            def Block1d(*args, _stt_val=1, **kw):
                return _v2_partial(*args, time_stride=_stt_val, **kw)
        else:
            _v1_partial = functools.partial(
                TimeContextBlock1d,
                block_type=self.block_1d_type,
                **self.att_dos,
            )

            def Block1d(*args, _stt_val=1, **kw):
                return _v1_partial(*args, **kw)
        Block2d = functools.partial(ConvBlock2d,block_type=self.block_2d_type)

        if self.fm_weigthing_type == 'NC':
            agg1d = functools.partial(weigth1d,C=F*C)
        elif self.fm_weigthing_type == 'N':
            agg1d = functools.partial(weigth1d,C=None)
        else:
            raise NotImplementedError()

        if not self.is_subnet:
            self.stem = nn.Sequential(
                nn.Conv2d(spec_in_channels, int(c), kernel_size=3, stride=1, padding='same'),
                LayerNorm(int(c), eps=1e-6, data_format="channels_first"),
                to1d()
            )
        else:
            # Subnet stem: aggregate offset_fm_weights incoming 1D feature maps,
            # reshape to 2D, then apply a standard conv+norm stem before to1d().
            assert self.offset_fm_weights > 0, \
                "offset_fm_weights must be > 0 when is_subnet=True"
            self.stem = nn.Sequential(
                agg1d(N=self.offset_fm_weights,
                      requires_grad=self.offset_fm_weights>1),
                to2d(f=F, c=C),
                nn.Conv2d(int(c), int(c), kernel_size=3, stride=1, padding='same'),
                LayerNorm(int(c), eps=1e-6, data_format="channels_first"),
                to1d()
            )

        if self.agg_gnorm:
            self.stem_gnorm = _make_agg_norm(self.agg_gnorm, C*F)

        # Track accumulated feature-map count for the weigth1d N parameter.
        # Starts at offset_fm_weights+1 to account for the subnet offset + stem output.
        feat_count = self.offset_fm_weights + 1
        self._stage_has_dual = []

        for stage_ind, (stride, num_blocks, conv_exp, kernel_sizes, att_block_red) in enumerate(stages_setup):
            (sf, st) = stride
            tot_stride = np.prod((sf, st))
            num_feats_to_weight = feat_count
            # if tot_stride > 1:
            layers = []
            sft = sft * sf
            stt = stt * st
            layers.append(agg1d(N=num_feats_to_weight, requires_grad=num_feats_to_weight>1))
            layers.append(to2d(f=f, c=c))
            if use_freq_pos_enc:
                layers.append(FreqEncoder(c=c,bins=f))

            layers.append(ShapeLogger(nn.Conv2d(int(c), int(sf*c*conv_exp),
                            kernel_size=(sf,stt),
                            stride=(sf,stt),
                            padding=0, groups=1 if not compress_tconvs else
                                        math.gcd(int(c),int(sf*c*conv_exp)))))

            c = sf * c
            assert f % sf == 0
            f = f // sf

            if stt >= max_stt:
                max_stt = stt

            for block_ind in range(num_blocks):
                layers.append(Block2d(c=int(c*conv_exp), f=f,
                                      kernel_sizes=kernel_sizes, Gdiv=group_divisor))

            if conv_exp != 1:
                _group_divisor = group_divisor
                layers.append(nn.Sequential(
                    nn.Conv2d(int(c*conv_exp), c, kernel_size=1, stride=1, padding='same'),
                    nn.BatchNorm2d(c, eps=1e-6)
                ))

            has_dual = self.dual_agg and att_block_red is not None

            if has_dual:
                # Split the stage so the 1D-attention branch runs in parallel with
                # a plain 2D->1D reshape branch; both are upsampled (+gnorm) and
                # aggregated alongside prior feature maps.
                if append_to1d_before_tcm:
                    layers.append(to1d())
                setattr(self, f'stage{stage_ind}_pre', nn.Sequential(*layers))

                if append_to1d_before_tcm:
                    blk_1d = Block1d(C*F, hC=(C*F)//att_block_red, _stt_val=stt)
                else:
                    blk_1d = Block1d(C=c, F=f, hC=att_block_red, _stt_val=stt)
                setattr(self, f'stage{stage_ind}_1d', blk_1d)

                up_2d = []
                if not append_to1d_before_tcm:
                    up_2d.append(to1d())
                up_2d.append(ShapeLogger(nn.Upsample(scale_factor=stt, mode='nearest')))
                if self.agg_gnorm:
                    up_2d.append(_make_agg_norm(self.agg_gnorm, C*F))
                setattr(self, f'stage{stage_ind}_up_2d', nn.Sequential(*up_2d))

                up_1d = []
                if not append_to1d_before_tcm:
                    up_1d.append(to1d())
                up_1d.append(ShapeLogger(nn.Upsample(scale_factor=stt, mode='nearest')))
                if self.agg_gnorm:
                    up_1d.append(_make_agg_norm(self.agg_gnorm, C*F))
                setattr(self, f'stage{stage_ind}_up_1d', nn.Sequential(*up_1d))

                self._stage_has_dual.append(True)
                feat_count += 2
            else:
                if append_to1d_before_tcm:
                    layers.append(to1d())
                if att_block_red is not None:
                    if append_to1d_before_tcm:
                        layers.append(Block1d(C*F,hC=(C*F)//att_block_red, _stt_val=stt))
                    else:
                        layers.append(Block1d(C=c,F=f,hC=att_block_red, _stt_val=stt))
                if not append_to1d_before_tcm:
                    layers.append(to1d())
                layers.append(ShapeLogger(nn.Upsample(scale_factor=stt, mode='nearest')))
                if self.agg_gnorm:
                    layers.append(_make_agg_norm(self.agg_gnorm, C*F))
                setattr(self,f'stage{stage_ind}',nn.Sequential(*layers))

                self._stage_has_dual.append(False)
                feat_count += 1

        self.fin_wght1d = agg1d(N=feat_count, requires_grad=feat_count>1)

        self.time_stride = max_stt
        self.freq_stride = sft
        self.head = nn.Identity()
        if return_2d_output:
            self.fin_to2d = to2d(f=f,c=c)
            if out_channels is not None:
                self.head = nn.Conv2d(c, out_channels, 1)
        else:
            self.fin_to2d = nn.Identity()
            if out_channels is not None:
                self.head = nn.Conv1d(C*F, out_channels, 1)
        
    def run_stage(self,prev_outs_1d, stage_ind):
        if self._stage_has_dual[stage_ind]:
            pre = getattr(self, f'stage{stage_ind}_pre')
            blk_1d = getattr(self, f'stage{stage_ind}_1d')
            up_2d = getattr(self, f'stage{stage_ind}_up_2d')
            up_1d = getattr(self, f'stage{stage_ind}_up_1d')
            x_pre = pre(prev_outs_1d)
            x_2d = up_2d(x_pre)
            x_1d = up_1d(blk_1d(x_pre))
            return [x_2d, x_1d]
        stage = getattr(self,f'stage{stage_ind}')
        return [stage(prev_outs_1d)]

    def forward(self,inp):
        if not self.is_subnet:
            bs, _, _, T = inp.size()
            inp = inp[:,:,:,:(T//self.time_stride)*self.time_stride] # Needed for right reshape operations
            # print(f"T = {T} -> T = {(T//self.time_stride)*self.time_stride}")
            x = self.stem(inp)
            if self.agg_gnorm:
                x = self.stem_gnorm(x)
            outputs_1d = [x]
        else:
            assert isinstance(inp, list), \
                "Subnet-mode ReDimNet2 expects a list of 1D feature maps as input"
            outputs_1d = list(inp)
            x = self.stem(inp)
            if self.agg_gnorm:
                x = self.stem_gnorm(x)
            outputs_1d.append(x)

        for stage_ind in range(self.num_stages):
            outputs_1d.extend(self.run_stage(outputs_1d,stage_ind))
        x = self.fin_wght1d(outputs_1d)
        outputs_1d.append(x)
        x = self.fin_to2d(x)
        x = self.head(x)

        if self.return_all_outputs:
            return x, outputs_1d
        return x

class ReDimNet2Wrap(nn.Module):
    def __init__(self,
        F = 72,
        C = 24,
        causal = False,
        spec_in_channels = 1, # Phase + Magnitude
        out_channels = None,
        block_1d_type = 'conv+att',
        block_2d_type = "basic_resnet", 
        compress_tconvs = True,
        return_2d_output = False,
        use_freq_pos_enc = False,
        fm_weigthing_type = 'NC',
        stages_setup = [
            # Encoder part:
            ((1,1),2,4,[(3,3)],24), # 16
            ((2,1),3,3,[(3,3)],24), # 32

            ((1,2),4,2,[(3,3)],24), # 64,
            ((2,1),5,1,[(3,3)],24), # 128

            ((1,2),4,1,[(3,3)],24), # 128
            ((2,1),3,1,[(3,3)],24), # 128
        ],
        group_divisor = 1,
        dual_agg = False,
        agg_gnorm = False,
        att_dos = None,
        #-------------------------
        embed_dim=192,
        num_classes=None,
        feat_agg_dropout=0.0,
        head_activation=None,
        hop_length=160,
        pooling_func='ASTP',
        pad_right_samples=None,
        before_pool_offset=None,
        feat_type='pt',
        global_context_att=True,
        emb_bn=False,
        #-------------------------
        spec_params = dict(
            do_spec_aug=False,
            do_preemph=True,
            norm_signal=True,
        ),
        #-------------------------
        return_all_outputs = False,
    ):
        super().__init__()

        self.return_all_outputs = return_all_outputs

        self.backbone = ReDimNet2(
            F = F, C = C,
            causal = causal,
            spec_in_channels = spec_in_channels, # Phase + Magnitude
            out_channels = out_channels,
            return_2d_output = return_2d_output,
            block_1d_type = block_1d_type,
            block_2d_type = block_2d_type,
            compress_tconvs = compress_tconvs,
            fm_weigthing_type = fm_weigthing_type,
            use_freq_pos_enc = use_freq_pos_enc,
            stages_setup = stages_setup,
            group_divisor = group_divisor,
            dual_agg = dual_agg,
            agg_gnorm = agg_gnorm,
            att_dos = att_dos,
            return_all_outputs = return_all_outputs,
        )
        if feat_type in ['pt','pt_mel']:
            # Keep torchaudio optional.  The competition checkpoint uses the
            # TensorFlow-compatible frontend below and the official runtime
            # image does not guarantee that torchaudio is installed.
            import redimnet2.layers.features as features
            self.spec = features.MelBanks(n_mels=F,hop_length=hop_length,**spec_params)
        elif feat_type in ['tf','tf_mel']:
            self.spec = features_tf.TFMelBanks(n_mels=F,hop_length=hop_length,**spec_params)
        elif feat_type == 'tf_spec':
            self.spec = features_tf.TFSpectrogram(**spec_params)
        elif feat_type == 'pt_stft':
            import redimnet2.layers.features as features
            self.spec = features.STFT(**spec_params)
        
        if out_channels is None:
            out_channels = C*F
        else:
            if return_2d_output:
                out_channels = (F//self.backbone.freq_stride) * out_channels
            else:
                out_channels = out_channels

        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=out_channels, global_context_att=global_context_att)

        self.pad_right_samples = pad_right_samples
        self.before_pool_offset = before_pool_offset
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.embed_dim = embed_dim
        self.emb_bn = emb_bn
        if emb_bn:  # better in SSL for SV
            self.bn2 = nn.BatchNorm1d(embed_dim)
        else:
            self.bn2 = None

    def forward(self,x):
        if self.pad_right_samples is not None:
            x = torch.nn.functional.pad(x, (0, self.pad_right_samples), mode='constant', value=None)
        x = self.spec(x)

        if x.ndim == 3:
            x = x.unsqueeze(1)
        # print(f"spec : {x.size()}")
        if self.return_all_outputs:
            out, all_outs_1d = self.backbone(x)
        else:
            out = self.backbone(x)
        # print(f"pre pool : {out.size()}")
        if out.ndim == 4:
            bs, C, F, T = out.size()
            out = out.reshape(bs, C*F, T)
        if self.before_pool_offset is not None:
            out = out[:,:,self.before_pool_offset:]
        out = self.bn(self.pool(out))
        out = self.linear(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.return_all_outputs:
            return out, all_outs_1d
        return out

def ReDimNet2Custom(**kwargs):
    return ReDimNet2Wrap(**kwargs)
