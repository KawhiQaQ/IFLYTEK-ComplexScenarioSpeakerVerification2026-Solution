"""Complete GRL speaker head with residual depth/time feature interaction."""
import copy
import torch
from torch import nn
from miwu.tidyvoice_w2vbert2_model import load_tidyvoice_w2vbert2_encoder

class DepthTimeResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(128)
        self.local=nn.Conv2d(128,128,3,padding=1,groups=128)
        self.expand=nn.Linear(128,256)
        self.project=nn.Linear(256,128)
        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)
    def forward(self,x):
        # [B,T,25,128]; convolution sees adjacent time steps and encoder depths.
        y=self.local(self.norm(x).permute(0,3,2,1)).permute(0,3,2,1)
        return x+self.project(torch.nn.functional.gelu(self.expand(y)))

class CompleteGRLHead(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.adapter_layers=copy.deepcopy(base.adapter_layers).float()
        self.pooling=copy.deepcopy(base.pooling).float()
        self.bottleneck=copy.deepcopy(base.bottleneck).float()
        self.interaction=DepthTimeResidual()
    def forward(self,states):
        x=torch.stack([m(h) for m,h in zip(self.adapter_layers,states)],dim=2)
        x=self.interaction(x)
        return self.bottleneck(self.pooling(x.flatten(2))).float()

class DepthTimeGRLW2VBert2(nn.Module):
    output_dimension=256
    def __init__(self,checkpoint,config_directory,head_checkpoint,trained_checkpoint=None):
        super().__init__()
        self.base=load_tidyvoice_w2vbert2_encoder(checkpoint,config_directory,device='cpu',adapted_checkpoint=head_checkpoint)
        # Same floating-point head computation in the frozen teacher and student.
        self.base.adapter_layers.float();self.base.pooling.float();self.base.bottleneck.float()
        self.head=CompleteGRLHead(self.base)
        self.requires_grad_(False)
        if trained_checkpoint:
            self.head.load_state_dict(torch.load(trained_checkpoint,map_location='cpu')['state_dict'],strict=True)
    def set_trainable(self):
        self.requires_grad_(False);self.eval();self.head.requires_grad_(True)
        # Pretrained BN running statistics stay fixed with small speaker batches.
        return self
    def _forward_equal(self,waveforms,return_parts=False):
        with torch.no_grad():
            h=self.base.encoder.feature_projection(self.base._extract_features(waveforms))[0]
            states=[h]
            for layer in self.base.encoder.encoder.layers:
                h=layer(h)[0];states.append(h)
            if return_parts:
                x=torch.cat([m(h) for m,h in zip(self.base.adapter_layers,states)],dim=-1)
                teacher=self.base.bottleneck(self.base.pooling(x)).float()
        result=self.head(states)
        return (result,teacher) if return_parts else result
    def forward(self,features=None,lengths=None,waveforms=None,waveform_lengths=None,return_parts=False):
        if waveforms is None:waveforms=features
        if waveform_lengths is not None and not (bool(torch.all(waveform_lengths==waveform_lengths[0])) and int(waveform_lengths[0])==waveforms.shape[1]):
            results=[self._forward_equal(waveforms[i:i+1,:int(n)],return_parts) for i,n in enumerate(waveform_lengths)]
            if return_parts:return tuple(torch.cat([r[j] for r in results],dim=0) for j in range(2))
            return torch.cat(results,dim=0)
        return self._forward_equal(waveforms,return_parts)
