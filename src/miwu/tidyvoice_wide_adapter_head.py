"""Full speaker head with zero-initialized wide residual SSL adapters."""
import torch
from torch import nn
from miwu.tidyvoice_w2vbert2_axial_head import ContinuedGRLW2VBert2
from miwu.tidyvoice_w2vbert2_depth_time_head import CompleteGRLHead


class WideResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(1024)
        self.expand = nn.Linear(1024, 512)
        self.project = nn.Linear(512, 128)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, hidden):
        return self.project(torch.nn.functional.gelu(self.expand(self.norm(hidden))))


class WideResidualGRLHead(CompleteGRLHead):
    def __init__(self, base):
        super().__init__(base)
        self.wide_adapters = nn.ModuleList([WideResidualAdapter() for _ in self.adapter_layers])

    def forward(self, states):
        with torch.cuda.amp.autocast(enabled=states[0].is_cuda, dtype=torch.float16):
            adapted = [old(h) + residual(h) for old, residual, h in
                       zip(self.adapter_layers, self.wide_adapters, states)]
            x = self.interaction(torch.stack(adapted, dim=2))
            return self.bottleneck(self.pooling(x.flatten(2))).float()


class WideAdapterGRLW2VBert2(ContinuedGRLW2VBert2):
    def __init__(self, checkpoint, config_directory, head_checkpoint, trained_checkpoint=None):
        super().__init__(checkpoint, config_directory, head_checkpoint)
        head = WideResidualGRLHead(self.base)
        missing, unexpected = head.load_state_dict(self.head.state_dict(), strict=False)
        assert not unexpected and missing and all(k.startswith('wide_adapters.') for k in missing)
        self.head = head
        if trained_checkpoint:
            self.head.load_state_dict(torch.load(trained_checkpoint, map_location='cpu')['state_dict'], strict=True)
        self.requires_grad_(False)
