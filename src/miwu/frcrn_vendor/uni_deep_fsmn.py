"""Inference-only UniDeepFsmn used by the released FRCRN checkpoint."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniDeepFsmn(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=1, hidden_size=None):
        super().__init__()
        self.lorder = lorder
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(
            output_dim, output_dim, (lorder, 1), (1, 1),
            groups=output_dim, bias=False,
        )

    def forward(self, inputs):
        projected = self.project(F.relu(self.linear(inputs)))
        temporal = projected.unsqueeze(1).permute(0, 3, 2, 1)
        padded = F.pad(temporal, [0, 0, self.lorder - 1, 0])
        memory = temporal + self.conv1(padded)
        return inputs + memory.permute(0, 3, 2, 1).squeeze(1)
