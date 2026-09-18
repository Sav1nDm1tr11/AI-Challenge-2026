"""The exact patch-only head used in the submitted three-seed ensemble."""

import torch
from torch import nn


class PatchClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["feature_dim"]
        projected = config["projection_dim"]
        classes = len(config["defect_columns"])
        self.dim = dim
        self.classes = classes
        self.projection = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, projected), nn.GELU()
        )
        self.attention = nn.Linear(projected, classes)
        # The zero global branch preserves the submitted checkpoint's LayerNorm
        # and weight layout. Removing it would change the trained function.
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * dim + projected),
            nn.Linear(2 * dim + projected, config["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
        )
        self.defect_head = nn.Linear(config["hidden_dim"], classes)

    def forward(self, patches, return_attention=False):
        tokens = self.projection(patches.flatten(1, 2))
        attention = self.attention(tokens).softmax(dim=1)
        local = torch.einsum("btc,btd->bcd", attention, tokens)
        zeros = torch.zeros(
            (patches.shape[0], self.classes, 2 * self.dim),
            device=patches.device,
            dtype=patches.dtype,
        )
        representation = self.fusion(torch.cat([zeros, local], dim=-1))
        logits = (representation * self.defect_head.weight[None]).sum(
            -1
        ) + self.defect_head.bias
        return (logits, attention) if return_attention else logits
