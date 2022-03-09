"""
If the last dimension all equal to zero.
1. mask in [True, False]
2.
"""

import torch

# Masking the sequence mask
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)