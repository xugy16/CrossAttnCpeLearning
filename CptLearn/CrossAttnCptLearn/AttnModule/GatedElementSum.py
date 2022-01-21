"""
1. We only have hop=1,  so we don't need to modify this part.


2. delete projector.
self.linear_merge = nn.Linear(
    config.hidden_dim * config.flat_hops,
    config.flat_out_dim
)


3. delete this step in forward.
x_atted = self.linear_merge(x_atted)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from CptLearn.Layers import MLP



class GatedElementSum(nn.Module):
    """
    Self-weighted  sum
    1. Batch * 14 * Emb --> Batch * 14 * 1 (Weight)
    2. Emb * Wight = Weighted_Sum
    """
    def __init__(self, config):
        super(GatedElementSum, self).__init__()
        self.config = config

        # Two-Layer Linear
        self.self_weight_mlp = MLP(
            in_size=config.hidden_dim,
            mid_size=config.flat_mid_dim,
            out_size=config.flat_hops,
            dropout_r=config.dropout_rate,
            use_relu=True
        )



    def forward(self, x, x_mask=None):
        """
        Take img for example, in this step.
        1. img: batch_sz * episode_sz, 49, 512
        2. MLP --> batch_sz * episode_sz, 49, 1
        3. Softmax --> weight
        4. weight sum  49  elements.
        """
        att = self.self_weight_mlp(x)

        if x_mask != None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        attn_list = []
        for i in range(self.config.flat_hops):
            # tmp1 = att[:, :, i: i + 1]
            # tmp2 = att[:, :, i: i + 1] * x
            attn_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(attn_list, dim=1)

        return x_atted
