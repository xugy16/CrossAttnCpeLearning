import torch
import torch.nn as nn
import torch.nn.functional as F

from CptLearn.Layers import MLP




class GatedElementSumProj(nn.Module):
    """
    Self-weighted  sum
    1. Batch * 14 * Emb --> Batch * 14 * 1 (Weight)
    2. Emb * Wight = Weighted_Sum
    3. we have hops, 3 hops mean we have 3 weights
    5. then  we have 3 weighted sum
    6. [weighted_vec_1, weighted_vec_2, weighted_vec_3] --> flat_out_dim
    """
    def __init__(self, config):
        super(GatedElementSumProj, self).__init__()
        self.config = config

        # ---------------------------------------------------
        # Two-Layer Linear: input_sz --> mid_sz --> out_sz
        # ---------------------------------------------------
        self.self_weight_mlp = MLP(
            in_size=config.hidden_dim,
            mid_size=config.flat_mid_dim,
            out_size=config.flat_hops,
            dropout_r=config.dropout_rate,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            config.hidden_dim * config.flat_hops,
            config.flat_out_dim
        )


    def forward(self, x, x_mask=None):
        att = self.self_weight_mlp(x)

        if x_mask != None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.config.flat_hops):
            # tmp1 = att[:, :, i: i + 1]
            # tmp2 = att[:, :, i: i + 1] * x
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted
