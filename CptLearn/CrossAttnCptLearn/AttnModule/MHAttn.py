import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHAttn(nn.Module):
    def __init__(self, config):
        super(MHAttn, self).__init__()
        self.config = config

        self.linear_v = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.linear_k = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.linear_q = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.linear_merge = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout_rate)


    def forward(self, qry, key, val, val_mask):
        n_batches = qry.size(0)

        qry = self.linear_q(qry).view(
            n_batches,
            -1,
            self.config.head_num,
            int(self.config.hidden_dim / self.config.head_num)
        ).transpose(1, 2)

        key = self.linear_k(key).view(
            n_batches,
            -1,
            self.config.head_num,
            int(self.config.hidden_dim / self.config.head_num)
        ).transpose(1, 2)

        val = self.linear_v(val).view(
            n_batches,
            -1,
            self.config.head_num,
            int(self.config.hidden_dim / self.config.head_num)
        ).transpose(1, 2)

        attn = self.att(qry, key, val, val_mask)
        attn = attn.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.config.hidden_dim
        )

        attn = self.linear_merge(attn)

        return attn


    def att(self, qry, key, val, val_mask):
        d_k = qry.size(-1)

        scores = torch.matmul(
            qry, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if val_mask is not None:
            scores = scores.masked_fill(val_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        return torch.matmul(attn, val)