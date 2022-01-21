"""
Here is an easy solution to combine the SelfAttn and CrossAttn
1. Basically, self_attn and cross_attn both depend on MHAttn (MultiHeadAttn)
2. Different Bert
    2.1 not pararrel.
    2.2 encode txt first, and then combine the visual feats.
"""
import torch.nn as nn

from CptLearn.Layers import FFN, LayerNorm
from CptLearn.CrossAttnCptLearn.AttnModule import MHAttn



class SelfAttn(nn.Module):
    """
    Multi-Head Self Attn
    """
    def __init__(self, config):
        super(SelfAttn, self).__init__()

        self.mhatt = MHAttn(config)
        self.ffn = FFN(config)

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.norm1 = LayerNorm(config.hidden_dim)

        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.norm2 = LayerNorm(config.hidden_dim)


    def forward(self, x, x_mask):

        # Step 1: self attn
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        # Step 2: residual conncection
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x



class CrossAttn(nn.Module):
    """
    Multi-Head Cross Attn
    """
    def __init__(self, config):
        super(CrossAttn, self).__init__()

        # Control model complexity
        self.vis_self_attn = config.vis_self_attn

        self.self_attn = MHAttn(config)
        self.cross_attn = MHAttn(config)
        self.ffn = FFN(config)

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.self_attn_layer_norm = LayerNorm(config.hidden_dim)

        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.cross_attn_layer_norm = LayerNorm(config.hidden_dim)

        self.dropout3 = nn.Dropout(config.dropout_rate)
        self.skipconn_layer_norm = LayerNorm(config.hidden_dim)


    def forward(self, x, y, x_mask, y_mask):
        """
        Train Setting:
        1. x --> img:             batch_sz,                     49, 512
        2. y --> cpt: batch_sz * episode_size (64 * 51 = 3264),  2, 512

        Test Setting:
        1. x --> img:             batch_sz,                     49, 512
        2. y --> cpt:             batch_sz,                      2, 512
        """
        batch_size = x.shape[0]
        if not self.training:   # test mode
            episode_size = y.shape[0]
            y = y.unsqueeze(0).repeat(batch_size, 1, 1, 1).view(-1, 2, 512)
        else:
            episode_size = int(y.shape[0] / batch_size)

        #---------------------------------------
        # Step 1: vis self attention
        # --------------------------------------
        if self.vis_self_attn == "true":
            x = self.self_attn_layer_norm(x + self.dropout1(
                self.self_attn(qry=x, key=x, val=x, val_mask=x_mask)
            ))

        # ---------------------------------------------
        # Step 2: cross attention: vis attended to cpt
        # ---------------------------------------------
        x = x.unsqueeze(1).repeat_interleave(episode_size, dim=1).view(-1, 49, 512)
        x = self.cross_attn_layer_norm(x + self.dropout2(
            self.cross_attn(qry=x, key=y, val=y, val_mask=y_mask)
        ))

        # ---------------------------------------------
        # Step 3: residual connection
        # ---------------------------------------------
        x = self.skipconn_layer_norm(x + self.dropout3(
            self.ffn(x)
        ))

        return x



class MHCA_EncDec(nn.Module):
    def __init__(self, config):
        super(MHCA_EncDec, self).__init__()

        # Do we need self-attn layer?
        self.cpt_self_attn = config.cpt_self_attn

        # -----------------------
        # Self-Attn for Cpt
        # -----------------------
        self.self_attn_enc_list = nn.ModuleList([SelfAttn(config) for _ in range(config.self_attn_layer_num)])

        # -------------------------
        # Cross Attn for Img
        # 1. Self-Attn for img
        # 2. Cross-Attn for img
        # -------------------------
        self.cross_attn_dec_list = nn.ModuleList([CrossAttn(config) for _ in range(config.cross_attn_layer_num)])



    def forward(self, cpt_feat, img_feat, cpt_mask, img_mask):

        batch_size = img_feat.shape[0]

        # ------------------------------------
        # Self-attn for cpt
        #   1. Get encoder last hidden vector
        # ------------------------------------
        if self.cpt_self_attn == "true":
            for self_attn_enc in self.self_attn_enc_list:
                cpt_feat = self_attn_enc(cpt_feat, cpt_mask)

        # ---------------------------------------------------------------
        # Cross-attn for img feats: representing img by weighted
        #   1. Input encoder last hidden vector
        #   2. And obtain decoder last hidden vectors
        # ---------------------------------------------------------------
        for cross_attn_dec in self.cross_attn_dec_list:
            img_feat = cross_attn_dec(img_feat, cpt_feat, img_mask, cpt_mask)

        # --------------------------------
        #  if test, repeat by ourselves
        # --------------------------------
        if not self.training:
            cpt_feat = cpt_feat.unsqueeze(0).repeat(batch_size, 1, 1, 1).view(-1, 2, 512)

        return cpt_feat, img_feat
