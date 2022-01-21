"""
Ref:
openvqa --> mcan

this is the key component for the project.
1. Based on Encoder_Decoder framework.
2. Previously:
    2.1 txt-->vis
    2.2 vis-->txt

We should follow SCAN setting:
1. self-encode txt
2. cros-attend img
[txt, cross_img] --> score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from CptLearn.CrossAttnCptLearn.AttnModule import MHCA_EncDec, GatedElementSumProj
from CptLearn.Layers import LayerNorm
from CptLearn.Modules import ImgEncoder
from CptLearn.Layers import l2norm


from Utils.GloveTools import load_cpt_glove


class SumCptNet(nn.Module):
    def __init__(self, config, glove_obj, dset):
        super(SumCptNet, self).__init__()
        self.config = config

        # pred_mode: sum or concat
        self.pred_mode = config.pred_mode

        self.train_episode_size = config.neg_num + 1
        self.test_episode_size = len(dset.all_pair_txt_list)

        self.attr_embedding = nn.Embedding(
            num_embeddings=len(dset.attr_txt_list),
            embedding_dim=300
        )

        self.obj_embedding = nn.Embedding(
            num_embeddings=len(dset.obj_txt_list),
            embedding_dim=300
        )

        # Init with GloVe, but update later
        if config.USE_GLOVE:
            attr_pretrained_weight, _ = load_cpt_glove(dset.attr_txt_list, glove_obj)
            self.attr_embedding.weight.data.copy_(attr_pretrained_weight)

            obj_pretrained_weight, _ = load_cpt_glove(dset.obj_txt_list, glove_obj)
            self.obj_embedding.weight.data.copy_(obj_pretrained_weight)

        self.pair_lstm = nn.LSTM(
            input_size=300,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.img_encoder = ImgEncoder(config).eval()
        self.img_proj = nn.Linear(512, 512)     # resnet_space --> hidden_space

        # ------------------------
        # EncDec for HiddenSpace
        # ------------------------
        self.backbone = MHCA_EncDec(config)

        # --------------------
        # Flatten to vector
        # --------------------
        self.img_ele_summer = GatedElementSumProj(config)
        self.cpt_ele_summer = GatedElementSumProj(config)

        # --------------------
        # PredLayer by Sum
        # --------------------
        self.pred_norm_by_sum = LayerNorm(config.flat_out_dim)
        self.pred_layer_by_sum = nn.Linear(config.flat_out_dim, 1)


        # --------------------
        # PredLayer by Concat
        # --------------------
        self.pred_norm_by_concat = LayerNorm(config.flat_out_dim*2)
        self.pred_layer_by_concat = nn.Linear(config.flat_out_dim*2, 1)


        # --------------------
        # For Validation
        # --------------------
        self.all_pair_txt_list = dset.all_pair_txt_list
        self.all_pair_attr_idx = torch.LongTensor([dset.dict_AttrTxt2Idx[attr] for attr, _ in dset.all_pair_txt_list]).cuda()
        self.all_pair_obj_idx = torch.LongTensor([dset.dict_ObjTxt2Idx[obj] for _, obj in dset.all_pair_txt_list]).cuda()



    def encoding_img(self, img_pixel_feat):
        # batch * 512 * (7,7) --> batch * 49 * 512
        img_vec_feat, img_region_feat = self.img_encoder(img_pixel_feat)
        img_region_feat = img_region_feat.view(img_region_feat.shape[0], img_region_feat.shape[1], -1).permute(0, 2, 1).contiguous()
        img_region_feat = self.img_proj(img_region_feat)

        if self.config.l2_norm:
            img_region_feat = l2norm(img_region_feat, dim=-1)
            img_vec_feat = l2norm(img_vec_feat, dim=-1)

        return img_vec_feat, img_region_feat



    def encoding_cpt(self, attr_idx, obj_idx):
        #  Meta_infor
        if len(attr_idx.shape)  == 1:
            batch_size = 1
            episode_size = attr_idx.shape[0]
        else:
            batch_size = attr_idx.shape[0]
            episode_size = attr_idx.shape[1]

        # Step 1: Episode Attr/Obj Embedding
        attr_emb = self.attr_embedding(attr_idx.view(-1)).view(batch_size * episode_size, -1)
        obj_emb = self.obj_embedding(obj_idx.view(-1)).view(batch_size * episode_size, -1)
        cpt_emb = torch.cat([attr_emb.unsqueeze(1), obj_emb.unsqueeze(1)], dim=1)

        # Step 2: LSTM Encoding
        cpt_feat, ((ht, ct)) = self.pair_lstm(cpt_emb)
        # Bidirectional --> Average
        cpt_feat = (cpt_feat[:, :, :int(cpt_feat.size(2) / 2)] + cpt_feat[:, :, int(cpt_feat.size(2) / 2):]) / 2
        ht = (ht[0, :, :] + ht[1, :, :])/2

        # L2-Norm
        if self.config.l2_norm:
            cpt_feat = l2norm(cpt_feat, dim=-1)
            ht = l2norm(ht, dim=-1)

        return ht, cpt_feat



    def attn_fusion(self, cpt_feat, img_region_feat, mode):
        """
        Encoding by cross-attn:
        1. Self-attn for visual part.
        2. Cross-attn vis --> cpt
        """
        cpt_feat, img_feat = self.backbone(
            cpt_feat,
            img_region_feat,
            cpt_mask=None,
            img_mask=None,
            mode=mode
        )
        return cpt_feat, img_feat



    def gated_sum_element(self, cpt_feat, img_feat):
        cpt_feat = self.cpt_ele_summer(cpt_feat)
        img_feat = self.img_ele_summer(img_feat)
        return cpt_feat, img_feat



    def predict_score_by_sum(self, cpt_feat, img_feat, batch_size, episode_size):
        # Reshape Dimension
        cpt_feat = cpt_feat.view(batch_size, episode_size, self.config.flat_out_dim)
        img_feat = img_feat.view(batch_size, episode_size, self.config.flat_out_dim)
        # Fuse
        fuse_feat = cpt_feat + img_feat
        fuse_feat = self.pred_norm_by_sum(fuse_feat)
        pred_score = self.pred_layer_by_sum(fuse_feat).squeeze()
        return pred_score



    def predict_score_by_concat(self, cpt_feat, img_feat, batch_size, episode_size):
        # Reshape Dimension
        cpt_feat = cpt_feat.view(batch_size, episode_size, self.config.flat_out_dim)
        img_feat = img_feat.view(batch_size, episode_size, self.config.flat_out_dim)
        # Fuse
        fuse_feat = torch.cat([cpt_feat, img_feat], dim = -1)
        fuse_feat = self.pred_norm_by_concat(fuse_feat)
        pred_score = self.pred_layer_by_concat(fuse_feat).squeeze()
        return pred_score



    def forward(self, img_pixel_feat, pos_attr_idx, pos_obj_idx, neg_attr_idx_list, neg_obj_idx_list):
        """
        1. neg_num samples + one positive sample;
        2. rel_net to calculate the engergy value;
        """
        # MetaInfor
        batch_size = img_pixel_feat.shape[0]

        # ------------------------
        # Cpt Encoder
        # ------------------------
        # Episode Attr/Obj Idx
        attr_idx = torch.cat((pos_attr_idx.unsqueeze(1), neg_attr_idx_list), 1)
        obj_idx = torch.cat((pos_obj_idx.unsqueeze(1), neg_obj_idx_list), 1)
        ht, cpt_feat = self.encoding_cpt(attr_idx, obj_idx)

        # ------------------------
        # Img Encoder
        # ------------------------
        img_vec_feat, img_region_feat = self.encoding_img(img_pixel_feat)

        # ------------------------
        # Fusion Part
        # ------------------------
        cpt_feat, img_feat = self.attn_fusion(cpt_feat, img_region_feat, mode="train")

        # ----------------------------------------------
        # Final Representation: ElementRep --> SumRep
        # ----------------------------------------------
        cpt_feat, img_feat = self.gated_sum_element(cpt_feat, img_feat)

        # -------------------------------------
        # How to combine cpt and img output.
        # -------------------------------------
        if self.pred_mode == "sum":
            pred_score = self.predict_score_by_sum(cpt_feat, img_feat, batch_size, self.train_episode_size)
        elif self.pred_mode == "concat":
            pred_score = self.predict_score_by_concat(cpt_feat, img_feat, batch_size, self.train_episode_size)

        return pred_score



    def forward_for_test(self, img_pixel_feat):
        """
        Given an image, any pair can be matched;
        1. For MITStates, we have 1926 pairs (1276 + 700).
        2. image --> [1926]

        For test/val, we need to transpose the table:
        [1926] --> [batch_dim]
        """
        batch_size = img_pixel_feat.shape[0]

        # ---------------------------------
        # Img Encoder using fixed Resnet18
        # ---------------------------------
        img_vec_feat, img_region_feat = self.encoding_img(img_pixel_feat)


        # ------------------------
        # Cpt using updated LSTM
        # ------------------------
        ht, cpt_feat = self.encoding_cpt(self.all_pair_attr_idx, self.all_pair_obj_idx)


        # --------------------
        # Scan each pair
        # --------------------
        dict_AllTxtPair2BatchImgPredScore = {}
        for i, (attr_txt, obj_txt) in enumerate(self.all_pair_txt_list):
            cpt_feat_i = cpt_feat[i].unsqueeze(0)
            cpt_feat_i, img_feat = self.attn_fusion(cpt_feat_i,  img_region_feat, mode="test")
            cpt_feat_i, img_feat = self.gated_sum_element(cpt_feat_i, img_feat)

            # How to combine cpt and img output.
            if self.pred_mode == "sum":
                pred_score = self.predict_score_by_sum(cpt_feat_i, img_feat, batch_size, 1)
            elif self.pred_mode == "concat":
                pred_score = self.predict_score_by_concat(cpt_feat_i, img_feat, batch_size, 1)

            dict_AllTxtPair2BatchImgPredScore[((attr_txt, obj_txt))] = pred_score

        return dict_AllTxtPair2BatchImgPredScore
