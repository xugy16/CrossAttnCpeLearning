"""
I think the main modifications are:
1. GatedSumCpt  --> [Attr, Obj, Cpt-Attn Img Representation]
    1.1 not gated sum (attr, obj).
    1.2 keep it to  calcuate  the pair matching scores.

2. [Attr, Attr-Attn Img] [Obj, Obj-Attn Img] [PairSum, ImgSum]


Change:
1. self.cpt_ele_summer = GatedElementSummer(config), no cpt summer.


2. delete this part
# --------------------
# PredLayer by Sum
# --------------------
self.pred_norm_by_sum = LayerNorm(config.flat_out_dim)
self.pred_layer_by_sum = nn.Linear(config.flat_out_dim, 1)


3. pred, we should change, we need three scores, inlcuding attr, obj and pair
    3.1 curretnly, we  only  have pair.
    3.2 we have three scores and we can combine the three scores


4. Modification: we only concat the feats.
if self.pred_mode == "sum":
    pred_score = self.predict_score_by_sum(cpt_feat, img_feat, batch_size, self.train_episode_size)
"""


import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from CptLearn.CrossAttnCptLearn.AttnModule import MHCA_EncDec, GatedElementSum
from CptLearn.Layers import LayerNorm
from CptLearn.Modules import ImgEncoder
from CptLearn.Layers import l2norm

from Utils.GloveTools import load_cpt_glove


class MLP(nn.Module):
    """
    clf for attr and obj
    """
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True):
        super(MLP, self).__init__()
        mod = []
        for L in range(num_layers-1):
            mod.append(nn.Linear(inp_dim, inp_dim, bias=bias))
            mod.append(nn.ReLU(True))

        mod.append(nn.Linear(inp_dim, out_dim, bias=bias))
        if relu:
            mod.append(nn.ReLU(True))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output


class ConcatCptNet(nn.Module):
    def __init__(self, config, glove_obj, dset):
        super(ConcatCptNet, self).__init__()
        self.config = config

        # dim
        self.hidden_dim = config.hidden_dim

        # pred_mode: sum or concat
        self.do_combine_learning = config.do_combine_learning
        self.do_adapt_score = config.do_adapt_score

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

        # cpt space --> hidden_space
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
        self.img_ele_summer = GatedElementSum(config)
        self.cpt_ele_summer = GatedElementSum(config)


        # ----------------------
        # Pred Layer by Concat
        # ----------------------
        # Pair Scoring
        self.pred_pair_norm = LayerNorm(config.hidden_dim * 2)
        self.pred_pair_layer = nn.Linear(config.hidden_dim * 2, 1)

        if self.do_combine_learning == 'true':
            # Attr Scoring
            self.pred_attr_norm = LayerNorm(config.hidden_dim * 2)
            self.pred_attr_layer = nn.Linear(config.hidden_dim * 2, 1)

            # Obj Scoring
            self.pred_obj_norm = LayerNorm(config.hidden_dim * 2)
            self.pred_obj_layer = nn.Linear(config.hidden_dim * 2, 1)

            # Activation Func to Combine Scores
            self.act_func = nn.Sequential()
            if config.act_func == "relu":
                self.act_func = nn.ReLU()
            elif config.act_func == "leaky_relu":
                self.act_func = nn.LeakyReLU(0.2)


        # ------------------------
        # adaptive combing scores
        # ------------------------
        if self.do_adapt_score == 'true':
            self.adp_score_layer = nn.Linear(3, 1, bias=False)


        # ------------------------
        # adaptive combing scores
        # ------------------------
        if self.do_combine_learning == 'clf':
            self.attr_clf = MLP(512, len(dset.attr_txt_list), 2, relu=False)
            self.obj_clf = MLP(512, len(dset.obj_txt_list), 2, relu=False)



        # --------------------
        # For Val/Test
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
        # Step 1: Episode Attr/Obj Embedding
        attr_emb = self.attr_embedding(attr_idx.view(-1))
        obj_emb = self.obj_embedding(obj_idx.view(-1))
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



    def attn_enc_dec(self, cpt_feat, img_region_feat):
        """
        Encoding by cross-attn:
        1. Self-attn for visual part.
        2. Cross-attn vis --> cpt
        """
        cpt_feat, img_feat = self.backbone(
            cpt_feat,
            img_region_feat,
            cpt_mask=None,
            img_mask=None
        )
        return cpt_feat, img_feat



    def gated_sum_element(self, cpt_feat, img_feat):
        cpt_feat = self.cpt_ele_summer(cpt_feat)
        img_feat = self.img_ele_summer(img_feat)
        return cpt_feat, img_feat



    def predict_pair_score(self, pair_feat, img_feat, batch_size, episode_size):
        # Reshape Dimension
        pair_feat = pair_feat.view(batch_size, episode_size, self.hidden_dim)
        img_feat = img_feat.view(batch_size, episode_size, self.hidden_dim)
        # Fuse
        concat_feat = torch.cat([pair_feat, img_feat], dim = -1)
        concat_feat = self.pred_pair_norm(concat_feat)
        pred_score = self.pred_pair_layer(concat_feat).squeeze()
        return pred_score



    def predict_attr_score(self, attr_feat, img_feat, batch_size, episode_size):
        # Reshape Dimension
        attr_feat = attr_feat.view(batch_size, episode_size, self.hidden_dim)
        img_feat = img_feat.view(batch_size, episode_size, self.hidden_dim)
        # Fuse
        concat_feat = torch.cat([attr_feat, img_feat], dim = -1)
        concat_feat = self.pred_attr_norm(concat_feat)
        pred_score = self.pred_attr_layer(concat_feat).squeeze()
        return pred_score



    def predict_obj_score(self, obj_feat, img_feat, batch_size, episode_size):
        # Reshape Dimension
        obj_feat = obj_feat.view(batch_size, episode_size, self.hidden_dim)
        img_feat = img_feat.view(batch_size, episode_size, self.hidden_dim)
        # Fuse
        fuse_feat = torch.cat([obj_feat, img_feat], dim = -1)
        fuse_feat = self.pred_obj_norm(fuse_feat)
        pred_score = self.pred_obj_layer(fuse_feat).squeeze()
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
        episode_size = neg_attr_idx_list.shape[1] + 1
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
        cpt_feat, img_feat = self.attn_enc_dec(cpt_feat, img_region_feat)

        # ----------------------------------------------
        # Pair Score Prediction
        # ----------------------------------------------
        # rep
        weight_cpt_feat, weight_img_feat = self.gated_sum_element(cpt_feat, img_feat)
        # score
        pred_pair_score = self.predict_pair_score(weight_cpt_feat, weight_img_feat, batch_size, self.train_episode_size)
        pred_score = pred_pair_score


        # keep positive pairs for further clf
        if self.do_combine_learning == 'clf':
            positive_index = list(range(0, episode_size*batch_size, episode_size))
            positive = cpt_feat[positive_index]


        if self.do_combine_learning == 'score':
            # ----------------------------------------------
            # Attr Score Prediction
            # ----------------------------------------------
            attr_feat = cpt_feat[:, 0, :]
            pred_attr_score = self.predict_attr_score(attr_feat, weight_img_feat, batch_size, self.train_episode_size)

            # ----------------------------------------------
            # Obj Score Prediction
            # ----------------------------------------------
            obj_feat = cpt_feat[:,1,:]
            pred_obj_score = self.predict_attr_score(obj_feat, weight_img_feat, batch_size, self.train_episode_size)

            wandb.log({"pair_ori_score": torch.mean(pred_pair_score[:, 0]),
                       "attr_ori_score": torch.mean(pred_attr_score[:, 0]),
                       "obj_ori_score": torch.mean(pred_obj_score[:, 0])})

            # -----------------------------------
            # Score --> Prob, in the same range
            # -----------------------------------
            pred_score = self.act_func(pred_score)
            pred_attr_score = self.act_func(pred_attr_score)
            pred_obj_score = self.act_func(pred_obj_score)

            with torch.no_grad():
                wandb.log({"pair_act_score": torch.mean(pred_pair_score[:,0]),
                           "attr_act_score": torch.mean(pred_attr_score[:,0]),
                           "obj_act_score": torch.mean(pred_obj_score[:,0])})

            if self.do_adapt_score == 'true':
                comb_score = torch.stack([pred_score, pred_attr_score, pred_obj_score], dim=-1)
                pred_score = self.adp_score_layer(comb_score).squeeze()
            elif self.do_adapt_score == 'false':
                pred_score = pred_score + pred_attr_score + pred_obj_score

        elif self.do_combine_learning == 'clf':
            """
            How to decide positive:
            1. 
            """
            obj_pred = self.obj_clf(positive)
            attr_pred = self.attr_clf(positive)
            loss_aux = F.cross_entropy(attr_pred, attrs) + F.cross_entropy(obj_pred, objs)
            loss.append(self.args.lambda_aux * loss_aux)

        with torch.no_grad():
            wandb.log({"pred_prob": torch.mean(pred_score[:,0])})

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
            # i is the i_th pair concept
            cpt_feat_i = cpt_feat[i].unsqueeze(0)
            # cpt_feat_i is the same, because we attend img to cpt
            cpt_feat_i, img_feat = self.attn_enc_dec(cpt_feat_i, img_region_feat, mode="test")
            # weighted sum
            weight_cpt_feat_i, weight_img_feat = self.gated_sum_element(cpt_feat_i, img_feat)
            # pred_scores
            pred_pair_score = self.predict_pair_score(weight_cpt_feat_i, weight_img_feat, batch_size, 1)
            pred_score  =  pred_pair_score

            if self.do_combine_learning == 'true':
                # ----------------------------------------------
                # Attr Score Prediction
                # ----------------------------------------------
                attr_feat_i = cpt_feat_i[:, 0, :]
                pred_attr_score = self.predict_attr_score(attr_feat_i, weight_img_feat, batch_size, 1)

                # ----------------------------------------------
                # Obj Score Prediction
                # ----------------------------------------------
                obj_feat_i = cpt_feat_i[:, 1, :]
                pred_obj_score = self.predict_attr_score(obj_feat_i, weight_img_feat, batch_size, 1)


                # ----------------------
                # Must follow the same
                # ----------------------
                pred_score = self.act_func(pred_score)
                pred_attr_score = self.act_func(pred_attr_score)
                pred_obj_score = self.act_func(pred_obj_score)

                # ----------------------
                # To Same Scale
                # ----------------------
                if self.do_adapt_score == 'true':
                    comb_score = torch.stack([pred_score, pred_attr_score, pred_obj_score], dim=-1)
                    pred_score = self.adp_score_layer(comb_score).squeeze()
                elif self.do_adapt_score == 'false':
                    pred_score = pred_score + pred_attr_score + pred_obj_score


            dict_AllTxtPair2BatchImgPredScore[((attr_txt, obj_txt))] = pred_score

        return dict_AllTxtPair2BatchImgPredScore
