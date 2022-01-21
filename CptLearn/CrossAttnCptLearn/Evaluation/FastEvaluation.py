"""
The most difficult part is that:
1. we have a very low speed to evaluation


Follow: IMRAM


tqdm_gen = tqdm.tqdm(test_dl)
for idx, (ts_data, txt_data) in enumerate(tqdm_gen, 1):
    ts_data = [d.cuda() for d in ts_data]
    img_pixel_feat, true_attr, true_obj = ts_data[0], ts_data[1], ts_data[2]
    dict_AllTxtPair2BatchImgPredScore = net.forward_for_test(img_pixel_feat)
    list_dict_AllTxtPair2BatchImgPredScore.append(dict_AllTxtPair2BatchImgPredScore)
    true_attr_idx_list.append(true_attr)
    true_obj_idx_list.append(true_obj)


we have different Evalutions:
1. Init_Encoding
2. Attn_Ecndoiing
3. Scoring
"""

import time
import tqdm
import numpy as np
import torch



def encode_data_before_attn(net, test_dl):
    # DataSet
    dset = test_dl.dataset

    # ----------------------------------
    # Cpt encoding using updated LSTM
    # ----------------------------------
    all_pair_num = len(dset.all_pair_txt_list)
    all_pair_attr_idx = torch.LongTensor([dset.dict_AttrTxt2Idx[attr] for attr, _ in dset.all_pair_txt_list]).cuda()
    all_pair_obj_idx = torch.LongTensor([dset.dict_ObjTxt2Idx[obj] for _, obj in dset.all_pair_txt_list]).cuda()
    cpt_ht, cpt_feat = net.encoding_cpt(all_pair_attr_idx, all_pair_obj_idx)


    # ---------------=-----------------------------------
    # Img encoding: np array to keep all the embeddings
    # ---------------------------------------------------
    # Save to list
    img_vec_feat_list = []
    img_region_feat_list = []
    true_attr_idx_list = []
    true_obj_idx_list = []
    # Scan all the items
    print("Step 1: Encoding cpt and img")
    tqdm_gen = tqdm.tqdm(test_dl)
    for idx, (ts_data, txt_data) in enumerate(tqdm_gen, 1):
        ts_data = [d.cuda() for d in ts_data]
        img_pixel_feat, true_attr, true_obj = ts_data[0], ts_data[1], ts_data[2]
        img_vec_feat, img_region_feat = net.encoding_img(img_pixel_feat)

        # Save img feats
        img_vec_feat_list.append(img_vec_feat.cpu().numpy())
        img_region_feat_list.append(img_region_feat.cpu().numpy())

        # Save labels
        true_attr_idx_list.extend(true_attr.cpu().tolist())
        true_obj_idx_list.extend(true_obj.cpu().tolist())


    # ---------------------
    #  List to np.array
    # ---------------------
    img_vec_feat_np = np.concatenate(img_vec_feat_list, axis=0)
    img_region_feat_np = np.concatenate(img_region_feat_list, axis=0)
    cpt_ht = cpt_ht.cpu().numpy()
    cpt_feat = cpt_feat.cpu().numpy()
    true_attr_idx_np = np.array(true_attr_idx_list)
    true_obj_idx_np = np.array(true_obj_idx_list)
    return img_vec_feat_np, img_region_feat_np, cpt_ht, cpt_feat, true_attr_idx_np, true_obj_idx_np



def block_cross_attn_score_func(net, img_vec_feat, img_region_feat, cpt_ht, cpt_feat, block_size):
    """
    All input are in numpy.array

    Previous work has 3 steps, but we only have one iteration_step
    1. img --> txt, we get img representation.
    2. calculate score
    3. update img using [ori_img, img-->txt]
    We can calcualte the score iteration_step times.
    """
    # Process block by block
    num_img_block = int((len(img_vec_feat) - 1) / block_size) + 1
    num_cpt_block = int((len(cpt_ht) - 1) / block_size) + 1

    print("num_img_block: %d, num_cap_block: %d, block_size: %d" % (num_img_block, num_cpt_block, block_size))

    # --------------------
    # 2 Tables:
    # zappos:    4228  * 112
    # mitstates: 19191 * 700
    # --------------------
    score_table = np.zeros((len(img_vec_feat), len(cpt_ht)))

    # ------------------------------
    # for block imgs:
    #   for each pairs:
    #       calcualte_score
    # ------------------------------
    for img_block_idx in range(num_img_block):
        img_start, img_end = block_size * img_block_idx, min(block_size * (img_block_idx + 1), len(img_vec_feat))
        batch_size = img_end - img_start
        for cpt_block_idx in range(num_cpt_block):
            # 1: cpt index
            cpt_start, cpt_end = block_size * cpt_block_idx, min(block_size * (cpt_block_idx + 1), len(cpt_ht))
            episode_size = cpt_end - cpt_start
            # 2: np --> cuda
            block_img_emb = torch.from_numpy(img_region_feat[img_start:img_end]).cuda()
            block_cpt_emb = torch.from_numpy(cpt_feat[cpt_start:cpt_end]).cuda()
            # 3: sim score
            attn_cpt_feat, attn_img_feat = net.attn_enc_dec(block_cpt_emb, block_img_emb)
            weight_cpt_feat, weight_img_feat = net.gated_sum_element(attn_cpt_feat, attn_img_feat)
            pred_pair_score = net.predict_pair_score(weight_cpt_feat, weight_img_feat, batch_size, episode_size)
            # 4: fill the table
            score_table[img_start:img_end, cpt_start:cpt_end] = pred_pair_score.data.cpu().numpy()
            print("--- ImgBlockNum {},  PairBlockNum {} ---".format(img_block_idx, cpt_block_idx))
    return score_table



def FastValidate(config, evaluator, net, test_dl):

    print("---------------------------- evaluation ----------------------------")

    img_vec_feat, img_region_feat, cpt_ht, cpt_feat, true_attr_idx, true_obj_idx = encode_data_before_attn(net, test_dl)
    print('Images: %d, Captions: %d' % (img_region_feat.shape[0], cpt_feat.shape[0]))


    # -------------------------------
    # we should have a score table
    # 1. Row is img, col is pairs
    # 2. Datasets:
    #    2.1 Zappos:     4228  *  116
    #    2.2 MitStates:  19191 *  700
    # -------------------------------
    start_time = time.time()
    score_table = block_cross_attn_score_func(net, img_vec_feat, img_region_feat, cpt_ht, cpt_feat, config.test_batch_size)
    end_time = time.time()
    print("---------------  Current Time is  {}  -----------------------".format(end_time -  start_time))

    # ---------------------------------
    #  Calculate best unseen acc
    #  1. close setting;
    #  2. open world setting;
    #  3. obj_oralce setting;
    # ---------------------------------
    # Pred_score --> Pred_Pair_Tuple_Idx(pred_attr_idx, pred_obj_idx)
    dict_PredPairTupleIdx = evaluator.get_TuplePairIdx_from_TensorTable(score_table, true_obj_idx)

    # Matching statistics based on different settings
    match_stat_list = evaluator.get_match_stat(dict_PredPairTupleIdx, true_attr_idx, true_obj_idx, topk = 1)

    # Final accuracy
    open_attr_accu, open_obj_accu, closed_pair_accu, open_pair_accu, oracle_obj_pair_accu, open_seen_pair_accu, open_unseen_pair_accu\
        = list(map(torch.mean, match_stat_list))


    return open_attr_accu, open_obj_accu, closed_pair_accu, open_pair_accu, oracle_obj_pair_accu, open_seen_pair_accu, open_unseen_pair_accu
