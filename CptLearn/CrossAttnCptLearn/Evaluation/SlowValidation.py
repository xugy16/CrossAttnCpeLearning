import tqdm
import torch


def SlowValidate(evaluator, net, test_dl):

    print("-------- evaluation --------")

    true_attr_idx_list = []
    true_obj_idx_list = []
    list_dict_AllTxtPair2BatchImgPredScore = []


    # ---------------------------------------------------------------------
    # txt_pair --> test_img_score
    # E.x., we have 1926 pairs and 700 test images with batch_size = 100
    # 1. dict_AllTxtPair2BatchImgPredScore: 1926_pairs --> BatchScore
    # 2. list_dict_AllTxtPair2BatchImgPredScore: length is 7
    # ---------------------------------------------------------------------
    tqdm_gen = tqdm.tqdm(test_dl)
    for idx, (ts_data, txt_data) in enumerate(tqdm_gen, 1):
        ts_data = [d.cuda() for d in ts_data]
        img_pixel_feat, true_attr, true_obj = ts_data[0], ts_data[1], ts_data[2]
        dict_AllTxtPair2BatchImgPredScore = net.forward_for_test(img_pixel_feat)
        list_dict_AllTxtPair2BatchImgPredScore.append(dict_AllTxtPair2BatchImgPredScore)
        true_attr_idx_list.append(true_attr)
        true_obj_idx_list.append(true_obj)



    #--------------------------------
    # list to tensor for True Labels
    # -------------------------------
    true_attr_idx_list = torch.cat(true_attr_idx_list)
    true_obj_idx_list = torch.cat(true_obj_idx_list)


    # -------------------------------------------------------------
    # Reorganize: ListOfDict_BatchPred -->  DictOfList_AllPred
    # -------------------------------------------------------------
    dict_AllTxtPair2AllTestPredScore = {}
    all_txt_pair_list = list_dict_AllTxtPair2BatchImgPredScore[0].keys()
    for txt_pair in all_txt_pair_list:
        dict_AllTxtPair2AllTestPredScore[txt_pair] = torch.cat([list_dict_AllTxtPair2BatchImgPredScore[i][txt_pair] for i in range(len(list_dict_AllTxtPair2BatchImgPredScore))])


    # ---------------------------------
    #  Calculate best unseen acc
    #  1. close setting;
    #  2. open world setting;
    #  3. obj_oralce setting;
    # ---------------------------------
    # Pred_score --> Pred_Pair_Tuple_Idx(pred_attr_idx, pred_obj_idx)
    dict_PredPairTupleIdx = evaluator.get_TuplePairIdx_from_score(dict_AllTxtPair2AllTestPredScore, true_obj_idx_list)

    # Matching statistics based on different settings
    match_stat_list = evaluator.get_match_stat(dict_PredPairTupleIdx, true_attr_idx_list, true_obj_idx_list, topk = 1)

    # Final accuracy
    open_attr_accu, open_obj_accu, closed_pair_accu, open_pair_accu, oracle_obj_pair_accu, open_seen_pair_accu, open_unseen_pair_accu\
        = list(map(torch.mean, match_stat_list))


    return open_attr_accu, open_obj_accu, closed_pair_accu, open_pair_accu, oracle_obj_pair_accu, open_seen_pair_accu, open_unseen_pair_accu
