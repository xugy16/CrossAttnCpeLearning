import torch

class Evaluator:
    """
    there are two settings:
    """
    def __init__(self, dset, net, split_mode):

        self.dset = dset

        # --------------------------------------------------
        #  text_pairs -> idx tensors:
        # ('sliced', 'apple') --> torch.LongTensor([0,1])
        # --------------------------------------------------=
        # All Pairs
        all_pair_tuple_idx = [(dset.dict_AttrTxt2Idx[attr], dset.dict_ObjTxt2Idx[obj])
                              for attr, obj in dset.all_pair_txt_list]
        self.all_pair_tuple_idx = torch.LongTensor(all_pair_tuple_idx)

        # Train Pairs
        train_pair_tuple_idx = [(dset.dict_AttrTxt2Idx[attr], dset.dict_ObjTxt2Idx[obj])
                                for attr, obj in dset.train_pair_txt_list]
        self.train_pair_tuple_idx_list = train_pair_tuple_idx


        # Test Pairs
        if dset.phase == 'train':
            print('Evaluation based on train pairs')
            test_pair_set = set(dset.train_pair_txt_list)
        elif dset.phase == 'val':
            print('Evaluation based on val pairs')
            test_pair_set = set(dset.val_pair_txt_list + dset.train_pair_txt_list)
        elif dset.phase == 'test':
            print('Evaluation based on test pairs')
            if split_mode == "natural":
                test_pair_set = set(dset.test_pair_txt_list + dset.train_pair_txt_list)
            elif split_mode == "close":
                test_pair_set = set(dset.test_pair_txt_list)
        self.test_pair_tuple_idx_list = [(dset.dict_AttrTxt2Idx[attr], dset.dict_ObjTxt2Idx[obj])
                                         for attr, obj in list(test_pair_set)]


        # --------------------------
        #  close zsl mask
        # --------------------------
        mask = [1 if pair in test_pair_set else 0 for pair in dset.all_pair_txt_list]
        self.test_pair_mask = torch.ByteTensor(mask)


        # --------------------------
        #  seen pair mask
        # --------------------------
        train_pair_set = set(dset.train_pair_txt_list)
        mask = [1 if pair in train_pair_set else 0 for pair in dset.all_pair_txt_list]
        self.train_pair_mask = torch.ByteTensor(mask)


        # -------------------------------------------------------------------------------
        # object specific mask over which pairs occur in the object oracle setting
        # -------------------------------------------------------------------------------
        oracle_obj_mask = []
        for _obj in dset.obj_txt_list:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.all_pair_txt_list]
            oracle_obj_mask.append(torch.ByteTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)      # obj_size * pair_num



    def generate_predictions(self, ts_Biased_score_TestImg2AllTxtPair, obj_truth):  # (B, #pairs)
        """
        generate masks for each setting, mask scores, and get prediction labels
        """
        def get_pred_from_tensor_scores(_ts_scores):
            """
            just get top 10 scores;
            """
            _, pred_pair_idx = _ts_scores.topk(10, dim=1)  # sort(1, descending=True)
            pred_pair_idx = pred_pair_idx[:, :10].contiguous().view(-1)
            pred_attr_idx, pred_obj_idx = self.all_pair_tuple_idx[pred_pair_idx][:, 0].view(-1, 10), self.all_pair_tuple_idx[pred_pair_idx][:, 1].view(-1, 10)
            return (pred_attr_idx, pred_obj_idx)

        # return dict of results;
        dict_PredPairTupleIdx = {}

        # -------------------------------------------------------------------------- #
        # close world setting --
        #       set the score for all NON test-pairs to -1e10;
        #       then only the test pairs are into consideration;
        # -------------------------------------------------------------------------- #
        closed_pos = self.test_pair_mask.repeat(ts_Biased_score_TestImg2AllTxtPair.shape[0], 1)
        biasd_closed_scores = ts_Biased_score_TestImg2AllTxtPair.clone()
        biasd_closed_scores[1 - closed_pos] = -1e10
        dict_PredPairTupleIdx.update({'closed': get_pred_from_tensor_scores(biasd_closed_scores)})


        # ----------------------------------------------------- #
        # open world setting -- no mask
        #       not manipulate the score;
        #       original score to determine the (attr,obj) pair
        # ----------------------------------------------------- #
        open_scores = ts_Biased_score_TestImg2AllTxtPair.clone()
        dict_PredPairTupleIdx.update({'open': get_pred_from_tensor_scores(open_scores)})


        # ---------------------------------------------------------------------------------------------------------- #
        # object_oracle setting - set the score to -1e10 for all pairs where no true objects
        # ---------------------------------------------------------------------------------------------------------- #
        oracle_obj_pos = self.oracle_obj_mask[obj_truth]
        oracle_obj_scores = ts_Biased_score_TestImg2AllTxtPair.clone()
        oracle_obj_scores[1 - oracle_obj_pos] = -1e10
        dict_PredPairTupleIdx.update({'object_oracle': get_pred_from_tensor_scores(oracle_obj_scores)})

        return dict_PredPairTupleIdx



    def get_TuplePairIdx_from_Dict(self, dict_AllTxtPair2AllTestImgScore, obj_truth, unseen_bias_score = 0.0):
        """
        Basically, this funciton is only used in SlowValidate
        1. Score_Table.
        2. Add bias to do prediction.
        """

        # GPU --> CPU
        dict_AllTxtPair2AllTestImgScore = {txt_pair: test_pred_score.cpu()
                                           for txt_pair, test_pred_score in dict_AllTxtPair2AllTestImgScore.items()}
        obj_truth = obj_truth.cpu()

        # --------------------------------
        # Dict_Pair2Img --> ts_Img2Pair
        # --------------------------------
        ts_score_TestImg2AllTxtPair = torch.stack([dict_AllTxtPair2AllTestImgScore[(attr, obj)]
                                                   for attr, obj in self.dset.all_pair_txt_list], 1)



        # -----------------------------
        # Adding bias to unseen pairs
        # -----------------------------
        ts_Biased_score_TestImg2AllTxtPair = ts_score_TestImg2AllTxtPair.clone()
        train_pair_pos = self.train_pair_mask.repeat(ts_score_TestImg2AllTxtPair.shape[0], 1)
        ts_Biased_score_TestImg2AllTxtPair[1 - train_pair_pos] += unseen_bias_score
        dict_PredPairTupleIdx = self.generate_predictions(ts_Biased_score_TestImg2AllTxtPair, obj_truth)


        # ------------------------------
        # Saving score tensor
        # ------------------------------
        dict_PredPairTupleIdx['biased_score_ts'] = ts_Biased_score_TestImg2AllTxtPair
        dict_PredPairTupleIdx['ori_score_ts'] = ts_score_TestImg2AllTxtPair


        """
        Then we have five items in the dict:
        1. closed setting
        2. open seting
        3. object setting
        4. original score
        5. biasd score
        """
        return dict_PredPairTupleIdx




    def get_TuplePairIdx_from_TensorTable(self, ts_TestImg2AllPair, obj_truth, unseen_bias_score=0.0):
        """
        Ths same function as score_clf_model and score_manifold_model
        1. Difference is that we have tensor as input
        2. For fast_validate
        """
        # -----------------------------
        # Adding bias to unseen pairs
        # -----------------------------
        ts_Biased_score_TestImg2AllTxtPair = ts_TestImg2AllPair.clone()
        train_pair_pos = self.train_pair_mask.repeat(ts_TestImg2AllPair.shape[0], 1)
        ts_Biased_score_TestImg2AllTxtPair[1 - train_pair_pos] += unseen_bias_score
        dict_PredPairTupleIdx = self.generate_predictions(ts_Biased_score_TestImg2AllTxtPair, obj_truth)

        return dict_PredPairTupleIdx




    def get_match_stat(self, dict_PredPairIdx, true_attr, obj_truth, topk):
        """
        Totally Three Settings:
        1. open: select pairs from all candidate pairs.
        2. close:
        3. obj_oracle: select pairs which set with same obj.
        """
        test_item_num = len(true_attr)

        # gpu --> cpu
        true_attr, obj_truth = true_attr.cpu(), obj_truth.cpu()
        true_pair_tuple_idx_list = list(zip(list(true_attr.cpu().numpy()), list(obj_truth.cpu().numpy())))


        """
        test imgs: 
        1. close_split:    can only from unseen pairs.
        2. natural_split:  can from [seen, unseen] pairs.
        """
        test_seen_img_idx = torch.LongTensor([i for i in range(test_item_num) if true_pair_tuple_idx_list[i] in self.train_pair_tuple_idx_list])
        test_unseen_img_idx = torch.LongTensor([i for i in range(test_item_num) if true_pair_tuple_idx_list[i] not in self.train_pair_tuple_idx_list])


        # --------------------------------------------
        # open world: attribute, object and pair
        # --------------------------------------------
        open_attr_pred = dict_PredPairIdx['open'][0]
        open_obj_pred= dict_PredPairIdx['open'][1]
        open_attr_match = (true_attr.unsqueeze(1).repeat(1, topk) == open_attr_pred[:, :topk])
        open_obj_match = (obj_truth.unsqueeze(1).repeat(1, topk) == open_obj_pred[:, :topk])


        # --------------------------------------
        # From open world, we have 3 matches:
        #   1. open match stats;
        #   2. attr match stats;
        #   3. obj match stats;
        # --------------------------------------
        open_pair_match = (open_attr_match * open_obj_match).any(1).float()
        open_attr_match = open_attr_match.any(1).float()
        open_obj_match = open_obj_match.any(1).float()

        open_seen_match = open_pair_match[test_seen_img_idx]
        open_unseen_match = open_pair_match[test_unseen_img_idx]


        # ---------------------------
        # closed world, only top 1;
        # ---------------------------
        close_attr_pred = dict_PredPairIdx['closed'][0]
        close_obj_pred = dict_PredPairIdx['closed'][1]
        closed_match = (true_attr == close_attr_pred[:, 0]).float() * (obj_truth == close_obj_pred[:, 0]).float()


        # ---------------------------
        # obj_oracle: pair
        # ---------------------------
        oracle_attr_pred = dict_PredPairIdx['object_oracle'][0]
        oracle_obj_pred = dict_PredPairIdx['object_oracle'][1]
        oracle_obj_match = (true_attr == oracle_attr_pred[:, 0]).float() * (obj_truth == oracle_obj_pred[:, 0]).float()


        # ------------------------------------------------------------------------------------
        # We must make a clearication here:
        # 1. open_seen_match
        # 2. open_unseen_match
        # We don't see the images, but the images can be from seen_pairs or unseen_pairs.
        # ------------------------------------------------------------------------------------
        return open_attr_match, open_obj_match, closed_match, open_pair_match, oracle_obj_match, open_seen_match, open_unseen_match