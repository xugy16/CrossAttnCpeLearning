"""
Inheritate from NormalDS, we have all the methods and members;
1. all sampling methods
2. all meta-infor
"""

import numpy as np
import torch
from Data.NormalDS import CptDataSet


class FineTuneCptDataSet(CptDataSet):
    """
    How to consturct the dataset:
    1. we will keep all the training data.
    2. we will update the test data.
    """

    def __init__(self, config, sample_train_items, pseudo_lableded_test_items):
        """
        cherry-pick dataset:
        1. At certain epoch, we consturct a new dataset
        2. combine both train_set and test_set
        """
        # when call parent cls, we have print some misleaning inforamtion, not matter;
        super(FineTuneCptDataSet, self).__init__(config=config,
                                                 data_dir=config.data_dir,
                                                 phase ='train',
                                                 split_dir='compositional-split')

        self.sampled_train_items = sample_train_items
        self.pseudo_labeled_test_items = pseudo_lableded_test_items
        self.data_items = sample_train_items + pseudo_lableded_test_items

        self.pseudo_test_item_num = len(pseudo_lableded_test_items)
        self.sample_train_item_num = len(sample_train_items)

        print(
            f"fine-tune dataset: train_item_size {len(sample_train_items)}, test_item_size {len(pseudo_lableded_test_items)}")


    def random_sample_negative_pair_from_all(self, attr_txt, obj_txt):
        """
        inductive: negative paris can only from the train set;
        tranductive: from all paris;
        """
        new_attr_txt, new_obj_txt = self.all_pairs_txt_list[np.random.choice(len(self.all_pairs_txt_list))]
        if new_attr_txt == attr_txt and new_obj_txt == obj_txt:
            # we should sample again;
            return self.random_sample_negative_pair_from_all(attr_txt, obj_txt)
        return (self.dict_AttrTxt2Idx[new_attr_txt], self.dict_ObjTxt2Idx[new_obj_txt])


    def __getitem__(self, index):
        """
        we even don't have index parameter;
        """
        # -----------------------------------------------------------
        # item = list of tuple as (img_file, attr_txt, obj_txt)
        # -----------------------------------------------------------
        img_file, attr_txt, obj_txt = self.data_items[index]
        attr_idx = self.dict_AttrTxt2Idx[attr_txt]
        obj_idx = self.dict_ObjTxt2Idx[obj_txt]

        # --------------------------
        #   close and open index
        # --------------------------
        pair_idx_in_close_set = self.dict_TrainPairTxt2Idx[(attr_txt, obj_txt)] \
            if (attr_txt, obj_txt) in self.dict_TrainPairTxt2Idx.keys() else self.dict_TestPairTxt2Idx[(attr_txt, obj_txt)]
        pair_idx_in_open_set = self.dict_AllPairTxt2Idx[(attr_txt, obj_txt)]

        # -----------------------------
        #   sample negative pairs;
        # -----------------------------
        neg_attr_list = []
        neg_obj_list = []
        # random neg pairs;
        if self.sample_strategy == "random":
            for _ in range(self.neg_num):
                neg_attr_idx, neg_obj_idx = self.random_sample_negative_pair_from_all(attr_txt, obj_txt)  # negative example for triplet loss
                neg_obj_list.append(neg_obj_idx)
                neg_attr_list.append(neg_attr_idx)
        # neg pairs with the same attr
        elif self.sample_strategy == "AttrOnly":
            sampled = 0
            obj_txt_list = self.dict_TrainTxtAttr2TxtObjs[attr_txt]
            for _obj_txt in obj_txt_list:
                if _obj_txt != obj_txt:
                    neg_attr_list.append(self.dict_AttrTxt2Idx[attr_txt])
                    neg_obj_list.append(self.dict_ObjTxt2Idx[_obj_txt])
                    sampled += 1
            for _ in range(sampled, self.neg_num):
                neg_attr_idx, neg_obj_idx = self.random_sample_negative_pair_from_all(attr_txt, obj_txt)  # negative example for triplet loss
                neg_obj_list.append(neg_obj_idx)
                neg_attr_list.append(neg_attr_idx)
        # neg pairs with the same obj
        elif self.sample_strategy == "ObjOnly":
            sampled = 0
            attr_txt_list = self.dict_TrainTxtObj2TxtAttrs[obj_txt]
            for _attr_txt in attr_txt_list:
                if _attr_txt != attr_txt:
                    neg_attr_list.append(self.dict_AttrTxt2Idx[_attr_txt])
                    neg_obj_list.append(self.dict_ObjTxt2Idx[obj_txt])
                    sampled += 1
            for _ in range(sampled, self.neg_num):
                neg_attr_idx, neg_obj_idx = self.random_sample_negative_pair_from_all(attr_txt, obj_txt)  # negative example for triplet loss
                neg_obj_list.append(neg_obj_idx)
                neg_attr_list.append(neg_attr_idx)
        # neg pairs with both attr and obj
        elif self.sample_strategy == "AttrAndObj":
            sampled = 0
            # the same attribtue
            obj_txt_list = self.dict_TrainTxtAttr2TxtObjs[attr_txt]
            for _obj_txt in obj_txt_list:
                if _obj_txt != obj_txt:
                    neg_attr_list.append(self.dict_AttrTxt2Idx[attr_txt])
                    neg_obj_list.append(self.dict_ObjTxt2Idx[_obj_txt])
                    sampled += 1
            # the same object
            attr_txt_list = self.dict_TrainTxtObj2TxtAttrs[obj_txt]
            for _attr_txt in attr_txt_list:
                if _attr_txt != attr_txt:
                    neg_attr_list.append(self.dict_AttrTxt2Idx[_attr_txt])
                    neg_obj_list.append(self.dict_ObjTxt2Idx[obj_txt])
                    sampled += 1
            # left;
            for _ in range(sampled, self.neg_num):
                neg_attr_idx, neg_obj_idx = self.random_sample_negative_pair_from_all(attr_txt, obj_txt)  # negative example for triplet loss
                neg_obj_list.append(neg_obj_idx)
                neg_attr_list.append(neg_attr_idx)
        neg_attr_ts = torch.LongTensor(neg_attr_list)
        neg_obj_ts = torch.LongTensor(neg_obj_list)


        # ------------------------------
        # img cnn feat;
        # ------------------------------
        img_cnn_feat = self.activations[img_file]

        # return
        return [img_cnn_feat, attr_idx, obj_idx, pair_idx_in_close_set, pair_idx_in_open_set, neg_attr_ts, neg_obj_ts], \
               [attr_txt, obj_txt, img_file]



    def __len__(self):
        """
        we train using the
        """
        return len(self.data_items)
