import os
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as tdata

from Utils.ImgTools import ImageLoader, imagenet_trans_func
from Utils.IOTools import load_from_pickle, save_to_pickle

import logging
logger = logging.getLogger()

class CptDataSet(tdata.Dataset):

    def __init__(self, config, phase):
        """ CptDataSet """

        # MetaInfor
        self.data_dir = config.data_dir
        self.phase = phase
        self.resnet_dir = config.resnet_dir
        self.neg_num = config.neg_num
        self.sample_strategy = config.sample_strategy
        self.FixResFeat = config.FixResFeat

        # ImgLoader
        self.img_loader = ImageLoader(self.data_dir + '/images')
        self.imgnet_trans_func = imagenet_trans_func("test")


        # check split mode: ["close", "natural"] --> ["compositional-split", "compositional-split-natural"]
        self.split_mode = config.split_mode
        if self.split_mode ==  'close':
            self.split_dir = "compositional-split"
            self.attr_txt_list, self.obj_txt_list, self.all_pair_txt_list, \
            self.train_pair_txt_list, self.test_pair_txt_list = self.parse_split()
        elif self.split_mode == "natural":
            self.split_dir = "compositional-split-natural"
            self.attr_txt_list, self.obj_txt_list, self.all_pair_txt_list, \
            self.train_pair_txt_list, self.val_pair_txt_list, self.test_pair_txt_list = self.parse_split()


        # no assert
        # assert len(set(self.train_pair_txt_list) & set(self.test_pair_txt_list)) == 0, 'pairs in train and in test are not mutually exclusive'

        # Current Dataset
        if self.split_mode == "close":
            self.train_items, self.test_items = self.extract_close_data_items_from_meta_file()
        elif self.split_mode == "natural":
            self.train_items, self.val_items, self.test_items = self.extract_natural_data_items_from_meta_file()

        if self.phase == 'train':
            self.data_items = self.train_items
        elif self.phase == "test":
            self.data_items = self.test_items
        elif self.phase == "val":
            self.data_items = self.val_items

        # Dict
        self.dict_AttrTxt2Idx = {attr_txt: idx for idx, attr_txt in enumerate(self.attr_txt_list)}
        self.dict_ObjTxt2Idx = {obj_txt: idx for idx, obj_txt in enumerate(self.obj_txt_list)}
        self.dict_TrainPairTxt2Idx = {pair_txt: idx for idx, pair_txt in enumerate(self.train_pair_txt_list)}
        self.dict_TestPairTxt2Idx = {pair_txt: idx for idx, pair_txt in enumerate(self.test_pair_txt_list)}
        self.dict_AllPairTxt2Idx = {pair_txt: idx for idx, pair_txt in enumerate(self.all_pair_txt_list)}
        if self.split_mode == "natural":
            self.dict_ValPairTxt2Idx = {pair_txt: idx for idx, pair_txt in enumerate(self.val_pair_txt_list)}

        # Pair2ImgList
        self.dict_TrainTxtPair2ImgList = defaultdict(list)
        for img_file, attr_txt, obj_txt in self.train_items:
            self.dict_TrainTxtPair2ImgList[(attr_txt, obj_txt)].append(img_file)

        self.dict_TestTxtPair2ImgList = defaultdict(list)
        for img_file, attr_txt, obj_txt in self.test_items:
            self.dict_TestTxtPair2ImgList[(attr_txt, obj_txt)].append(img_file)

        if self.split_mode == "natural":
            self.dict_ValTxtPair2ImgList = defaultdict(list)
            for img_file, attr_txt, obj_txt in self.val_items:
                self.dict_ValTxtPair2ImgList[(attr_txt, obj_txt)].append(img_file)

        # ------------------------------------------------------------------
        # Affordance and Effects for Sampling, only from traininng
        # ------------------------------------------------------------------
        self.dict_TrainObj2TrainAttrList = self.collect_attr_list_for_obj()
        self.dict_TrainAttr2TrainObjList = self.collect_obj_list_for_attr()


        # print some information
        if self.split_mode == "close":
            print('# train pairs: %d | # test pairs: %d' % (len(self.train_pair_txt_list), len(self.test_pair_txt_list)))
            print('# train images: %d | # test images: %d' % (len(self.train_items), len(self.test_items)))
        elif self.split_mode == "natural":
            print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(self.train_pair_txt_list), len(self.val_pair_txt_list), len(self.test_pair_txt_list)))
            print('# train images: %d | # val images: %d | # test images: %d' % (len(self.train_items), len(self.val_items), len(self.test_items)))


        # generate 1D image feats
        if config.FixResFeat == 'true':
            # CoNLL:   img_1D_CNN_feats_res18.t7
            # Current: img_1D_feat_res18.t7
            img_res_feat_file = f'{self.data_dir}/img_1D_CNN_feats_res18.t7'
            if not os.path.exists(img_res_feat_file):
                with torch.no_grad():
                    self.gen_img_resnet_feats(img_res_feat_file)
            img_res_feat = torch.load(img_res_feat_file)
            self.dict_Img2ResFeat = dict(zip(img_res_feat['files'], img_res_feat['features']))


    def collect_attr_list_for_obj(self):
        """ affordance of objects """
        train_obj_affordance_file = "{}/train_obj_affordance.pkl".format(self.data_dir)
        if not os.path.exists(train_obj_affordance_file):
            dict_TrainObj2TrainAttrList = {}
            for _obj in self.obj_txt_list:
                candidates = [attr for (_, attr, obj) in self.train_items if obj == _obj]
                dict_TrainObj2TrainAttrList[_obj] = list(set(candidates))
            save_to_pickle(dict_TrainObj2TrainAttrList, train_obj_affordance_file)
        dict_TrainObj2TrainAttrList = load_from_pickle(train_obj_affordance_file)
        return dict_TrainObj2TrainAttrList



    def collect_obj_list_for_attr(self):
        """
        1. for each attr: it can only modify certain objects;
        2. we collect such information from our training data;
        """
        train_attr_effect_file = "{}/train_attr_effect.pkl".format(self.data_dir)
        if not os.path.exists(train_attr_effect_file):
            dict_TrainAttr2TrainObjList = {}
            for _attr in self.attr_txt_list:
                candidates = [obj for (_, attr, obj) in self.train_items if attr == _attr]
                dict_TrainAttr2TrainObjList[_attr] = list(set(candidates))
            save_to_pickle(dict_TrainAttr2TrainObjList, train_attr_effect_file)
        dict_TrainAttr2TrainObjList = load_from_pickle(train_attr_effect_file)
        return dict_TrainAttr2TrainObjList



    def extract_close_data_items_from_meta_file(self):
        """
        Parse metadata.t7
        1. list of dict objects;
        2. [attr_txt, obj_txt, img_file]
        """
        item_list = torch.load(self.data_dir + f'/metadata.t7')
        train_items, test_items = [], []

        # scan list item
        for item in item_list:
            img_file, attr_txt, obj_txt = item['image'], item['attr'], item['obj']
            if attr_txt == 'NA' or (attr_txt, obj_txt) not in self.all_pair_txt_list:
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue
            data_i = [img_file, attr_txt, obj_txt]
            if (attr_txt, obj_txt) in self.train_pair_txt_list:
                train_items.append(data_i)
            elif (attr_txt, obj_txt) in self.test_pair_txt_list:
                test_items.append(data_i)
        return train_items, test_items



    def extract_natural_data_items_from_meta_file(self):
        """
        metadata.t7 is given for mit-states and zaoop:
        1. list of dict objects;
        2. [attr_txt, obj_txt, img_file]
        """
        item_list = torch.load(self.data_dir + f'/metadata_compositional-split-natural.t7')
        train_items, val_items, test_items = [], [], []

        for item in item_list:
            # adding anohter field: item['set']
            img_file, attr_txt, obj_txt, settype = item['image'], item['attr'], item['obj'], item['set']
            if attr_txt == 'NA' or (attr_txt, obj_txt) not in self.all_pair_txt_list or settype == 'NA':
                # ----------------------------------------------
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                # ----------------------------------------------
                continue
            data_i = [img_file, attr_txt, obj_txt]
            if settype == 'train':
                train_items.append(data_i)
            elif settype == 'val':
                val_items.append(data_i)
            else:
                test_items.append(data_i)
        return train_items, val_items, test_items




    def parse_split(self):
        """
        Parse the txt split file
        1. train_pairs.txt
        2. test_pairs.txt
        """
        def parse_pairs(txt_pair_file):
            with open(txt_pair_file, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        # --------------------------
        # common: train and test
        # --------------------------
        train_attr_txt_list, train_obj_txt_list, train_pair_txt_list = \
            parse_pairs('{}/{}/train_pairs.txt'.format(self.data_dir, self.split_dir))
        test_attr_txt_list, test_obj_txt_list, test_pair_txt_list = \
            parse_pairs('{}/{}/test_pairs.txt'.format(self.data_dir, self.split_dir))

        # ---------------------------
        # natural has specific val
        # ---------------------------
        if self.split_mode == "close":
            attr_txt_list = sorted(list(set(train_attr_txt_list + test_attr_txt_list)))
            obj_txt_list = sorted(list(set(train_obj_txt_list + test_obj_txt_list)))
            all_pair_txt_list = sorted(list(set(train_pair_txt_list + test_pair_txt_list)))
            train_pair_txt_list = sorted(list(set(train_pair_txt_list)))
            test_pair_txt_list = sorted(list(set(test_pair_txt_list)))
            return attr_txt_list, obj_txt_list, all_pair_txt_list, train_pair_txt_list, test_pair_txt_list

        elif self.split_mode == "natural":
            # additional step to extract val set.
            val_attr_txt_list, val_obj_txt_list, val_pair_txt_list = \
                parse_pairs('{}/{}/val_pairs.txt'.format(self.data_dir, self.split_dir))
            attr_txt_list = sorted(list(set(train_attr_txt_list + val_attr_txt_list + test_attr_txt_list)))
            obj_txt_list = sorted(list(set(train_obj_txt_list + val_obj_txt_list + test_obj_txt_list)))
            all_pair_txt_list = sorted(list(set(train_pair_txt_list + val_pair_txt_list + test_pair_txt_list)))
            train_pair_txt_list = sorted(list(set(train_pair_txt_list)))
            val_pair_txt_list = sorted(list(set(val_pair_txt_list)))
            test_pair_txt_list = sorted(list(set(test_pair_txt_list)))
            return attr_txt_list, obj_txt_list, all_pair_txt_list, \
                   train_pair_txt_list, val_pair_txt_list, test_pair_txt_list



    def sample_pair(self, attr_txt, obj_txt):
        """ base operation to sample pair. """
        new_attr_txt, new_obj_txt = self.train_pair_txt_list[np.random.choice(len(self.train_pair_txt_list))]
        if new_attr_txt == attr_txt and new_obj_txt == obj_txt:
            # we should sample again;
            return self.sample_pair(attr_txt, obj_txt)
        return self.dict_AttrTxt2Idx[new_attr_txt], self.dict_ObjTxt2Idx[new_obj_txt]



    def neg_sample_random(self, attr_txt, obj_txt):
        neg_attr_idx_list = []
        neg_obj_idx_list = []
        for _ in range(self.neg_num):
            neg_attr_idx, neg_obj_idx = self.sample_pair(attr_txt, obj_txt)  # negative example for triplet loss
            neg_attr_idx_list.append(neg_attr_idx)
            neg_obj_idx_list.append(neg_obj_idx)
        return neg_attr_idx_list, neg_obj_idx_list


    def neg_sample_with_same_obj(self, attr_txt, obj_txt):
        neg_attr_idx_list = []
        neg_obj_idx_list = []

        sampled = 0
        attr_txt_list = self.dict_TrainObj2TrainAttrList[obj_txt]
        for _attr_txt in attr_txt_list:
            if _attr_txt != attr_txt:
                neg_attr_idx_list.append(self.dict_AttrTxt2Idx[_attr_txt])
                neg_obj_idx_list.append(self.dict_ObjTxt2Idx[obj_txt])
                sampled += 1
        for _ in range(sampled, self.neg_num):
            neg_attr_idx, neg_obj_idx = self.sample_pair(attr_txt, obj_txt)  # negative example for triplet loss
            neg_obj_idx_list.append(neg_obj_idx)
            neg_attr_idx_list.append(neg_attr_idx)

        return neg_attr_idx_list, neg_obj_idx_list



    def neg_sample_with_same_attr(self, attr_txt, obj_txt):
        neg_attr_idx_list = []
        neg_obj_idx_list = []

        sampled = 0
        obj_txt_list = self.dict_TrainAttr2TrainObjList[attr_txt]
        for _obj_txt in obj_txt_list:
            if _obj_txt != obj_txt:
                neg_attr_idx_list.append(self.dict_AttrTxt2Idx[attr_txt])
                neg_obj_idx_list.append(self.dict_ObjTxt2Idx[_obj_txt])
                sampled += 1
        for _ in range(sampled, self.neg_num):
            neg_attr_idx, neg_obj_idx = self.sample_pair(attr_txt, obj_txt)  # negative example for triplet loss
            neg_obj_idx_list.append(neg_obj_idx)
            neg_attr_idx_list.append(neg_attr_idx)

        return neg_attr_idx_list, neg_obj_idx_list



    def neg_sample_with_same_attr_and_obj(self, attr_txt, obj_txt):
        neg_attr_idx_list = []
        neg_obj_idx_list = []

        sampled = 0
        # the same attribute
        obj_txt_list = self.dict_TrainAttr2TrainObjList[attr_txt]
        for _obj_txt in obj_txt_list:
            if _obj_txt != obj_txt:
                neg_attr_idx_list.append(self.dict_AttrTxt2Idx[attr_txt])
                neg_obj_idx_list.append(self.dict_ObjTxt2Idx[_obj_txt])
                sampled += 1

        # the same object
        attr_txt_list = self.dict_TrainObj2TrainAttrList[obj_txt]
        for _attr_txt in attr_txt_list:
            if _attr_txt != attr_txt:
                neg_attr_idx_list.append(self.dict_AttrTxt2Idx[_attr_txt])
                neg_obj_idx_list.append(self.dict_ObjTxt2Idx[obj_txt])
                sampled += 1

        # sample the left
        for _ in range(sampled, self.neg_num):
            neg_attr_idx, neg_obj_idx = self.sample_pair(attr_txt, obj_txt)  # negative example for triplet loss
            neg_obj_idx_list.append(neg_obj_idx)
            neg_attr_idx_list.append(neg_attr_idx)

        return neg_attr_idx_list, neg_obj_idx_list


    def __getitem__(self, index):
        # item = list of tuple as (img_file, attr_txt, obj_txt)
        img_file, attr_txt, obj_txt = self.data_items[index]
        img_pixel_feat = self.img_loader(img_file)
        img_pixel_feat = self.imgnet_trans_func(img_pixel_feat)
        img_feat = img_pixel_feat

        if self.FixResFeat == 'true':
            img_feat = self.dict_Img2ResFeat[img_file]

        # txt --> idx
        attr_idx = self.dict_AttrTxt2Idx[attr_txt]
        obj_idx = self.dict_ObjTxt2Idx[obj_txt]
        pair_idx_in_open_set = self.dict_AllPairTxt2Idx[(attr_txt, obj_txt)]

        # neg_sample
        if self.phase == "train":
            if self.sample_strategy == "Random":
                neg_attr_idx_list, neg_obj_list = self.neg_sample_random(attr_txt, obj_txt)
            elif self.sample_strategy == "SameAttr":
                neg_attr_idx_list, neg_obj_list = self.neg_sample_with_same_attr(attr_txt, obj_txt)
            elif self.sample_strategy == "SameObj":
                neg_attr_idx_list, neg_obj_list = self.neg_sample_with_same_obj(attr_txt, obj_txt)
            elif self.sample_strategy == "AttrAndObj":
                neg_attr_idx_list, neg_obj_list = self.neg_sample_with_same_attr_and_obj(attr_txt, obj_txt)
            neg_attr_ts = torch.LongTensor(neg_attr_idx_list)
            neg_obj_ts = torch.LongTensor(neg_obj_list)

            pair_idx_in_train_set = self.dict_TrainPairTxt2Idx[(attr_txt, obj_txt)]
            return [img_feat, attr_idx, obj_idx, pair_idx_in_train_set, pair_idx_in_open_set, neg_attr_ts, neg_obj_ts], \
                   [attr_txt, obj_txt, img_file]

        elif self.phase == "test":
            pair_idx_in_test_set = self.dict_TestPairTxt2Idx[(attr_txt, obj_txt)]
            return [img_feat, attr_idx, obj_idx, pair_idx_in_test_set, pair_idx_in_open_set], \
                   [attr_txt, obj_txt, img_file]


    def __len__(self):
        return len(self.data_items)



    def gen_img_resnet_feats(self, out_file):
        """
        In order to recover our results.
        """
        # 1: data
        data = self.test_items + self.train_items
        # 2: tools
        import tqdm
        from Utils.ImgTools import imagenet_trans_func
        from Utils.misc import chunks
        import torchvision.models as tmodels
        import torch.nn as nn
        transform = imagenet_trans_func('test')
        feat_extractor = tmodels.resnet18(pretrained=True)
        feat_extractor.fc = nn.Sequential()
        feat_extractor.eval().cuda()
        # 3: extract feats
        img_feats = []
        img_files = []
        for chunk in tqdm.tqdm(chunks(data, 512), total=len(data) // 512):
            files, attrs, objs = zip(*chunk)
            imgs = list(map(self.img_loader, files))
            imgs = list(map(transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).cuda())
            img_feats.append(feats.data.cpu())
            img_files += files
        img_feats = torch.cat(img_feats, 0)
        # 4: save feats
        print('features for %d images generated' % (len(img_files)))
        torch.save({'features': img_feats, 'files': img_files}, out_file)

