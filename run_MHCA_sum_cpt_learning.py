import torch
import torch.nn as nn
import torchtext.vocab as vocab
import torch.utils.data as tdata
from torch.nn.utils.clip_grad import clip_grad_norm_

from Config.get_config import get_config
from CptLearn.CrossAttnCptLearn.CrossSumModalNet import SumCptNet
from CptLearn.CrossAttnCptLearn.Optimizer import get_optim, adjust_lr
from CptLearn.CrossAttnCptLearn.Evaluation import Evaluator, SlowValidate
from Data.NormalDS import CptDataSet

import wandb



if __name__ == '__main__':

    # --------------------
    #  Config
    # --------------------
    config = get_config()
    wandb.init(project="simple_cpt_learn",
               entity="xugy007",
               config=config)
    config = wandb.config
    print(config)

    # --------------------
    #  Config
    # --------------------
    print('Loading dataset........')
    train_ds = CptDataSet(config, phase = "train")
    train_dl = tdata.DataLoader(train_ds, batch_size=config.train_batch_size, shuffle=True, num_workers=config.workers)
    test_ds = CptDataSet(config, phase = "test")
    test_dl = tdata.DataLoader(test_ds, batch_size=config.test_batch_size, shuffle=False, num_workers=config.workers)

    # ------------------------------------------
    # Init Glove to initialize CptNet Emb_Layer
    # ------------------------------------------
    glove_obj = vocab.GloVe(name='6B', dim=300, cache=config.cache_dir)
    net = SumCptNet(config, glove_obj, train_ds)
    net.cuda()
    # count_parameters_require_grads(net)
    # net = nn.DataParallel(net)

    # ---------------------
    # Watch net
    # ---------------------
    wandb.watch(net)

    # ------------------------------------------
    # Loss and Opt
    # ------------------------------------------
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    warmup_adam = get_optim(config, net, len(train_ds))

    # ------------------------------------
    # Evaluation for validate
    # ------------------------------------
    evaluator = Evaluator(test_ds, net, split_mode="close")
    # with torch.no_grad():
    #        validate(evaluator, net, test_dl)


    start_epoch = 0
    global_count = 0
    episode_size = net.train_episode_size
    for epoch in range(start_epoch, config.max_epochs):
        # ------------------------------------------
        # Epoch Training
        # ------------------------------------------
        #  Set to training mode
        net.train()

        #  Adjust LR by Epoch
        if epoch in config.lr_decay_list:
            adjust_lr(warmup_adam, config.lr_decay_rate)

        #  Batch Training
        for batch, (ts_data, txt_data) in enumerate(train_dl):
            global_count += 1
            warmup_adam.zero_grad()

            ts_data = [feat.cuda() for feat in ts_data]
            img_pixel_feat, pos_attr_idx, pos_obj_idx, neg_attr_idx_list, neg_obj_idx_list = \
                ts_data[0], ts_data[1], ts_data[2], ts_data[5], ts_data[6]

            # Pred
            pred_score = net(img_pixel_feat, pos_attr_idx, pos_obj_idx, neg_attr_idx_list, neg_obj_idx_list)

            # Label
            batch_size = ts_data[0].shape[0]
            batch_label = torch.zeros(batch_size).long().cuda()  # CE
            # batch_label = F.one_hot(torch.zeros(batch_size).long(), num_classes = episode_size).float().cuda()

            # Loss
            loss = criterion(pred_score, batch_label)


            # Gradients and Updata
            loss.backward()
            # Grad Clip
            if config.grad_clip > 0:
                clip_grad_norm_(net.parameters(), config.grad_clip)
            # Update
            warmup_adam.step()


            # Monitor
            acc = (pred_score.argmax(1) == torch.zeros(batch_size).long().cuda()).sum().float() / batch_size
            log = "epoch: %d; batch: %d/%d; loss: %.4f; accu: %.4f" % (epoch,
                        batch, len(train_dl), loss.data.item(), acc)
            wandb.log({"batch_loss": loss.data.item(), "batch_accu": acc})
            print(log, flush=True)

            global_count += 1

        # ------------------------------------------
        # Epoch Validation
        # ------------------------------------------
        with torch.no_grad():
            net.eval()
            open_attr_accu, open_obj_accu, closed_pair_accu, open_pair_accu, \
            oracle_obj_pair_accu, open_seen_pair_accu, open_unseen_pair_accu = SlowValidate(evaluator, net, test_dl)
            print(
                '(val) Epoch: %d | Open_Attr_accu: %.4f | Open_Obj_accu: %.4f | Close_Pair_accu: %.4f | Open_Pair_accu: %.4f | OpHM: %.4f | OpAvg: %.4f | open_seen_pair_accu: %.4f | open_unseen_pair_accu: %.4f  | oracle_obj_pair_accu: %.4f | bias: %.3f'
                % (
                    epoch,
                    open_attr_accu,
                    open_obj_accu,
                    closed_pair_accu,
                    open_pair_accu,
                    (open_seen_pair_accu * open_unseen_pair_accu) ** 0.5,
                    (open_seen_pair_accu + open_unseen_pair_accu) * 0.5,
                    open_seen_pair_accu,
                    open_unseen_pair_accu,
                    oracle_obj_pair_accu,
                    config.unseen_bias_score
                ))
            wandb.log({"Close_Pair_accu": closed_pair_accu})
