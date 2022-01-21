import os
import argparse



def get_config():

    parser = argparse.ArgumentParser()

    parser.add_argument('--RUN', dest='RUN_MODE',
                        choices=['ind_train', 'tran_train', 'val', 'test'],
                        help='{ind_train, tran_train, val, test}',
                        type=str, required=True)
    parser.add_argument('--Seed', type=int, help='manual seed')

    # ----------------
    # Random Seed
    # ----------------
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # --------------------------
    # Recover previous results
    # --------------------------
    parser.add_argument('--FixResFeat', type=str, default="false")

    # --------------------------
    # do combine learning
    # --------------------------
    parser.add_argument('--do_combine_learning', type=str, default="clf", choices=["clf","score"])
    parser.add_argument('--do_adapt_score', type=str, default="false")
    parser.add_argument('--act_func', type=str, default="identity", choices=["identity", "relu", "leaky_relu"])

    # ----------------
    # Dataset
    # ----------------
    # 1. Dataset
    parser.add_argument('--dataset', default='mitstates',
                        choices=['mitstates', 'zappos'], help='dataset')
    # 2.Split
    parser.add_argument('--split_mode', type=str,
                        default="close",
                        choices=["natural", "close"])
    parser.add_argument("--neg_num", type=int, default=100)
    parser.add_argument('--sample_strategy',
                        type=str, default="Random",
                        choices=["Random", "SameAttr", "SameObj", "AttrAndObj"])


    # --------------------------
    # MultiHead Model Setting
    # --------------------------
    # 1. Txt Part
    parser.add_argument('--txt_dim',
                        type=int, default=300,
                        help='input dim of the embedding network, '
                             'For MIT-States and UT-Zappos, '
                             'the embedding dim is 600')
    parser.add_argument('--USE_GLOVE', action='store_false',
                        help='initialize inputs with word vectors')
    parser.add_argument('--fix_emb', action='store_false',
                        help='do not optimize input representations')
    # 2. Vis Part
    parser.add_argument("--vis_dim", type=int, default=512)
    # 3. Latent-Space Part
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--mid_dim", type=int, default=2048)
    parser.add_argument('--out_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=512)
    # 4. MultiHead Attention Part
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--cross_attn_layer_num', type=int, default=1)
    parser.add_argument('--self_attn_layer_num', type=int, default=1)
    # 6: Dropout Rate
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    # 7: GatedElementSummer
    parser.add_argument('--flat_mid_dim', type=int, default=512)
    parser.add_argument('--flat_out_dim', type=int, default=1024)
    parser.add_argument('--flat_hops', type=int, default=1)
    # 8: pred_mode
    parser.add_argument('--pred_mode', type=str,
                        default="concat",
                        choices=["sum", "concat"])
    parser.add_argument('--vis_self_attn', type=str,
                        default="false",
                        choices=["true", "false"])
    parser.add_argument('--cpt_self_attn', type=str,
                        default="false",
                        choices=["true", "false"])

    # --------------------------
    # Attn Model Setting
    # --------------------------
    parser.add_argument('--embed_size', type=int, default=1024)
    parser.add_argument('--l2_norm', action='store_false')
    parser.add_argument('--attn_mode', type=str, default='full', choices=["full","txt2img","img2txt"])
    parser.add_argument('--ele_summer', type=str, default='weight', choices=["weight", "max", "avg", "max_avg"])
    parser.add_argument('--self_attn', type=str, default='false', choices=["true", "false"])
    parser.add_argument('--fuse_step', type=int, default=1)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--max_violation', action='store_true')
    parser.add_argument('--cos_attn_lr', type=float, default=0.0005)
    parser.add_argument('--cos_attn_lr_update', type=int, default=5)

    # ---------------------------
    # concat option
    # ---------------------------
    parser.add_argument('--concat_all', type=str, default='true', choices=["false", "true"])

    # ---------------------------------------------
    # SCAN: just keep, and we can modify later.
    # ---------------------------------------------
    parser.add_argument('--iteration_step', type=int, default=1)
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')

    # --------------------------
    # Simple Model Setting
    # --------------------------
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--simple_lr', type=float, default=0.001)
    parser.add_argument('--simple_lr_update', type=int, default=10)



    # --------------------
    # optimizer setting
    # --------------------
    parser.add_argument('--warmup_epoch', type=int,
                        default=3)
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.2)
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    # 1. Ind Training
    parser.add_argument('--ind_lr',
                        default=1e-4, type=float,
                        help='initial learning rate for new parameters')
    parser.add_argument('--ind_lr_gamma',
                        type=float, default=0.5)
    parser.add_argument('--ind_lr_update', type=int,
                        default=10)
    # 2. Trans Training
    parser.add_argument('--trans_lr',
                        default=3e-5, type=float,
                        help='initial learning rate for new parameters')
    parser.add_argument('--trans_lr_gamma',
                        type=float, default=0.5)
    parser.add_argument('--trans_lr_update',
                        type=int, default=10)
    parser.add_argument("--gce_q", type=float, default=0.5)
    parser.add_argument("--fine_tune_train_item_num",
                        type=int, default=70)
    parser.add_argument("--fine_tune_test_item_num",
                        type=int, default=1)
    parser.add_argument('--cherry_pick_interval',
                        type=int, default=10)


    # ---------------
    # Training
    # ---------------
    parser.add_argument('--workers', default=8, type=int,
                        help='number of workers')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='evaluate every k epochs')
    parser.add_argument('--train_batch_size',
                        type=int, default=128)
    parser.add_argument('--val_batch_size',
                        type=int, default=128)
    parser.add_argument('--test_batch_size',
                        type=int, default=128)
    parser.add_argument('--max_epochs',
                        type=int, default=25)
    parser.add_argument('--eval_interval',
                        type=int, default=5)


    # -----------
    # Evaluation
    # -----------
    parser.add_argument('--unseen_bias_score',
                        type=float, default=0)
    parser.add_argument('--auc_mode',
                        type=str, default="test",
                        choices = ["val", "test"])
    parser.add_argument("--topK",
                        type=int, default=1)

    config = parser.parse_args()

    # -----------------------------
    # Warmup Opt
    # -----------------------------
    config.lr_decay_list = [15, 20, 30]

    # -----------------------------
    # Pretrained Model Setting
    # -----------------------------
    # config.pretrain_attr_clf =
    # config.pretrain_obj_clf =


    # ----------------------------
    # Proj File System
    # ----------------------------
    config.root_dir = os.path.abspath('.')
    config.data_dir = '/tank/space/xugy07/tmp2/Data'

    # DataFile
    if config.dataset == 'mitstates':
        config.data_dir = os.path.join(config.data_dir, "mit-states")
    elif config.dataset == 'zappos':
        config.data_dir = os.path.join(config.data_dir, "ut-zap50k")

    # EmbFile
    config.attr_emb_file = os.path.join(config.data_dir, "attr_embedding_{}.pth".format(config.dataset))
    config.obj_emb_file = os.path.join(config.data_dir, "obj_embedding_{}.pth".format(config.dataset))

    # CacheDir For Saving ResNet/GloVe
    config.cache_dir = os.path.join(config.root_dir, ".cache")
    config.resnet_dir = os.path.join(config.cache_dir, ".resnet")

    # ResultDir
    config.result_dir = os.path.join(config.root_dir, 'results')
    # config.ckpt_dir = os.path.join(config.result_dir, 'ckpt')
    config.log_dir = os.path.join(config.result_dir, 'log')
    config.pred_dir = os.path.join(config.result_dir, 'pred')
    config.result_test_dir = os.path.join(config.result_dir, 'result_test')

    # check point dir
    config.ckpt_dir = os.path.join(config.root_dir, 'ckpt_dir')
    os.makedirs(config.ckpt_dir, exist_ok=True)


    return config
