"""
not sure whether we need this classifier;
1. vis;
2.
img_encoder = nn.Sequential(
    fc;
    batch;
    relu;
)

another way to employ the CNN Models;
"""
import os
import torch
import torch.nn as nn
import torchvision.models as tmodels


class ImgEncoder(nn.Module):
    def __init__(self, config):
        super(ImgEncoder, self).__init__()

        # download model
        os.environ['TORCH_HOME'] = config.cache_dir
        # --------------------------------------
        # we must set the model to eval mode
        # --------------------------------------
        ref_model = tmodels.resnet18(pretrained=True).eval()

        # freeze model: not learning
        for param in ref_model.parameters():
            param.requires_grad = False

        # define own model
        self.define_module(ref_model)


    def define_module(self, ref_model):
        # init processing;
        self.conv1 = ref_model.conv1
        self.bn1 = ref_model.bn1
        self.relu = ref_model.relu
        self.maxpool = ref_model.maxpool

        # basic block
        self.layer1 = ref_model.layer1
        self.layer2 = ref_model.layer2
        self.layer3 = ref_model.layer3
        self.layer4 = ref_model.layer4

        # cls layers;
        self.avgpool = ref_model.avgpool


    def forward(self, x):
        # pre-processing layers
        x = self.conv1(x)   # 64 * 112 * 112
        x = self.bn1(x)     # 64 * 112 * 112
        x = self.relu(x)    # 64 * 112 * 112
        x = self.maxpool(x)   # 64 * 56 * 56

        # basic block
        x_size_56 = self.layer1(x)              # 64  *  56 * 56
        x_size_28 = self.layer2(x_size_56)      # 128 *  28 * 28
        x_size_14 = self.layer3(x_size_28)      # 256 *  14 * 14
        x_size_7 = self.layer4(x_size_14)       # 512 *   7 *  7

        # for 1D feats and also for verification;
        x_vec = self.avgpool(x_size_7)
        x_vec = torch.flatten(x_vec, 1)

        return x_vec, x_size_7


if __name__ == "__main__":

    # load lib
    from Config.get_config import get_config

    config = get_config()
    test_model = ImgEncoder(config)
    dummy_img = torch.ones(1, 3, 224, 224)

    # img encoder from tmodels
    img_encoder = tmodels.resnet18(pretrained=True)
    img_encoder.fc = nn.Sequential()
    img_encoder.eval()


    #
    self_img_encoder = ImgEncoder(config).eval()


    out_1 = img_encoder(dummy_img)
    out_2 = self_img_encoder(dummy_img)
    print()
