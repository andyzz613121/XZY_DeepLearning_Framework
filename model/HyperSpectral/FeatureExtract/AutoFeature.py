import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HyperSpectral.Base_Network import get_mlp
from model.HyperSpectral.Basic_Operation import *
from model.Self_Module.Layer_operations import *
from model.ResNet.ResNet import resnet50

class AutoFeature(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.net = resnet50(1)

        ratio = 2
        ipt_c_re = input_channels
        self.mlp1_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        self.mlp1_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp2_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp2_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp3_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp3_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        self.mlp2_1 = get_mlp(2, [int((ipt_c_re)/2+0.99), int((ipt_c_re)/2+0.99)*ratio])
        self.mlp2_2 = get_mlp(2, [int((ipt_c_re)/2+0.99), int((ipt_c_re)/2+0.99)*ratio])
        self.mlp3_1 = get_mlp(2, [int((ipt_c_re)/4+0.99), int((ipt_c_re)/4+0.99)*ratio])
        self.mlp3_2 = get_mlp(2, [int((ipt_c_re)/4+0.99), int((ipt_c_re)/4+0.99)*ratio])
        self.mlp4_1 = get_mlp(2, [int((ipt_c_re)/6+0.99), int((ipt_c_re)/6+0.99)*ratio])
        self.mlp4_2 = get_mlp(2, [int((ipt_c_re)/6+0.99), int((ipt_c_re)/6+0.99)*ratio])
        self.mlp5_1 = get_mlp(2, [int((ipt_c_re)/8+0.99), int((ipt_c_re)/8+0.99)*ratio])
        self.mlp5_2 = get_mlp(2, [int((ipt_c_re)/8+0.99), int((ipt_c_re)/8+0.99)*ratio])
        self.mlp_list1 = [self.mlp1_1, self.mlp2_1, self.mlp3_1, self.mlp4_1, self.mlp5_1]
        self.mlp_list2 = [self.mlp1_2, self.mlp2_2, self.mlp3_2, self.mlp4_2, self.mlp5_2]

        self.GAP2D = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.size()
        MS_Spectral = MS_spectral(center_pixel(x), rsp_rate_list=[1])
        MS_Feat = MS_spectral_FeatureExtract(MS_Spectral, self.mlp_list1, self.mlp_list2, samesize=True)
        MS_Feat = torch.cat([x for x in MS_Feat], 1)

        codes = self.net(MS_Feat)
        return codes, MS_Feat
