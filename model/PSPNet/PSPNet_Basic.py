from model.PSPNet.network.pspnet import pspnet
from model.Self_Module.CARB import CARB_Block

import torch
from torch import nn

class PSPNet(pspnet):
    def __init__(self, input_channel, n_classes, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101', pretrained=True):
        super().__init__(input_channel, n_classes, sizes, psp_size, deep_features_size, backend, pretrained)


class PSPNet_AW(pspnet):
    def __init__(self, input_channel, n_classes, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101', pretrained=True):
        super().__init__(input_channel, n_classes, sizes, psp_size, deep_features_size, backend, pretrained)

        self.CARB0 = CARB_Block(6, 1)
        self.CARB1 = CARB_Block(6, 1)
        self.CARB2 = CARB_Block(6, 1)
        self.CARB3 = CARB_Block(6, 1)
        self.CARB4 = CARB_Block(6, 1)
        self.CARB5 = CARB_Block(6, 1)

        self.auto_weight0 = nn.Parameter(torch.ones([6], requires_grad=True))
        self.auto_weight1 = nn.Parameter(torch.ones([6], requires_grad=True))
        self.auto_weight2 = nn.Parameter(torch.ones([6], requires_grad=True))
        self.auto_weight3 = nn.Parameter(torch.ones([6], requires_grad=True))
        self.auto_weight4 = nn.Parameter(torch.ones([6], requires_grad=True))
        self.auto_weight5 = nn.Parameter(torch.ones([6], requires_grad=True))

if __name__ == "__main__":
    PSPNet(3,3)