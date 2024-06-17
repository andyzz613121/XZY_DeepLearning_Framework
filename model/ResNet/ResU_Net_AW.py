import re
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from model.ResNet.ResU_Net import ResU_Net
from model.Self_Module.CARB import CARB_Block

class ResU_Net_AW(ResU_Net):
    def __init__(self, arch, input_channels, num_classes, pre_train=True):
        super(ResU_Net_AW, self).__init__(arch, input_channels, num_classes, pre_train)

        self.CARB0 = CARB_Block(6, 1)
        self.CARB1 = CARB_Block(6, 1)
        self.CARB2 = CARB_Block(6, 1)
        self.CARB3 = CARB_Block(6, 1)
        self.CARB4 = CARB_Block(6, 1)
        self.CARB5 = CARB_Block(6, 1)

        self.auto_weight0 = nn.Parameter(torch.ones([input_channels], requires_grad=True))
        self.auto_weight1 = nn.Parameter(torch.ones([input_channels], requires_grad=True))
        self.auto_weight2 = nn.Parameter(torch.ones([input_channels], requires_grad=True))
        self.auto_weight3 = nn.Parameter(torch.ones([input_channels], requires_grad=True))
        self.auto_weight4 = nn.Parameter(torch.ones([input_channels], requires_grad=True))
        self.auto_weight5 = nn.Parameter(torch.ones([input_channels], requires_grad=True))

        
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


def add_pre_model(model, premodel):
    model_dict = model.state_dict()
    premodel_dict = premodel.state_dict()
    
    model_dict_copy = model_dict.copy()
    for key, value in model_dict.items():
        for key_pre, value_pre in premodel_dict.items():
            if key == key_pre:
                model_dict_copy[key] = value_pre
                continue
    model.load_state_dict(model_dict_copy)
    print('set model with pretrained ResNet model')
    return model