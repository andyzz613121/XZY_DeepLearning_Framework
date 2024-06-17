import re
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from model.ResNet.ResU_Net import ResU_Net
from model.Self_Module.CARB import CARB_Block

class ResU_Net_Skeleton_softmax(ResU_Net):
    def __init__(self, arch, input_channels, num_classes, pre_train=True):
        super(ResU_Net_Skeleton_softmax, self).__init__(arch, input_channels, num_classes, pre_train)
        
        self.skeleton_classifier = nn.Sequential(
            nn.Linear(num_classes*num_classes, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64, bias=True),
        )
        self.skeleton_deconv_1 = nn.Sequential(
            nn.Linear(64, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64, bias=True),
        )
        self.skeleton_deconv_2 = nn.Sequential(
            nn.Linear(64, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=True),
        )
        self.skeleton_deconv_3 = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
        )
        self.skeleton_deconv_4 = nn.Sequential(
            nn.Linear(256, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
        )
        
        self.skeleton_conv_1 = nn.Sequential(
            nn.Linear(64, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64, bias=True),
        )
        self.skeleton_conv_2 = nn.Sequential(
            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64, bias=True),
        )
        self.skeleton_conv_3 = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=True),
        )
        self.skeleton_conv_4 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
        )
        self.skeleton_head = nn.Sequential(
            nn.Linear(64, input_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels, input_channels, bias=True),
        )

        self.CARB5 = CARB_Block(512, 6)
        self.CARB4 = CARB_Block(256, 6)
        self.CARB3 = CARB_Block(128, 6)
        self.CARB2 = CARB_Block(64, 6)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, accuracy_list):
        batch_size, channel_size, height, width = x.size()
        
        #skeleton
        weight_classifier = self.softmax(self.skeleton_classifier(accuracy_list))
        weight_classifier = weight_classifier*weight_classifier.shape[1]

        weight_dc1 = self.softmax(self.skeleton_deconv_1(weight_classifier))
        weight_dc1 = weight_dc1*weight_dc1.shape[1]
        weight_dc2 = self.softmax(self.skeleton_deconv_2(weight_dc1))
        weight_dc2 = weight_dc2*weight_dc2.shape[1]
        weight_dc3 = self.softmax(self.skeleton_deconv_3(weight_dc2))
        weight_dc3 = weight_dc3*weight_dc3.shape[1]
        weight_dc4 = self.softmax(self.skeleton_deconv_4(weight_dc3))
        weight_dc4 = weight_dc4*weight_dc4.shape[1]

        weight_c4 = self.softmax(self.skeleton_conv_4(weight_dc4))
        weight_c4 = weight_c4*weight_c4.shape[1]
        weight_c3 = self.softmax(self.skeleton_conv_3(weight_c4))
        weight_c3 = weight_c3*weight_c3.shape[1]
        weight_c2 = self.softmax(self.skeleton_conv_2(weight_c3))
        weight_c2 = weight_c2*weight_c2.shape[1]
        weight_c1 = self.softmax(self.skeleton_conv_1(weight_c2))
        weight_c1 = weight_c1*weight_c1.shape[1]

        weight_head = self.softmax(self.skeleton_head(weight_c1))
        weight_head = weight_head*weight_head.shape[1]

        #encoder
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(x, weight_head).permute(0, 3, 1, 2).contiguous()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_unpool = x
        x = self.resnet.maxpool(x)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(x, weight_c1).permute(0, 3, 1, 2).contiguous()
        e1 = self.encoder1(x)
        e1 = e1.permute(0, 2, 3, 1).contiguous()
        e1 = torch.mul(e1, weight_c2).permute(0, 3, 1, 2).contiguous()
        e2 = self.encoder2(e1)
        e2 = e2.permute(0, 2, 3, 1).contiguous()
        e2 = torch.mul(e2, weight_c3).permute(0, 3, 1, 2).contiguous()
        e3 = self.encoder3(e2)
        e3 = e3.permute(0, 2, 3, 1).contiguous()
        e3 = torch.mul(e3, weight_c4).permute(0, 3, 1, 2).contiguous()
        e4 = self.encoder4(e3)

        e4 = e4.permute(0, 2, 3, 1).contiguous()
        e4 = torch.mul(e4, weight_dc4).permute(0, 3, 1, 2).contiguous()
        d4 = self.decoder4(e4) + e3
        d4_map = self.Sideout4(d4)

        d4 = d4.permute(0, 2, 3, 1).contiguous()
        d4 = torch.mul(d4, weight_dc3).permute(0, 3, 1, 2).contiguous()
        d3 = self.decoder3(d4) + e2
        d3_map = self.Sideout3(d3)

        d3 = d3.permute(0, 2, 3, 1).contiguous()
        d3 = torch.mul(d3, weight_dc2).permute(0, 3, 1, 2).contiguous()
        d2 = self.decoder2(d3) + e1
        d2_map = self.Sideout2(d2)

        d2 = d2.permute(0, 2, 3, 1).contiguous()
        d2 = torch.mul(d2, weight_dc1).permute(0, 3, 1, 2).contiguous()
        d1 = self.decoder1(d2) + x_unpool
        d1_map = self.Sideout1(d1)

        d1 = d1.permute(0, 2, 3, 1).contiguous()
        d1 = torch.mul(d1, weight_classifier).permute(0, 3, 1, 2).contiguous()
        out = self.classifier(d1)

        
        d1_map = F.interpolate(d1_map, size=(height, width), mode='bilinear')
        d2_map = F.interpolate(d2_map, size=(height, width), mode='bilinear')
        d3_map = F.interpolate(d3_map, size=(height, width), mode='bilinear')
        d4_map = F.interpolate(d4_map, size=(height, width), mode='bilinear')
        return out, d1_map, d2_map, d3_map, d4_map
        

    
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