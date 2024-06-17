import re
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from model.ResNet.ResNet import resnet101, resnet50, resnet34
from model.Self_Module.CARB import CARB_Block
class ResU_Net(nn.Module):
    def __init__(self, arch, input_channels, num_classes, pre_train=True):
        super(ResU_Net, self).__init__()
        
        if arch == 'ResNet34': 
            self.resnet = resnet34(input_channels, True)
            self.decoder4 = DecoderBlock(512, 256)
            self.decoder3 = DecoderBlock(256, 128)
            self.decoder2 = DecoderBlock(128, 64)
            self.decoder1 = DecoderBlock(64, 64)

            self.Sideout4 = CARB_Block(256, 6)
            self.Sideout3 = CARB_Block(128, 6)
            self.Sideout2 = CARB_Block(64, 6)
            self.Sideout1 = CARB_Block(64, 6)
        
        elif arch == 'ResNet50': 
            self.resnet = resnet50(input_channels, True)
            self.decoder4 = DecoderBlock(2048, 1024)
            self.decoder3 = DecoderBlock(1024, 512)
            self.decoder2 = DecoderBlock(512, 256)
            self.decoder1 = DecoderBlock(256, 64)

            self.Sideout4 = CARB_Block(1024, 6)
            self.Sideout3 = CARB_Block(512, 6)
            self.Sideout2 = CARB_Block(256, 6)
            self.Sideout1 = CARB_Block(64, 6)
            
        elif arch == 'ResNet101': 
            self.resnet = resnet101(input_channels, True)
            self.decoder4 = DecoderBlock(2048, 1024)
            self.decoder3 = DecoderBlock(1024, 512)
            self.decoder2 = DecoderBlock(512, 256)
            self.decoder1 = DecoderBlock(256, 64)

            self.Sideout4 = CARB_Block(1024, 6)
            self.Sideout3 = CARB_Block(512, 6)
            self.Sideout2 = CARB_Block(256, 6)
            self.Sideout1 = CARB_Block(64, 6)
            
            
        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4

        self.classifier = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, num_classes, 3, padding=1)
            )


    def forward(self, x):
        batch_size, channel_size, height, width = x.size()

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_unpool = x
        x = self.resnet.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + e3
        d4_map = self.Sideout4(d4)
        
        d3 = self.decoder3(d4) + e2
        d3_map = self.Sideout3(d3)
        
        d2 = self.decoder2(d3) + e1
        d2_map = self.Sideout2(d2)       

        d1 = self.decoder1(d2) + x_unpool
        d1_map = self.Sideout1(d1)
        
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


