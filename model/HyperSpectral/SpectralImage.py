import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from model.HyperSpectral.Base_Network import HS_Base

class HS_SI(HS_Base):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI model')
        super(HS_SI,self).__init__(input_channels, out_channels)
        self.SI_conv_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SI_conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SI_conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.SI_fc1 = self.get_mlp(3, [32, 64, out_channels])
        self.SI_fc2 = self.get_mlp(3, [64, 128, out_channels])
        self.SI_fc3 = self.get_mlp(3, [128, 256, out_channels])

        self.SA_fc1 = self.get_mlp(3, [32, 64, out_channels])
        self.SA_fc2 = self.get_mlp(3, [64, 128, out_channels])
        self.SA_fc3 = self.get_mlp(3, [128, 256, out_channels])
        self.SA_fcout = self.get_mlp(3, [128, 256, out_channels])

        self.weight_MLP1 = self.get_mlp(3, [out_channels*out_channels, 64, 32])
        self.weight_MLP2 = self.get_mlp(3, [out_channels*out_channels, 128, 64])
        self.weight_MLP3 = self.get_mlp(3, [out_channels*out_channels, 256, 128])

        self.fuse1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32))
        self.fuse2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64))
        self.fuse3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128))

        self.softmax = nn.Softmax(dim=-1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

    def get_mlp(self, layer_num, node_list, drop_rate=0.2):
        layers = []
        for layer in range(layer_num-1):
            layers.append(nn.Linear(node_list[layer], node_list[layer+1]))
            if layer+1 != (layer_num-1):  #Last layer
                layers.append(nn.Dropout(drop_rate))
                layers.append(nn.ReLU())
        mlp = nn.Sequential(*layers)
        for m in mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        return mlp

    def forward1(self, x):
        b, c, h, w = x.size()

        # x_select = torch.cat([x[:,0:1,:,:], x[:,51:52,:,:], x[:,81:82,:,:]], 1)
        # print(x_select.shape)
        
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)

        x3_GAP = torch.nn.AdaptiveAvgPool2d(1)(x3)
        x3_GAP = x3_GAP.view(b, -1)
        out1 = self.SA_fc3(x3_GAP)

        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)

        pt = x[:,:,pth,ptw].view(b, 1, 1, -1)
        pt = F.interpolate(pt, size=(1, 144), mode='bilinear')
        pt = pt.view(b, 1, 1, -1)
        
        p1 = self.SI_conv_1(pt)
        p2 = self.SI_conv_2(p1)
        p3 = self.SI_conv_3(p2)

        p3_GAP = torch.nn.AdaptiveAvgPool2d(1)(p3)
        p3_GAP = p3_GAP.view(b, -1)
        out2 = self.SI_fc3(p3_GAP)

        return out1 + out2
        # return out2

    def forward(self, x):
        b, c, h, w = x.size()

        # x_select = torch.cat([x[:,0:1,:,:], x[:,51:52,:,:], x[:,81:82,:,:]])

        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt = x[:,:,pth,ptw].view(b, 1, 12, 12)

        #--------------------Layer 1---------------------
        #SA
        x1 = self.conv_1(x)
        x1_GAP = torch.nn.AdaptiveAvgPool2d(1)(x1).view(b, -1)
        side_SA1 = self.SA_fc1(x1_GAP)
        #SI
        pt1 = self.SI_conv_1(pt)
        pt1_GAP = torch.nn.AdaptiveAvgPool2d(1)(pt1).view(b, -1)
        side_SI1 = self.SI_fc1(pt1_GAP)

        # diff1 = torch.bmm(self.softmax(side_SA1).view(b, -1, 1), self.softmax(side_SI1).view(b, -1, 1).permute(0, 2, 1)).view(b, -1)
        # weight1 = self.weight_MLP1(diff1).view(b, -1, 1, 1).repeat(1, 1, pt1.shape[2], pt1.shape[3])
        # pt1 = pt1 + weight1*pt1
        x1 = self.fuse1(torch.cat([x1, pt1], 1))
        #--------------------Layer 2---------------------
        #SA
        x2 = self.conv_2(x1)
        x2_GAP = torch.nn.AdaptiveAvgPool2d(1)(x2).view(b, -1)
        side_SA2 = self.SA_fc2(x2_GAP)
        #SI
        pt2 = self.SI_conv_2(pt1)
        pt2_GAP = torch.nn.AdaptiveAvgPool2d(1)(pt2).view(b, -1)
        side_SI2 = self.SI_fc2(pt2_GAP)

        # # diff2 = self.softmax(side_SA2 - side_SI2)
        # diff2 = torch.bmm(self.softmax(side_SA2).view(b, -1, 1), self.softmax(side_SI2).view(b, -1, 1).permute(0, 2, 1)).view(b, -1)
        # weight2 = self.weight_MLP2(diff2).view(b, -1, 1, 1).repeat(1, 1, pt2.shape[2], pt2.shape[3])
        # pt2 = pt2 + weight2*pt2
        x2 = self.fuse2(torch.cat([x2, pt2], 1))
        #--------------------Layer 3---------------------
        #SA
        x3 = self.conv_3(x2)
        x3_GAP = torch.nn.AdaptiveAvgPool2d(1)(x3).view(b, -1)
        side_SA3 = self.SA_fc3(x3_GAP)
        #SI
        pt3 = self.SI_conv_3(pt2)
        pt3_GAP = torch.nn.AdaptiveAvgPool2d(1)(pt3).view(b, -1)
        side_SI3 = self.SI_fc3(pt3_GAP)
        
        # # diff3 = self.softmax(side_SA3 - side_SI3)
        # diff3 = torch.bmm(self.softmax(side_SA3).view(b, -1, 1), self.softmax(side_SI3).view(b, -1, 1).permute(0, 2, 1)).view(b, -1)
        # weight3 = self.weight_MLP3(diff3).view(b, -1, 1, 1).repeat(1, 1, pt3.shape[2], pt3.shape[3])
        # pt3 = pt3 + weight3*pt3
        x3 = self.fuse3(torch.cat([x3, pt3], 1))
        #--------------------Layer 3---------------------
        x3_GAP = torch.nn.AdaptiveAvgPool2d(1)(x3).view(b, -1)
        x3_GAP = x3_GAP.view(b, -1)
        out = self.fc(x3_GAP)
        
        return out, side_SA1, side_SI1, side_SA2, side_SI2, side_SA3, side_SI3

