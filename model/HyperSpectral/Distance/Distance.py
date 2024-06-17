from turtle import forward
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Self_Module.ASPP import ASPP
from model.HyperSpectral.Grid.SpectralImage_Grid import CVT3D_2D, get_mlp
from model.HyperSpectral.Grid.SpectralImage_GridaAtten import compute_ratio_withstep_entireimage

class SP_2D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP 2D Net
        print('SP_2D: Using SP_2D model')
        self.SP_conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_2 = nn.Sequential(
            nn.Conv2d(18, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Conv2d(20, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_4 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout1 = get_mlp(2, [16, out_channels])
        self.SP_Sideout2 = get_mlp(2, [16, out_channels])
        self.SP_Sideout3 = get_mlp(2, [16, out_channels])
        self.SP_Sideout4 = get_mlp(2, [16, out_channels])
        self.SP_Sideout5 = get_mlp(2, [16, out_channels])

class SA3D2D(nn.Module):
    def __init__(self, input_channels, out_channels):
        print('SA_3D: Using SA3D2D model')
        super(SA3D2D,self).__init__()
        #///////////////////////////////////////////////////////////////////////
        # SP & SA Model
        self.l1_3D = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(7,3,3), padding=(3,1,1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.l2_3D = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(7,3,3), padding=(3,1,1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.l3_3D = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(7,3,3), padding=(3,1,1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.l4_2D = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.l5_2D = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.SA_CPR1 = CVT3D_2D(16, 16, input_channels)
        self.SA_CPR2 = CVT3D_2D(16, 16, input_channels)
        self.SA_CPR3 = CVT3D_2D(16, 16, input_channels)
        #///////////////////////////////////////////////////////////////////////
        # fuse side mlps
        self.SA_Sideout1 = get_mlp(2, [16, out_channels])
        self.SA_Sideout2 = get_mlp(2, [16, out_channels])
        self.SA_Sideout3 = get_mlp(2, [16, out_channels])
        self.SA_Sideout4 = get_mlp(2, [16, out_channels])
        self.SA_Sideout5 = get_mlp(2, [16, out_channels])
        #///////////////////////////////////////////////////////////////////////
        # Fusion Paras
        self.fuse_para = nn.Parameter(torch.ones([15]))
        #///////////////////////////////////////////////////////////////////////
        # Other Operations
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.GAP2D = torch.nn.AdaptiveAvgPool2d(1)
        #///////////////////////////////////////////////////////////////////////
        # Init Para
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)

class DilaConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilaConv, self).__init__()
        self.DilaConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
    def forward(self, x):
        return self.DilaConv(x)

class SP_2DDilaConv(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(SP_2DDilaConv, self).__init__()
        # SP 2D Net
        print('SP_2D: Using SP_2DDilaConv model')
        self.SP_conv_1 = DilaConv(1, 16, 1)
        self.SP_conv_2 = DilaConv(18, 16, 2)
        self.SP_conv_3 = DilaConv(20, 16, 4)
        self.SP_conv_4 = DilaConv(24, 16, 8)
        self.SP_conv_5 = DilaConv(32, 16, 16)
        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout1 = get_mlp(2, [16, out_channels])
        self.SP_Sideout2 = get_mlp(2, [16, out_channels])
        self.SP_Sideout3 = get_mlp(2, [16, out_channels])
        self.SP_Sideout4 = get_mlp(2, [16, out_channels])
        self.SP_Sideout5 = get_mlp(2, [16, out_channels])

class Distance_3D2D(nn.Module):
    def __init__(self, input_channels, out_channels):
        print('Using Distance_3D2D model')
        super(Distance_3D2D, self).__init__(input_channels, out_channels)
        # SA Model
        self.SA_model = SA3D2D(input_channels, out_channels)
        #///////////////////////////////////////////////////////////////////////
        # Distance MLPS
        self.dis_mlp1 = get_mlp(3, [121, 64, 5])
        self.dis_mlp2 = get_mlp(3, [121, 64, 5])
        self.dis_mlp3 = get_mlp(3, [121, 64, 5])
        self.dis_mlp4 = get_mlp(3, [121, 64, 5])
        self.dis_mlp5 = get_mlp(3, [121, 64, 5])
        #///////////////////////////////////////////////////////////////////////
        # Fusion Paras
        self.fuse_para = nn.Parameter(torch.ones([15]))
        #///////////////////////////////////////////////////////////////////////
        # Other Operations
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.GAP2D = torch.nn.AdaptiveAvgPool2d(1)
        #///////////////////////////////////////////////////////////////////////
        # Init Para
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)
         #///////////////////////////////////////////////////////////////////////
    def compute_dismap(self):
        dis_para_square = self.dis_para * self.dis_para
        return 1

    def center_pixel(self, x):
        b, l, h, w = x.size()

        # 得到光谱图
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt = x[:,:,pth,ptw].view(b, -1)
        return pt

    def forward(self, x):
        b, c, h, w = x.size()

        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.SA_model.l1_3D(x_3d)
        SA1_2D = self.SA_model.SA_CPR1(SA1)
        print(SA1_2D.view(b, -1).shape)
        DIS_para1 = self.dis_mlp1(SA1_2D.view(b, -1))
        SA_side1 = self.sideout2d(SA1_2D, self.SA_model.SA_Sideout1)  # 是否加上SP1的sideout
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.l2_3D(SA1)
        SA2_2D = self.SA_model.SA_CPR2(SA2)
        SA_side2 = self.sideout2d(SA2_2D, self.SA_model.SA_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.l3_3D(SA2)
        SA3_2D = self.SA_model.SA_CPR3(SA3)
        SA_side3 = self.sideout2d(SA3_2D, self.SA_model.SA_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.SA_model.l4_2D(SA3_2D)
        SA_side4 = self.sideout2d(SA4_2D, self.SA_model.SA_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.SA_model.l5_2D(SA4_2D)
        SA_side5 = self.sideout2d(SA5_2D, self.SA_model.SA_Sideout5)

        Total_fuse = self.fuse_para[10]*SA_side5 + self.fuse_para[11]*SA_side4 + self.fuse_para[12]*SA_side3+\
                     self.fuse_para[13]*SA_side2 + self.fuse_para[14]*SA_side1
        
        return [Total_fuse, SA_side5, SA_side4, SA_side3, SA_side2, SA_side1]

        
def get_mlp(layer_num, node_list, drop_rate=0.2):
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


