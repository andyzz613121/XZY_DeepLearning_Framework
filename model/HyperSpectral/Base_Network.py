import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.Self_Module.Layer_operations import CVT3D_2D_SA

class SP_2D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP 2D Net
        print('SP_2D: Using SP_2D model in Base_Network')
        self.SP_conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
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

    def forward(self, x):
        x1 = self.SP_conv_1(x)
        x2 = self.SP_conv_2(x1)
        x3 = self.SP_conv_3(x2)
        x4 = self.SP_conv_4(x3)
        x5 = self.SP_conv_5(x4)
        return x5
        
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
        self.SA_CPR1 = CVT3D_2D_SA(16, 16, input_channels)
        self.SA_CPR2 = CVT3D_2D_SA(16, 16, input_channels)
        self.SA_CPR3 = CVT3D_2D_SA(16, 16, input_channels)
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

class SA_3D(nn.Module):
    def __init__(self, input_channels, out_channels):
        print('SA_3D: Using SA_3D model')
        super(SA_3D,self).__init__()
        #///////////////////////////////////////////////////////////////////////
        self.SP_conv_1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.SP_conv_2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.SP_conv_4 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.SP_conv_5 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.SA_CPR1 = CVT3D_2D_SA(16, 16, input_channels)
        self.SA_CPR2 = CVT3D_2D_SA(16, 16, input_channels)
        self.SA_CPR3 = CVT3D_2D_SA(16, 16, input_channels)
        self.SA_CPR4 = CVT3D_2D_SA(16, 16, input_channels)
        self.SA_CPR5 = CVT3D_2D_SA(16, 16, input_channels)
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
    def forward(self, x):
        b, c, h, w = x.size()
        
        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.l1_3D(x_3d)
        SA1_2D = self.SA_CPR1(SA1)
        SA1_sideout = sideout2d(SA1_2D, self.SA_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.l2_3D(SA1)
        SA2_2D = self.SA_CPR2(SA2)
        SA2_sideout = sideout2d(SA2_2D, self.SA_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.l3_3D(SA2)
        SA3_2D = self.SA_CPR3(SA3)
        SA3_sideout = sideout2d(SA3_2D, self.SA_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4 = self.l4_3D(SA3)
        SA4_2D = self.SA_CPR4(SA4)
        SA4_sideout = sideout2d(SA4_2D, self.SA_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5 = self.l5_3D(SA4)
        SA5_2D = self.SA_CPR5(SA5)
        SA5_sideout = sideout2d(SA5_2D, self.SA_Sideout5)

        Total_fuse = self.fuse_para[0]*SA1_sideout + self.fuse_para[1]*SA2_sideout + self.fuse_para[2]*SA3_sideout+\
                     self.fuse_para[3]*SA4_sideout + self.fuse_para[4]*SA5_sideout
                     
        return [Total_fuse, SA1_sideout, SA2_sideout, SA3_sideout, SA4_sideout, SA5_sideout]

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
        self.SP_conv_1 = DilaConv(input_channels, 16, 1)
        self.SP_conv_2 = DilaConv(16, 16, 1)
        self.SP_conv_3 = DilaConv(16, 16, 1)
        self.SP_conv_4 = DilaConv(16, 16, 1)
        self.SP_conv_5 = DilaConv(16, 16, 1)
        # self.SP_conv_1 = DilaConv(1, 16, 2)
        # self.SP_conv_2 = DilaConv(16, 16, 4)
        # self.SP_conv_3 = DilaConv(16, 16, 8)
        # self.SP_conv_4 = DilaConv(16, 16, 16)
        # self.SP_conv_5 = DilaConv(16, 16, 32)
        # self.SP_conv_2 = DilaConv(18, 16, 2)
        # self.SP_conv_3 = DilaConv(20, 16, 4)
        # self.SP_conv_4 = DilaConv(24, 16, 8)
        # self.SP_conv_5 = DilaConv(32, 16, 16)
        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout1 = get_mlp(2, [16, out_channels])
        self.SP_Sideout2 = get_mlp(2, [16, out_channels])
        self.SP_Sideout3 = get_mlp(2, [16, out_channels])
        self.SP_Sideout4 = get_mlp(2, [16, out_channels])
        self.SP_Sideout5 = get_mlp(2, [16, out_channels])

        
    def forward(self, x):
        x1 = self.SP_conv_1(x)
        x2 = self.SP_conv_2(x1)
        x3 = self.SP_conv_3(x2)
        x4 = self.SP_conv_4(x3)
        x5 = self.SP_conv_5(x4)
        return x5

class HS_Base(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(HS_Base,self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.spatial_fc = self.get_mlp(3, [128, 256, out_channels])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
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

    def forward(self, x):
        b, c, h, w = x.size()

        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)

        x3_GAP = torch.nn.AdaptiveAvgPool2d(1)(x3)
        
        x3_GAP = x3_GAP.view(b, -1)
        out = self.spatial_fc(x3_GAP)
        return out

class HS_Base3D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(HS_Base3D,self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.spatial_fc = get_mlp(3, [128, 256, out_channels])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        b, c, h, w = x.size()

        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)

        x3_GAP = torch.nn.AdaptiveAvgPool2d(1)(x3)
        
        x3_GAP = x3_GAP.view(b, -1)
        out = self.spatial_fc(x3_GAP)
        return out

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