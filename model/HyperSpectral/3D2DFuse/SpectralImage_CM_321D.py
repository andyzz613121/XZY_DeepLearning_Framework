from ast import If
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from model.HyperSpectral.SpectralImage_CM_3D_SAM import SP_1D, SP_2D, SA_3D

class HS_321D(nn.Module):
    def __init__(self, input_channels, out_channels):
        print('Using HS_321D model')
        super(HS_321D,self).__init__()
        #///////////////////////////////////////////////////////////////////////
        # SP & SA Model
        self.model_1D = SP_1D(input_channels, out_channels)
        self.model_2D = SP_2D(input_channels, out_channels)
        self.model_3D = SA_3D(input_channels, out_channels)
        
        #///////////////////////////////////////////////////////////////////////
        # fuse side mlps
        self.Fuse_fc1 = get_mlp(3, [3*out_channels, 256, out_channels])
        self.Fuse_fc2 = get_mlp(3, [3*out_channels, 256, out_channels])
        self.Fuse_fc3 = get_mlp(3, [3*out_channels, 256, out_channels])
        
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
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)


    def sideout2d(self, feats, mlp):
        b, c, h, w = feats.size()
        return mlp(self.GAP2D(feats).view(b, -1))
    
    def sideout1d(self, feats, mlp):
        b, c, l = feats.size()
        feats = feats.view(b, c, l, 1)
        return mlp(self.GAP2D(feats).view(b, -1))

    def sideout3d(self, feats, mlp):
        b, c, l, h, w = feats.size()
        feats = feats.view(b, c*l, h, w)
        return mlp(self.GAP2D(feats).view(b, -1))

    def CVT3d_2dCompress(self, feats, compress_conv):
        feats = torch.reshape(feats, (feats.shape[0], -1, feats.shape[3], feats.shape[4]))
        return compress_conv(feats)
    
    def forward(self, x):
        b, c, h, w = x.size()

        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        x_3d = torch.unsqueeze(x, 1)
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        x_1d = x[:,:,pth,ptw].unsqueeze(1)

        l1_3d = self.model_3D.SA_conv_1(x_3d)
        l1_2d = self.model_2D.SP_conv_1(x)
        l1_1d = self.model_1D.SP_conv_1(x_1d)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        l2_3d = self.model_3D.SA_conv_2(l1_3d)
        l2_2d = self.model_2D.SP_conv_2(l1_2d)
        l2_1d = self.model_1D.SP_conv_2(l1_1d)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        l3_3d = self.model_3D.SA_conv_3(l2_3d)
        l3_2d = self.model_2D.SP_conv_3(l2_2d)
        l3_1d = self.model_1D.SP_conv_3(l2_1d)
        #///////////////////////////////////////////////////////////////////////
        # Out
        # out_3D = self.sideout3d(l3_3d, self.model_3D.SA_Sideout3)
        
        out_3D = self.sideout2d(self.CVT3d_2dCompress(l3_3d, self.model_3D.SA_CPR3), self.model_3D.SA_Sideout3)
        out_2D = self.sideout2d(l3_2d, self.model_2D.SP_Sideout3)
        out_1D = self.sideout1d(l3_1d, self.model_1D.SP_Sideout3)
        out_fuse = self.Fuse_fc3(torch.cat([out_3D, out_2D, out_1D], 1))
        return [out_3D]

class SP_2D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP 2D Net
        self.SP_conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout1 = get_mlp(3, [16, 256, out_channels])
        self.SP_Sideout2 = get_mlp(3, [32, 256, out_channels])
        self.SP_Sideout3 = get_mlp(3, [64, 256, out_channels])

class SP_1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # SP 1D Net
        self.SP_conv_1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.SP_conv_2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout1 = get_mlp(3, [16, 256, out_channels])
        self.SP_Sideout2 = get_mlp(3, [32, 256, out_channels])
        self.SP_Sideout3 = get_mlp(3, [64, 256, out_channels])

class SP_MLP(nn.Module):
    def __init__(self, class_num, out_channels):
        super().__init__()
        # SP 1D Net
        self.SP_conv_1 = nn.Sequential(
            nn.Linear(class_num, 16),
            nn.ReLU(inplace=True),
        )
        self.SP_conv_2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Linear(32, 64),
        )
        self.SP_conv_4 = nn.Sequential(
            nn.Linear(64, out_channels),
        )

        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout1 = get_mlp(3, [16, 256, out_channels])
        self.SP_Sideout2 = get_mlp(3, [32, 256, out_channels])
        self.SP_Sideout3 = get_mlp(3, [64, 256, out_channels])

class SA_3D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(SA_3D,self).__init__()
        self.SA_conv_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.SA_conv_2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.SA_conv_3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.SA_conv_4 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.SA_conv_5 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.spatial_fc = get_mlp(3, [128, 256, out_channels])
        
        #///////////////////////////////////////////////////////////////////////
        # SA sideout mlps & compress conv空间
        self.SA_Sideout1 = get_mlp(3, [16, 256, out_channels])
        self.SA_Sideout2 = get_mlp(3, [32, 256, out_channels])
        self.SA_Sideout3 = get_mlp(3, [64, 256, out_channels])
        self.SA_Sideout5 = get_mlp(3, [128, 256, out_channels])

        # 3D压缩为2D
        # Houston18
        self.SA_CPR1 = nn.Sequential(
            nn.Conv2d(16*42, 16, kernel_size=1),
            nn.BatchNorm2d(16))
        self.SA_CPR2 = nn.Sequential(
            nn.Conv2d(32*36, 32, kernel_size=1),
            nn.BatchNorm2d(32))
        self.SA_CPR3 = nn.Sequential(
            nn.Conv2d(64*30, 64, kernel_size=1),
            nn.BatchNorm2d(64))
        self.SA_CPR5 = nn.Sequential(
            nn.Conv2d(128*18, 128, kernel_size=1),
            nn.BatchNorm2d(128))
        # # Houston13
        # self.SA_CPR1 = nn.Sequential(
        #     nn.Conv2d(16*138, 16, kernel_size=1),
        #     nn.BatchNorm2d(16))
        # self.SA_CPR2 = nn.Sequential(
        #     nn.Conv2d(32*132, 32, kernel_size=1),
        #     nn.BatchNorm2d(32))
        # self.SA_CPR3 = nn.Sequential(
        #     nn.Conv2d(64*126, 64, kernel_size=1),
        #     nn.BatchNorm2d(64))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        b, c, h, w = x.size()

        x1 = self.SA_conv_1(x)
        x2 = self.SA_conv_2(x1)
        x3 = self.SA_conv_3(x2)

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