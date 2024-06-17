from ast import If
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from model.HyperSpectral.Base_Network import HS_Base, HS_Base3D
from model.Self_Module.Attention import SPA_Att
class HS_3_2D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        
        n1 = input_channels+1
        n2 = input_channels+3

        self.L_3D_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(7, 1, 1), padding=(3,0,0)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )
        self.L_2D_2 = nn.Sequential(
            nn.Conv2d(in_channels=16*n1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.L_3D_3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(7, 1, 1), padding=(3,0,0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.L_2D_4 = nn.Sequential(
            nn.Conv2d(in_channels=32*n2, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.L_3D_5 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(7, 1, 1), padding=(3,0,0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

class HS_2_3D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()

        n1 = input_channels+2
        n2 = input_channels+4

        self.L_2D_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.L_3D_2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(7, 1, 1), padding=(3,0,0)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )
        self.L_2D_3 = nn.Sequential(
            nn.Conv2d(in_channels=16*n1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.L_3D_4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(7, 1, 1), padding=(3,0,0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.L_2D_5 = nn.Sequential(
            nn.Conv2d(in_channels=32*n2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

class HS_2D(nn.Module):
    def __init__(self):
        super().__init__()
        print('Using HS_2D model')
        self.L_2D_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.L_2D_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.L_2D_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.L_2D_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.L_2D_5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

class HS_1D(nn.Module):
    def __init__(self):
        super().__init__()
        print('Using HS_1D model')
        self.L_1D_1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.L_1D_2 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.L_1D_3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.L_1D_4 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.L_1D_5 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

class HS_MLP(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        print('Using HS_MLP model')
        self.L_mlp_1 = nn.Sequential(
            nn.Linear(input_channel, 16),
            nn.ReLU(inplace=True),
        )
        self.L_mlp_2 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
        )
        self.L_mlp_3 = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
        )
        self.L_mlp_4 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.L_mlp_5 = nn.Sequential(
            nn.Linear(32, 64)
        )

class Fuse23D_Net(nn.Module):
    def __init__(self, input_channels, out_channels):
        print('Using Fuse23D_Net model')
        super(Fuse23D_Net,self).__init__()
        #///////////////////////////////////////////////////////////////////////
        # 23D & 32D Model
        self.model_23D = HS_2_3D(input_channels, out_channels)
        self.model_32D = HS_3_2D(input_channels, out_channels)
        # self.model_SI = HS_2D(1, out_channels)
        self.model_SI = HS_2D()
        #///////////////////////////////////////////////////////////////////////
        # FC
        # Houston 13
        self.sideout_SA1 = get_mlp(3, [16*(input_channels+1), 256, out_channels])
        self.sideout_SA2 = get_mlp(3, [16*(input_channels+2), 256, out_channels])
        self.sideout_SA3 = get_mlp(3, [32*(input_channels+3), 256, out_channels])
        self.sideout_SA4 = get_mlp(3, [32*(input_channels+4), 256, out_channels])
        self.sideout_SA5 = get_mlp(3, [64*(input_channels+5), 256, out_channels])

        # Houston 18
        # self.sideout_SA1 = get_mlp(3, [16*49, 256, out_channels])
        # self.sideout_SA2 = get_mlp(3, [16*50, 256, out_channels])
        # self.sideout_SA3 = get_mlp(3, [32*51, 256, out_channels])
        # self.sideout_SA4 = get_mlp(3, [32*52, 256, out_channels])
        # self.sideout_SA5 = get_mlp(3, [3392, 256, out_channels])
        
        #Pavia
        # self.sideout_SA1 = get_mlp(3, [16*104, 256, out_channels])
        # self.sideout_SA2 = get_mlp(3, [16*105, 256, out_channels])
        # self.sideout_SA3 = get_mlp(3, [32*106, 256, out_channels])
        # self.sideout_SA4 = get_mlp(3, [32*107, 256, out_channels])
        # self.sideout_SA5 = get_mlp(3, [64*108, 256, out_channels])
        
        self.sideout_SI1 = get_mlp(3, [16, 256, out_channels])
        self.sideout_SI2 = get_mlp(3, [16, 256, out_channels])
        self.sideout_SI3 = get_mlp(3, [32, 256, out_channels])
        self.sideout_SI4 = get_mlp(3, [32, 256, out_channels])
        self.sideout_SI5 = get_mlp(3, [64, 256, out_channels])
        self.fuse_para = nn.Parameter(torch.ones([10]))
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

    def pixel2image(self, x):
        b, c, h, w = x.size()

        SI = x.transpose(2, 1).transpose(2, 3).view(b, h*w, 1, c)  
        SI = F.interpolate(SI, size=(1, 121), mode='bilinear', align_corners=False)
        SI = SI.view(b, h*w, 11, 11)
        return SI

    # forward Base 
    def forward_Base(self, x):
        b, c, h, w = x.size()
        
        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        l1_23d = self.model_23D.L_2D_1(x)
        l1_32d = self.model_32D.L_3D_1(x_3d)

        l1_2d_cat = torch.cat([l1_23d, l1_32d.view(b, -1, h, w)], 1)
        l1_3d_cat = torch.cat([l1_32d, l1_23d.view(b, -1, 1, h, w)], 2)
        # print(l1_2d_cat.shape, l1_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        l2_23d = self.model_23D.L_3D_2(l1_3d_cat)
        l2_32d = self.model_32D.L_2D_2(l1_2d_cat)

        l2_2d_cat = torch.cat([l2_32d, l2_23d.view(b, -1, h, w)], 1)
        l2_3d_cat = torch.cat([l2_23d, l2_32d.view(b, -1, 1, h, w)], 2)
        # print(l2_2d_cat.shape, l2_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        l3_23d = self.model_23D.L_2D_3(l2_2d_cat)
        l3_32d = self.model_32D.L_3D_3(l2_3d_cat)

        l3_2d_cat = torch.cat([l3_23d, l3_32d.view(b, -1, h, w)], 1)
        l3_3d_cat = torch.cat([l3_32d, l3_23d.view(b, -1, 1, h, w)], 2)
        # print(l3_2d_cat.shape, l3_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        l4_23d = self.model_23D.L_3D_4(l3_3d_cat)
        l4_32d = self.model_32D.L_2D_4(l3_2d_cat)

        l4_2d_cat = torch.cat([l4_32d, l4_23d.view(b, -1, h, w)], 1)
        l4_3d_cat = torch.cat([l4_23d, l4_32d.view(b, -1, 1, h, w)], 2)
        # print(l4_2d_cat.shape, l4_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5       
        l5_23d = self.model_23D.L_2D_5(l4_2d_cat)
        l5_32d = self.model_32D.L_3D_5(l4_3d_cat)

        #///////////////////////////////////////////////////////////////////////
        # Out
        l5_cat = torch.cat([l5_23d, l5_32d.view(b, -1, h, w)], 1)
        out_SA5 = self.sideout_SA5(self.GAP2D(l5_cat).view(b, -1))
        out_SA4 = self.sideout_SA4(self.GAP2D(l4_2d_cat).view(b, -1))
        out_SA3 = self.sideout_SA3(self.GAP2D(l3_2d_cat).view(b, -1))
        out_SA2 = self.sideout_SA2(self.GAP2D(l2_2d_cat).view(b, -1))
        out_SA1 = self.sideout_SA1(self.GAP2D(l1_2d_cat).view(b, -1))

        
        
        out_fusetotal = self.fuse_para[0]*out_SA5 + self.fuse_para[1]*out_SA4 + \
                        self.fuse_para[2]*out_SA3 + self.fuse_para[3]*out_SA2 + \
                        self.fuse_para[4]*out_SA1
        return [out_fusetotal, out_SA5, out_SA4, out_SA3, out_SA2, out_SA1]

    # forward 2D  
    def forward(self, x):
        b, c, h, w = x.size()
        
        x_3d = torch.unsqueeze(x, 1)
        SI = self.pixel2image(x)
        px_2d = SI[:,60,:,:].unsqueeze(1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        l1_23d = self.model_23D.L_2D_1(x)
        l1_32d = self.model_32D.L_3D_1(x_3d)
        l1_SI = self.model_SI.L_2D_1(px_2d)
        # print(l1_23d.shape, l1_32d.shape)
        l1_2d_cat = torch.cat([l1_23d, l1_32d.view(b, -1, h, w)], 1)
        l1_3d_cat = torch.cat([l1_32d, l1_23d.view(b, -1, 1, h, w)], 2)
        # print(l1_2d_cat.shape, l1_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        l2_23d = self.model_23D.L_3D_2(l1_3d_cat)
        l2_32d = self.model_32D.L_2D_2(l1_2d_cat)
        l2_SI = self.model_SI.L_2D_2(l1_SI)
        # print(l2_23d.shape, l2_32d.shape)
        l2_2d_cat = torch.cat([l2_32d, l2_23d.view(b, -1, h, w)], 1)
        l2_3d_cat = torch.cat([l2_23d, l2_32d.view(b, -1, 1, h, w)], 2)
        # print(l2_2d_cat.shape, l2_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        l3_23d = self.model_23D.L_2D_3(l2_2d_cat)
        l3_32d = self.model_32D.L_3D_3(l2_3d_cat)
        l3_SI = self.model_SI.L_2D_3(l2_SI)
        # print(l3_23d.shape, l3_32d.shape)
        l3_2d_cat = torch.cat([l3_23d, l3_32d.view(b, -1, h, w)], 1)
        l3_3d_cat = torch.cat([l3_32d, l3_23d.view(b, -1, 1, h, w)], 2)
        # print(l3_2d_cat.shape, l3_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        l4_23d = self.model_23D.L_3D_4(l3_3d_cat)
        l4_32d = self.model_32D.L_2D_4(l3_2d_cat)
        l4_SI = self.model_SI.L_2D_4(l3_SI)
        # print(l4_23d.shape, l4_32d.shape)
        l4_2d_cat = torch.cat([l4_32d, l4_23d.view(b, -1, h, w)], 1)
        l4_3d_cat = torch.cat([l4_23d, l4_32d.view(b, -1, 1, h, w)], 2)
        # print(l4_2d_cat.shape, l4_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5       
        l5_23d = self.model_23D.L_2D_5(l4_2d_cat)
        l5_32d = self.model_32D.L_3D_5(l4_3d_cat)
        l5_SI = self.model_SI.L_2D_5(l4_SI)
        #///////////////////////////////////////////////////////////////////////
        # Out
        l5_cat = torch.cat([l5_23d, l5_32d.view(b, -1, h, w)], 1)
        out_SA5 = self.sideout_SA5(self.GAP2D(l5_cat).view(b, -1))
        out_SA4 = self.sideout_SA4(self.GAP2D(l4_2d_cat).view(b, -1))
        out_SA3 = self.sideout_SA3(self.GAP2D(l3_2d_cat).view(b, -1))
        out_SA2 = self.sideout_SA2(self.GAP2D(l2_2d_cat).view(b, -1))
        out_SA1 = self.sideout_SA1(self.GAP2D(l1_2d_cat).view(b, -1))

        out_SI5 = self.sideout_SI5(self.GAP2D(l5_SI).view(b, -1))
        out_SI4 = self.sideout_SI4(self.GAP2D(l4_SI).view(b, -1))
        out_SI3 = self.sideout_SI3(self.GAP2D(l3_SI).view(b, -1))
        out_SI2 = self.sideout_SI2(self.GAP2D(l2_SI).view(b, -1))
        out_SI1 = self.sideout_SI1(self.GAP2D(l1_SI).view(b, -1))
        
        out_fusetotal = self.fuse_para[0]*out_SA5 + self.fuse_para[1]*out_SA4 + \
                        self.fuse_para[2]*out_SA3 + self.fuse_para[3]*out_SA2 + \
                        self.fuse_para[4]*out_SA1 + self.fuse_para[5]*out_SI5 + \
                        self.fuse_para[6]*out_SI4 + self.fuse_para[7]*out_SI3 + \
                        self.fuse_para[8]*out_SI2 + self.fuse_para[9]*out_SI1
        return [out_fusetotal, out_SA5, out_SA4, out_SA3, out_SA2, out_SA1, out_SI5, out_SI4, out_SI3, out_SI2, out_SI1]

    # forward 1D
    def forward_1D(self, x):
        b, c, h, w = x.size()
        
        x_3d = torch.unsqueeze(x, 1)
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt_center = x[:,:,pth,ptw].unsqueeze(1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        l1_23d = self.model_23D.L_2D_1(x)
        l1_32d = self.model_32D.L_3D_1(x_3d)
        l1_SI = self.model_SI.L_1D_1(pt_center)

        l1_2d_cat = torch.cat([l1_23d, l1_32d.view(b, -1, h, w)], 1)
        l1_3d_cat = torch.cat([l1_32d, l1_23d.view(b, -1, 1, h, w)], 2)
        # print(l1_2d_cat.shape, l1_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        l2_23d = self.model_23D.L_3D_2(l1_3d_cat)
        l2_32d = self.model_32D.L_2D_2(l1_2d_cat)
        l2_SI = self.model_SI.L_1D_2(l1_SI)

        l2_2d_cat = torch.cat([l2_32d, l2_23d.view(b, -1, h, w)], 1)
        l2_3d_cat = torch.cat([l2_23d, l2_32d.view(b, -1, 1, h, w)], 2)
        # print(l2_2d_cat.shape, l2_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        l3_23d = self.model_23D.L_2D_3(l2_2d_cat)
        l3_32d = self.model_32D.L_3D_3(l2_3d_cat)
        l3_SI = self.model_SI.L_1D_3(l2_SI)

        l3_2d_cat = torch.cat([l3_23d, l3_32d.view(b, -1, h, w)], 1)
        l3_3d_cat = torch.cat([l3_32d, l3_23d.view(b, -1, 1, h, w)], 2)
        # print(l3_2d_cat.shape, l3_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        l4_23d = self.model_23D.L_3D_4(l3_3d_cat)
        l4_32d = self.model_32D.L_2D_4(l3_2d_cat)
        l4_SI = self.model_SI.L_1D_4(l3_SI)

        l4_2d_cat = torch.cat([l4_32d, l4_23d.view(b, -1, h, w)], 1)
        l4_3d_cat = torch.cat([l4_23d, l4_32d.view(b, -1, 1, h, w)], 2)
        # print(l4_2d_cat.shape, l4_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5       
        l5_23d = self.model_23D.L_2D_5(l4_2d_cat)
        l5_32d = self.model_32D.L_3D_5(l4_3d_cat)
        l5_SI = self.model_SI.L_1D_5(l4_SI)
        #///////////////////////////////////////////////////////////////////////
        # Out
        l5_cat = torch.cat([l5_23d, l5_32d.view(b, -1, h, w)], 1)
        out_SA5 = self.sideout_SA5(self.GAP2D(l5_cat).view(b, -1))
        out_SA4 = self.sideout_SA4(self.GAP2D(l4_2d_cat).view(b, -1))
        out_SA3 = self.sideout_SA3(self.GAP2D(l3_2d_cat).view(b, -1))
        out_SA2 = self.sideout_SA2(self.GAP2D(l2_2d_cat).view(b, -1))
        out_SA1 = self.sideout_SA1(self.GAP2D(l1_2d_cat).view(b, -1))

        out_SI5 = self.sideout_SI5(self.GAP2D(l5_SI.unsqueeze(3)).view(b, -1))
        out_SI4 = self.sideout_SI4(self.GAP2D(l4_SI.unsqueeze(3)).view(b, -1))
        out_SI3 = self.sideout_SI3(self.GAP2D(l3_SI.unsqueeze(3)).view(b, -1))
        out_SI2 = self.sideout_SI2(self.GAP2D(l2_SI.unsqueeze(3)).view(b, -1))
        out_SI1 = self.sideout_SI1(self.GAP2D(l1_SI.unsqueeze(3)).view(b, -1))
        
        out_fusetotal = self.fuse_para[0]*out_SA5 + self.fuse_para[1]*out_SA4 + \
                        self.fuse_para[2]*out_SA3 + self.fuse_para[3]*out_SA2 + \
                        self.fuse_para[4]*out_SA1 + self.fuse_para[5]*out_SI5 + \
                        self.fuse_para[6]*out_SI4 + self.fuse_para[7]*out_SI3 + \
                        self.fuse_para[8]*out_SI2 + self.fuse_para[9]*out_SI1
        return [out_fusetotal, out_SA5, out_SA4, out_SA3, out_SA2, out_SA1, out_SI5, out_SI4, out_SI3, out_SI2, out_SI1]

    # forward mlp
    def forward_mlp(self, x):
        b, c, h, w = x.size()
        
        x_3d = torch.unsqueeze(x, 1)
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt_center = x[:,:,pth,ptw].unsqueeze(1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        l1_23d = self.model_23D.L_2D_1(x)
        l1_32d = self.model_32D.L_3D_1(x_3d)
        l1_SI = self.model_SI.L_mlp_1(pt_center)

        l1_2d_cat = torch.cat([l1_23d, l1_32d.view(b, -1, h, w)], 1)
        l1_3d_cat = torch.cat([l1_32d, l1_23d.view(b, -1, 1, h, w)], 2)
        # print(l1_2d_cat.shape, l1_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        l2_23d = self.model_23D.L_3D_2(l1_3d_cat)
        l2_32d = self.model_32D.L_2D_2(l1_2d_cat)
        l2_SI = self.model_SI.L_mlp_2(l1_SI)

        l2_2d_cat = torch.cat([l2_32d, l2_23d.view(b, -1, h, w)], 1)
        l2_3d_cat = torch.cat([l2_23d, l2_32d.view(b, -1, 1, h, w)], 2)
        # print(l2_2d_cat.shape, l2_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        l3_23d = self.model_23D.L_2D_3(l2_2d_cat)
        l3_32d = self.model_32D.L_3D_3(l2_3d_cat)
        l3_SI = self.model_SI.L_mlp_3(l2_SI)

        l3_2d_cat = torch.cat([l3_23d, l3_32d.view(b, -1, h, w)], 1)
        l3_3d_cat = torch.cat([l3_32d, l3_23d.view(b, -1, 1, h, w)], 2)
        # print(l3_2d_cat.shape, l3_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        l4_23d = self.model_23D.L_3D_4(l3_3d_cat)
        l4_32d = self.model_32D.L_2D_4(l3_2d_cat)
        l4_SI = self.model_SI.L_mlp_4(l3_SI)

        l4_2d_cat = torch.cat([l4_32d, l4_23d.view(b, -1, h, w)], 1)
        l4_3d_cat = torch.cat([l4_23d, l4_32d.view(b, -1, 1, h, w)], 2)
        # print(l4_2d_cat.shape, l4_3d_cat.shape)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5       
        l5_23d = self.model_23D.L_2D_5(l4_2d_cat)
        l5_32d = self.model_32D.L_3D_5(l4_3d_cat)
        l5_SI = self.model_SI.L_mlp_5(l4_SI)
        #///////////////////////////////////////////////////////////////////////
        # Out
        l5_cat = torch.cat([l5_23d, l5_32d.view(b, -1, h, w)], 1)

        out_SA5 = self.sideout_SA5(self.GAP2D(l5_cat).view(b, -1))
        out_SA4 = self.sideout_SA4(self.GAP2D(l4_2d_cat).view(b, -1))
        out_SA3 = self.sideout_SA3(self.GAP2D(l3_2d_cat).view(b, -1))
        out_SA2 = self.sideout_SA2(self.GAP2D(l2_2d_cat).view(b, -1))
        out_SA1 = self.sideout_SA1(self.GAP2D(l1_2d_cat).view(b, -1))

        out_SI5 = self.sideout_SI5(l5_SI.view(b, -1))
        out_SI4 = self.sideout_SI4(l4_SI.view(b, -1))
        out_SI3 = self.sideout_SI3(l3_SI.view(b, -1))
        out_SI2 = self.sideout_SI2(l2_SI.view(b, -1))
        out_SI1 = self.sideout_SI1(l1_SI.view(b, -1))
        
        out_fusetotal = self.fuse_para[0]*out_SA5 + self.fuse_para[1]*out_SA4 + \
                        self.fuse_para[2]*out_SA3 + self.fuse_para[3]*out_SA2 + \
                        self.fuse_para[4]*out_SA1 + self.fuse_para[5]*out_SI5 + \
                        self.fuse_para[6]*out_SI4 + self.fuse_para[7]*out_SI3 + \
                        self.fuse_para[8]*out_SI2 + self.fuse_para[9]*out_SI1
        return [out_fusetotal, out_SA5, out_SA4, out_SA3, out_SA2, out_SA1, out_SI5, out_SI4, out_SI3, out_SI2, out_SI1]

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