import imp
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Self_Module.ASPP import ASPP
from model.HyperSpectral.Base_Network import *
from model.HyperSpectral.Basic_Operation import compute_ratio_withstep_entireimage
def patch_split(vector, size, overlap=0):
    b, l = vector.size()
    # 计算波段长度并重采样
    img_num = math.ceil(l/(size*size))
    l_re = img_num * size * size
    vector = vector.view(b, 1, l)
    vector_re = F.interpolate(vector, size=(l_re), mode='linear', align_corners=False)

    img_list = []
    for i in range(img_num):
        start = i*size*size
        end = (i+1)*size*size
        split_vect = vector_re[:, :, start:end].view(b, 1, size, size)
        img_list.append(split_vect)
    
    return torch.cat([img for img in img_list], 1)

class HS_SI_3D_SWIN(nn.Module):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI_3D_SWIN model')
        super(HS_SI_3D_SWIN, self).__init__()
        #///////////////////////////////////////////////////////////////////////
        # SP & SA Model
        self.SP_model = SP_2D(7, out_channels)
        self.SA_model = SA3D2D(input_channels, out_channels)
        #///////////////////////////////////////////////////////////////////////
        # fuse side mlps
        self.Fuse_fc1 = get_mlp(2, [2*out_channels, out_channels])
        self.Fuse_fc2 = get_mlp(2, [2*out_channels, out_channels])
        self.Fuse_fc3 = get_mlp(2, [2*out_channels, out_channels])
        self.Fuse_fc4 = get_mlp(2, [2*out_channels, out_channels])
        self.Fuse_fc5 = get_mlp(2, [2*out_channels, out_channels])
        #///////////////////////////////////////////////////////////////////////
        # Fusion Paras
        self.fuse_para = nn.Parameter(torch.ones([15]))
        # Weight Paras
        self.w_para_SP1 = nn.Parameter(torch.ones([1]))
        self.w_para_SP2 = nn.Parameter(torch.ones([1]))
        self.w_para_SP3 = nn.Parameter(torch.ones([1]))
        self.w_para_SA1 = nn.Parameter(torch.ones([1]))
        self.w_para_SA2 = nn.Parameter(torch.ones([1]))
        self.w_para_SA3 = nn.Parameter(torch.ones([1]))
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

    def sideout2d(self, feats, mlp):
        b, c, h, w = feats.size()
        return mlp(self.GAP2D(feats).view(b, -1))
    
    def center_pixel(self, x):
        b, l, h, w = x.size()

        # 得到光谱图
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt = x[:,:,pth,ptw].view(b, -1)
        return pt

    def center_pixel3D(self, x):
        b, c, l, h, w = x.size()

        # 得到光谱图
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt = x[:,:,:,pth,ptw].view(b, c, -1)
        return pt
    
    # forward swin
    def forward(self, x):
        b, c, h, w = x.size()

        pt_center = self.center_pixel(x)
        pt_img = patch_split(pt_center, 4)
        pt_img = compute_ratio_withstep_entireimage(pt_img.view(b, -1, 4*4))
        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.SA_model.l1_3D(x_3d)
        SP1 = self.SP_model.SP_conv_1(pt_img)

        SA1_2D = self.SA_model.SA_CPR1(SA1)
        SA_side1 = self.sideout2d(SA1_2D, self.SA_model.SA_Sideout1)  # 是否加上SP1的sideout
        SP_side1 = self.sideout2d(SP1, self.SP_model.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.l2_3D(SA1)
        SP2 = self.SP_model.SP_conv_2(SP1)

        SA2_2D = self.SA_model.SA_CPR2(SA2)
        SA_side2 = self.sideout2d(SA2_2D, self.SA_model.SA_Sideout2)
        SP_side2 = self.sideout2d(SP2, self.SP_model.SP_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.l3_3D(SA2)
        SP3 = self.SP_model.SP_conv_3(SP2)

        SA3_2D = self.SA_model.SA_CPR3(SA3)
        SA_side3 = self.sideout2d(SA3_2D, self.SA_model.SA_Sideout3)
        SP_side3 = self.sideout2d(SP3, self.SP_model.SP_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.SA_model.l4_2D(SA3_2D)
        SP4 = self.SP_model.SP_conv_4(SP3)

        SA_side4 = self.sideout2d(SA4_2D, self.SA_model.SA_Sideout4)
        SP_side4 = self.sideout2d(SP4, self.SP_model.SP_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.SA_model.l5_2D(SA4_2D)
        SP5 = self.SP_model.SP_conv_5(SP4)

        SA_side5 = self.sideout2d(SA5_2D, self.SA_model.SA_Sideout5)
        SP_side5 = self.sideout2d(SP5, self.SP_model.SP_Sideout5)

        Total_fuse = self.fuse_para[5]*SP_side5 + self.fuse_para[6]*SP_side4 + self.fuse_para[7]*SP_side3+\
                     self.fuse_para[8]*SP_side2 + self.fuse_para[9]*SP_side1+\
                     self.fuse_para[10]*SA_side5 + self.fuse_para[11]*SA_side4 + self.fuse_para[12]*SA_side3+\
                     self.fuse_para[13]*SA_side2 + self.fuse_para[14]*SA_side1
        
        return [Total_fuse, SA_side5, SA_side4, SA_side3, SA_side2, SA_side1, 
                            SP_side5, SP_side4, SP_side3, SP_side2, SP_side1]


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