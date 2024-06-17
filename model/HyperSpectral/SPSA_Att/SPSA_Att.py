import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HyperSpectral.Grid.SpectralImage_Grid import get_mlp
from model.HyperSpectral.Cascade.Cascade import SA3D2D
from model.Self_Module.Layer_operations import sideout2d, CVT3D_2D_SA, CVT3D_2D_SP

class SA_dotatt(nn.Module):
    def __init__(self, feat_num, band_num):
        super(SA_dotatt, self).__init__()

        self.proj = CVT3D_2D_SA(feat_num, feat_num, band_num)

        self.conv1 = nn.Conv2d(feat_num, feat_num, kernel_size=1)
        self.conv2 = nn.Conv2d(feat_num, feat_num, kernel_size=1)
        self.conv3 = nn.Conv2d(feat_num, feat_num, kernel_size=1)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feats):
        '''
        Input: 3D Feats(b, c, l, h, w)
        Output: A Spatial attention map(b, c, h, w)
        '''
        b, c, l, h, w = feats.size()

        feats_2D = self.proj(feats)
        q = self.conv1(feats_2D).view(b, -1, h * w).permute(0, 2, 1)
        k = self.conv2(feats_2D).view(b, -1, h * w)
        attention_s = self.softmax(torch.bmm(q, k))
        v = self.conv3(feats_2D).view(b, -1, h * w)
        feat_e = torch.bmm(v, attention_s.permute(0, 2, 1)).view(b, -1, h, w)
        out = self.alpha * feat_e + feats_2D    # 是否加上2D
        
        # return out
        return feat_e

class SP_dotatt(nn.Module):
    def __init__(self, feat_num, h):
        super(SP_dotatt, self).__init__()

        self.proj = CVT3D_2D_SP(feat_num, feat_num, h)

        self.conv1 = nn.Conv1d(feat_num, feat_num, kernel_size=1)
        self.conv2 = nn.Conv1d(feat_num, feat_num, kernel_size=1)
        self.conv3 = nn.Conv1d(feat_num, feat_num, kernel_size=1)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feats):
        '''
        Input: 3D Feats(b, c, l, h, w)
        Output: A Spatial attention map(b, c, h, w)
        '''
        b, c, l, h, w = feats.size()

        feats_1D = self.proj(feats)
        q = self.conv1(feats_1D).view(b, -1, l).permute(0, 2, 1)
        k = self.conv2(feats_1D).view(b, -1, l)
        attention_s = self.softmax(torch.bmm(q, k))
        v = self.conv3(feats_1D).view(b, -1, l)
        feat_e = torch.bmm(v, attention_s.permute(0, 2, 1)).view(b, -1, l)
        out = self.alpha * feat_e + feats_1D    # 是否加上2D

        # return out
        return feat_e

class spsa_att(SA3D2D):
    def __init__(self, input_channels, out_channels):
        print('Using spsa_att model')
        super(spsa_att,self).__init__(input_channels, out_channels)
        #///////////////////////////////////////////////////////////////////////
        self.SA_att1 = SA_dotatt(16, input_channels)
        self.SA_att2 = SA_dotatt(16, input_channels)
        self.SA_att3 = SA_dotatt(16, input_channels)
        self.SA_att4 = SA_dotatt(16, input_channels)
        self.SA_att5 = SA_dotatt(16, input_channels)
        self.SP_att1 = SP_dotatt(16, 11)
        self.SP_att2 = SP_dotatt(16, 11)
        self.SP_att3 = SP_dotatt(16, 11)
        self.SP_att4 = SP_dotatt(16, 11)
        self.SP_att5 = SP_dotatt(16, 11)
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
        L1_out = self.l1_3D(x_3d)
        L1_out_SA_ATT = self.SA_att1(L1_out).unsqueeze(2)
        L1_out_SP_ATT = self.SP_att1(L1_out).unsqueeze(3).unsqueeze(4)
        L1_W = L1_out_SA_ATT * L1_out_SP_ATT
        L1_W_2D = self.SA_CPR1(L1_W)
        SA_side1 = sideout2d(L1_W_2D, self.SA_Sideout1)  # 是否加上SP1的sideout
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        L2_out = self.l2_3D(L1_W)
        L2_out_SA_ATT = self.SA_att2(L2_out).unsqueeze(2)
        L2_out_SP_ATT = self.SP_att2(L2_out).unsqueeze(3).unsqueeze(4)
        L2_W = L2_out_SA_ATT * L2_out_SP_ATT
        L2_W_2D = self.SA_CPR2(L2_W)
        SA_side2 = sideout2d(L2_W_2D, self.SA_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        L3_out = self.l3_3D(L2_W)
        L3_out_SA_ATT = self.SA_att3(L3_out).unsqueeze(2)
        L3_out_SP_ATT = self.SP_att3(L3_out).unsqueeze(3).unsqueeze(4)
        L3_W = L3_out_SA_ATT * L3_out_SP_ATT
        L3_W_2D = self.SA_CPR3(L3_W)
        SA_side3 = sideout2d(L3_W_2D, self.SA_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        L4_out = self.l4_2D(L3_W_2D)
        SA_side4 = sideout2d(L4_out, self.SA_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        L5_out = self.l5_2D(L4_out)
        SA_side5 = sideout2d(L5_out, self.SA_Sideout5)

        Total_fuse = self.fuse_para[10]*SA_side5 + self.fuse_para[11]*SA_side4 + self.fuse_para[12]*SA_side3+\
                     self.fuse_para[13]*SA_side2 + self.fuse_para[14]*SA_side1
        
        return [Total_fuse, SA_side5, SA_side4, SA_side3, SA_side2, SA_side1]
