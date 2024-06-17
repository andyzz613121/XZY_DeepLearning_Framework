
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.HyperSpectral.Base_Network import *
from model.HyperSpectral.Basic_Operation import *
from model.Self_Module.Layer_operations import *
from model.HyperSpectral.SISP_2Branch.SISP_2Branch import SISP_2Branch

class SP_2D_MultiScale(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP_2D_MultiScale Net
        print('SP_2D: Using SP_2D_MultiScale model')
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

class SP_2DDilaConv_MultiScale(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(SP_2DDilaConv, self).__init__()
        # SP 2D Net
        print('SP_2D: Using SP_2DDilaConv model')
        self.SP_conv_1 = DilaConv(1, 16, 1)
        self.SP_conv_2 = DilaConv(17, 16, 2)
        self.SP_conv_3 = DilaConv(17, 16, 4)
        self.SP_conv_4 = DilaConv(17, 16, 8)
        self.SP_conv_5 = DilaConv(17, 16, 16)
        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout1 = get_mlp(2, [16, out_channels])
        self.SP_Sideout2 = get_mlp(2, [16, out_channels])
        self.SP_Sideout3 = get_mlp(2, [16, out_channels])
        self.SP_Sideout4 = get_mlp(2, [16, out_channels])
        self.SP_Sideout5 = get_mlp(2, [16, out_channels])

class SISP_2Branch_ImgFuse(SISP_2Branch):
    def __init__(self, in_channel, out_channel):
        super(SISP_2Branch_ImgFuse, self).__init__(in_channel, out_channel)
        print('Using SISP_2Branch_ImgFuse')
        #///////////////////////////////////////////////////////////////////////
        self.Net2D = SP_2D_MultiScale(1, out_channel)
        self.GAP2D = torch.nn.AdaptiveAvgPool2d(1)

        self.fuse_mlp1 = get_mlp(2, [32, out_channel])
        self.fuse_mlp2 = get_mlp(2, [32, out_channel])
        self.fuse_mlp3 = get_mlp(2, [32, out_channel])
        self.fuse_mlp4 = get_mlp(2, [32, out_channel])
        self.fuse_mlp5 = get_mlp(2, [32, out_channel])
        #///////////////////////////////////////////////////////////////////////


#///////////////////////////////////////////////////////////////////////
    # _MultiScale
    def forward(self, x):
        b, c, h, w = x.size()
        
        x_3d = torch.unsqueeze(x, 1)
        pt_img = compute_ratio_withstep_entireimage(x.view(b, c, -1).permute(0, 2, 1))
        pt_input = pt_img.view(-1, c, c).unsqueeze(1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.Net3D.l1_3D(x_3d)
        SA1_2D = self.Net3D.SA_CPR1(SA1)
        SA1_sideout = sideout2d(SA1_2D, self.Net3D.SA_Sideout1)

        SP1 = self.Net2D.SP_conv_1(pt_input)
        SP1_2D = self.GAP2D(SP1).view(b, h*w, -1).permute(0, 2, 1).view(b, -1, h, w)
        SP1_sideout = sideout2d(SP1_2D, self.Net2D.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.Net3D.l2_3D(SA1)
        SA2_2D = self.Net3D.SA_CPR2(SA2)
        SA2_sideout = sideout2d(SA2_2D, self.Net3D.SA_Sideout2)
        
        SP2 = self.Net2D.SP_conv_2(SP1)
        SP2_2D = self.GAP2D(SP2).view(b, h*w, -1).permute(0, 2, 1).view(b, -1, h, w)
        SP2_sideout = sideout2d(SP2_2D, self.Net2D.SP_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.Net3D.l3_3D(SA2)
        SA3_2D = self.Net3D.SA_CPR3(SA3)
        SA3_sideout = sideout2d(SA3_2D, self.Net3D.SA_Sideout3)
    
        SP3 = self.Net2D.SP_conv_3(SP2)
        SP3_2D = self.GAP2D(SP3).view(b, h*w, -1).permute(0, 2, 1).view(b, -1, h, w)
        SP3_sideout = sideout2d(SP3_2D, self.Net2D.SP_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.Net3D.l4_2D(SA3_2D)
        SA4_sideout = sideout2d(SA4_2D, self.Net3D.SA_Sideout4)

        SP4 = self.Net2D.SP_conv_4(SP3)
        SP4_2D = self.GAP2D(SP4).view(b, h*w, -1).permute(0, 2, 1).view(b, -1, h, w)
        SP4_sideout = sideout2d(SP4_2D, self.Net2D.SP_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.Net3D.l5_2D(SA4_2D)
        SA5_sideout = sideout2d(SA5_2D, self.Net3D.SA_Sideout5)

        SP5 = self.Net2D.SP_conv_5(SP4)
        SP5_2D = self.GAP2D(SP5).view(b, h*w, -1).permute(0, 2, 1).view(b, -1, h, w)
        SP5_sideout = sideout2d(SP5_2D, self.Net2D.SP_Sideout5)
        
        # Total_fuse = self.prob_norm(self.fuse_para[0], self.fuse_para[1], SA1_sideout, SP1_sideout) + \
        #              self.prob_norm(self.fuse_para[2], self.fuse_para[3], SA2_sideout, SP2_sideout) + \
        #              self.prob_norm(self.fuse_para[4], self.fuse_para[5], SA3_sideout, SP3_sideout) + \
        #              self.prob_norm(self.fuse_para[6], self.fuse_para[7], SA4_sideout, SP4_sideout) + \
        #              self.prob_norm(self.fuse_para[8], self.fuse_para[9], SA5_sideout, SP5_sideout)
        Fuse_1 = sideout2d(torch.cat([SP1_2D, SA1_2D], 1), self.fuse_mlp1)
        Fuse_2 = sideout2d(torch.cat([SP2_2D, SA2_2D], 1), self.fuse_mlp2)
        Fuse_3 = sideout2d(torch.cat([SP3_2D, SA3_2D], 1), self.fuse_mlp3)
        Fuse_4 = sideout2d(torch.cat([SP4_2D, SA4_2D], 1), self.fuse_mlp4)
        Fuse_5 = sideout2d(torch.cat([SP5_2D, SA5_2D], 1), self.fuse_mlp5)

        Total_fuse = self.fuse_para[0]*SA1_sideout + self.fuse_para[1]*SA2_sideout + self.fuse_para[2]*SA3_sideout+\
                     self.fuse_para[3]*SA4_sideout + self.fuse_para[4]*SA5_sideout+\
                     self.fuse_para[5]*SP1_sideout + self.fuse_para[6]*SP2_sideout + self.fuse_para[7]*SP3_sideout+\
                     self.fuse_para[8]*SP4_sideout + self.fuse_para[9]*SP5_sideout+\
                     self.fuse_para[10]*Fuse_1 + self.fuse_para[11]*Fuse_2 + self.fuse_para[12]*Fuse_3+\
                     self.fuse_para[13]*Fuse_4 + self.fuse_para[14]*Fuse_5
                     
        return [Total_fuse, SA1_sideout, SA2_sideout, SA3_sideout, SA4_sideout, SA5_sideout, 
                            SP1_sideout, SP2_sideout, SP3_sideout, SP4_sideout, SP5_sideout,
                            Fuse_1, Fuse_2, Fuse_3, Fuse_4, Fuse_5]
