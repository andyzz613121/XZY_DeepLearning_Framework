from ast import If
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from model.HyperSpectral.Base_Network import HS_Base, HS_Base3D
from model.Self_Module.Deform_Conv import DeformableConv2d, DeformConv2D
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
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout1 = get_mlp(2, [16, out_channels])
        self.SP_Sideout2 = get_mlp(2, [16, out_channels])
        self.SP_Sideout3 = get_mlp(2, [32, out_channels])
        self.SP_Sideout4 = get_mlp(2, [32, out_channels])
        self.SP_Sideout5 = get_mlp(2, [64, out_channels])

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
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.SA_conv_2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.SA_conv_3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.SA_conv_4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.SA_conv_5 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.spatial_fc = get_mlp(3, [128, 256, out_channels])
        
        #///////////////////////////////////////////////////////////////////////
        # SA sideout mlps & compress conv空间
        self.SA_Sideout1 = get_mlp(2, [16, out_channels])
        self.SA_Sideout2 = get_mlp(2, [16, out_channels])
        self.SA_Sideout3 = get_mlp(2, [32, out_channels])
        self.SA_Sideout4 = get_mlp(2, [32, out_channels])
        self.SA_Sideout5 = get_mlp(2, [64, out_channels])

        # 3D压缩为2D
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
        # self.SA_CPR5 = nn.Sequential(
        #     nn.Conv2d(128*120, 128, kernel_size=1),
        #     nn.BatchNorm2d(128))
        
        # Pavia
        self.SA_CPR1 = CVT3D_2D(16, 16, 103)
        self.SA_CPR2 = CVT3D_2D(16, 16, 103)
        self.SA_CPR3 = CVT3D_2D(32, 32, 103)
        self.SA_CPR4 = CVT3D_2D(32, 32, 103)
        self.SA_CPR5 = CVT3D_2D(64, 64, 103)

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

class CVT3D_2D(nn.Module):
    def __init__(self, in_channels, out_channels, k=49, bias=True):
        super(CVT3D_2D, self).__init__()
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k,3,3), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.s1(input)
        out = out.squeeze(2)
        return self.bn(self.relu(out))

class HS_SI_3D_Deform(nn.Module):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI_3D_Deform model')
        super(HS_SI_3D_Deform,self).__init__()
        #///////////////////////////////////////////////////////////////////////
        # SP & SA Model
        self.SP_model = SP_2D(1, out_channels)
        print('Using SP_2D')
        # self.SP_model = SP_1D(input_channels, out_channels)
        # print('Using SP_1D')
        # self.SP_model = SP_MLP(input_channels, out_channels)
        # print('Using SP_MLP')
        self.SA_model = SA_3D(input_channels, out_channels)
        
        #///////////////////////////////////////////////////////////////////////
        # fuse side mlps
        # self.Fuse_fc1 = get_mlp(3, [40, 256, out_channels])
        # self.Fuse_fc2 = get_mlp(3, [40, 256, out_channels])
        # self.Fuse_fc3 = get_mlp(3, [40, 256, out_channels])
        self.Fuse_fc1 = get_mlp(2, [2*out_channels, out_channels])
        self.Fuse_fc2 = get_mlp(2, [2*out_channels, out_channels])
        self.Fuse_fc3 = get_mlp(2, [2*out_channels, out_channels])
        self.Fuse_fc4 = get_mlp(2, [2*out_channels, out_channels])
        self.Fuse_fc5 = get_mlp(2, [2*out_channels, out_channels])
        # self.Fuse_fc1 = self.get_mlp(3, [out_channels*2, 256, out_channels])
        # self.Fuse_fc2 = self.get_mlp(3, [out_channels*2, 256, out_channels])
        # self.Fuse_fc3 = self.get_mlp(3, [out_channels*2, 256, out_channels])
        #///////////////////////////////////////////////////////////////////////
        # SAM 融合权重
        self.SAM_fc1 = get_mlp(3, [121, 16, 16*2])
        self.SAM_fc2 = get_mlp(3, [121, 32, 32*2])
        self.SAM_fc3 = get_mlp(3, [121, 64, 64*2])

        #///////////////////////////////////////////////////////////////////////
        # 空间注意力
        self.SA_ATT1 = SpatialAtt()
        self.SA_ATT2 = SpatialAtt()
        self.SA_ATT3 = SpatialAtt()

        self.SP_ATT1 = SpatialAtt()
        self.SP_ATT2 = SpatialAtt()
        self.SP_ATT3 = SpatialAtt()

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
        self.std_T = nn.Parameter(torch.tensor(0.5))
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

        self.bn3d1 = nn.BatchNorm3d(16)
        self.bn3d2 = nn.BatchNorm3d(32)
        self.bn2d1 = nn.BatchNorm2d(16)
        self.bn2d2 = nn.BatchNorm2d(32)
    def sideout2d(self, feats, mlp):
        b, c, h, w = feats.size()
        return mlp(self.GAP2D(feats).view(b, -1))
    
    def sideout1d(self, feats, mlp):
        b, c, l = feats.size()
        feats = feats.view(b, c, l, 1)
        return mlp(self.GAP2D(feats).view(b, -1))
    
    def sideoutMLP(self, feats, mlp):
        return mlp(feats)

    def sideout2d_noGAP(self, feats, mlp):
        b, c, h, w = feats.size()
        return mlp(feats.view(b, -1))
    
    def CVT3d_2dCompress(self, feats, compress_conv):
        feats = torch.reshape(feats, (feats.shape[0], -1, feats.shape[3], feats.shape[4]))
        return compress_conv(feats)

    def sideout3d(self, feats, mlp, compress_conv):
        b, c, _, h, w = feats.size()
        return mlp(self.GAP2D(self.compress_conv(feats)).view(b, -1))

    def sideout3d_noGAP(self, feats, mlp):
        b, c, _, h, w = feats.size()
        return mlp(feats.view(b, -1))

    def pixel2image(self, x):
        b, c, h, w = x.size()

        SI = x.transpose(2, 1).transpose(2, 3).view(b, h*w, 1, c)  
        SI = F.interpolate(SI, size=(1, 121), mode='bilinear', align_corners=False)
        SI = SI.view(b, h*w, 11, 11)

        return SI
    
    # forward 2D
    def forward(self, x):
        b, c, h, w = x.size()

        SI = self.pixel2image(x)
        SI_center = SI[:,60,:,:].unsqueeze(1)

        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        x_3d = torch.unsqueeze(x, 1)
        SA1 = self.SA_model.SA_conv_1(x_3d)
        SP1 = self.SP_model.SP_conv_1(SI_center)
        # SA1_2D = self.CVT3d_2dCompress(SA1, self.SA_model.SA_CPR1)
        SA1_2D = self.SA_model.SA_CPR1(SA1)
        
        SA_side1 = self.sideout2d(SA1_2D, self.SA_model.SA_Sideout1)
        SP_side1 = self.sideout2d(SP1, self.SP_model.SP_Sideout1)
        Fuse_side1 = self.Fuse_fc1(torch.cat([SA_side1, SP_side1], 1))
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.SA_conv_2(SA1)
        SA2 = self.bn3d1(SA2 + SA1)
        SP2 = self.SP_model.SP_conv_2(SP1)
        SP2 = self.bn2d1(SP2 + SP1)

        # SA2_2D = self.CVT3d_2dCompress(SA2, self.SA_model.SA_CPR2)
        SA2_2D = self.SA_model.SA_CPR2(SA2)

        SA_side2 = self.sideout2d(SA2_2D, self.SA_model.SA_Sideout2)
        SP_side2 = self.sideout2d(SP2, self.SP_model.SP_Sideout2)
        Fuse_side2 = self.Fuse_fc2(torch.cat([SA_side2, SP_side2], 1))
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.SA_conv_3(SA2)
        SP3 = self.SP_model.SP_conv_3(SP2)
        # SA3_2D = self.CVT3d_2dCompress(SA3, self.SA_model.SA_CPR3)
        SA3_2D = self.SA_model.SA_CPR3(SA3)

        SA_side3 = self.sideout2d(SA3_2D, self.SA_model.SA_Sideout3)
        SP_side3 = self.sideout2d(SP3, self.SP_model.SP_Sideout3)
        Fuse_side3 = self.Fuse_fc3(torch.cat([SA_side3, SP_side3], 1))
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4 = self.SA_model.SA_conv_4(SA3)
        SP4 = self.SP_model.SP_conv_4(SP3)
        SA4 = self.bn3d2(SA4 + SA3)
        SP4 = self.bn2d2(SP4 + SP3)
        # SA4_2D = self.CVT3d_2dCompress(SA4, self.SA_model.SA_CPR4)
        SA4_2D = self.SA_model.SA_CPR4(SA4)

        SA_side4 = self.sideout2d(SA4_2D, self.SA_model.SA_Sideout4)
        SP_side4 = self.sideout2d(SP4, self.SP_model.SP_Sideout4)
        Fuse_side4 = self.Fuse_fc4(torch.cat([SA_side4, SP_side4], 1))
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5 = self.SA_model.SA_conv_5(SA4)
        SP5 = self.SP_model.SP_conv_5(SP4)
        # SA5_2D = self.CVT3d_2dCompress(SA5, self.SA_model.SA_CPR5)
        SA5_2D = self.SA_model.SA_CPR5(SA5)

        SA_side5 = self.sideout2d(SA5_2D, self.SA_model.SA_Sideout5)
        SP_side5 = self.sideout2d(SP5, self.SP_model.SP_Sideout5)
        Fuse_side5 = self.Fuse_fc5(torch.cat([SA_side5, SP_side5], 1))
        #///////////////////////////////////////////////////////////////////////
        # Total_fuse = self.fuse_para[0]*Fuse_side3 + self.fuse_para[1]*Fuse_side2 + self.fuse_para[2]*Fuse_side1+\
        #              self.fuse_para[3]*SP_side3 + self.fuse_para[4]*SP_side2 + self.fuse_para[5]*SP_side1+\
        #              self.fuse_para[6]*SA_side3 + self.fuse_para[7]*SA_side2 + self.fuse_para[8]*SA_side1
        # return [Total_fuse, Fuse_side3, Fuse_side2, Fuse_side1, SP_side3, SP_side2, SP_side1, SA_side3, SA_side2, SA_side1]
        Total_fuse = self.fuse_para[0]*Fuse_side5 + self.fuse_para[1]*Fuse_side4 + self.fuse_para[2]*Fuse_side3+\
                     self.fuse_para[3]*Fuse_side2 + self.fuse_para[4]*Fuse_side1 +\
                     self.fuse_para[5]*SP_side5 + self.fuse_para[6]*SP_side4 + self.fuse_para[7]*SP_side3+\
                     self.fuse_para[8]*SP_side2 + self.fuse_para[9]*SP_side1+\
                     self.fuse_para[10]*SA_side5 + self.fuse_para[11]*SA_side4 + self.fuse_para[12]*SA_side3+\
                     self.fuse_para[13]*SA_side2 + self.fuse_para[14]*SA_side1
        
        return [Total_fuse, Fuse_side5, Fuse_side4, Fuse_side3, Fuse_side2, Fuse_side1, 
                SP_side5, SP_side4, SP_side3, SP_side2, SP_side1, 
                SA_side5, SA_side4, SA_side3, SA_side2, SA_side1]


    # forward 1D
    def forward_1D(self, x):
        b, c, h, w = x.size()

        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt_center = x[:,:,pth,ptw].unsqueeze(1)

        # # Compute SAM weights
        # SAM_b = compute_SAM(SI_center, SI, norm=True)
        # SAM_W1 = self.SAM_fc1(SAM_b).view(b, -1, 1, 1)
        # SAM_W2 = self.SAM_fc2(SAM_b).view(b, -1, 1, 1)
        # SAM_W3 = self.SAM_fc3(SAM_b).view(b, -1, 1, 1)

        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        x_3d = torch.unsqueeze(x, 1)
        SA1 = self.SA_model.SA_conv_1(x_3d)
        SP1 = self.SP_model.SP_conv_1(pt_center)
        SA1_2D = self.CVT3d_2dCompress(SA1, self.SA_model.SA_CPR1)

        SA_side1 = self.sideout2d(SA1_2D, self.SA_model.SA_Sideout1)
        SP_side1 = self.sideout1d(SP1, self.SP_model.SP_Sideout1)
        fuse_vec1 = torch.cat([SA_side1, SP_side1], 1)
        # fuse_vec1_w = fuse_vec1 * self.softmax((1+SAM_W1))  #融合权重
        Fuse_side1 = self.Fuse_fc1(fuse_vec1)

        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.SA_conv_2(SA1)
        SP2 = self.SP_model.SP_conv_2(SP1)
        SA2_2D = self.CVT3d_2dCompress(SA2, self.SA_model.SA_CPR2)

        SA_side2 = self.sideout2d(SA2_2D, self.SA_model.SA_Sideout2)
        SP_side2 = self.sideout1d(SP2, self.SP_model.SP_Sideout2)
        fuse_vec2 = torch.cat([SA_side2, SP_side2], 1)
        Fuse_side2 = self.Fuse_fc2(fuse_vec2)

        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.SA_conv_3(SA2)
        SP3 = self.SP_model.SP_conv_3(SP2)
        SA3_2D = self.CVT3d_2dCompress(SA3, self.SA_model.SA_CPR3)

        SA_side3 = self.sideout2d(SA3_2D, self.SA_model.SA_Sideout3)
        SP_side3 = self.sideout1d(SP3, self.SP_model.SP_Sideout3)
        fuse_vec3 = torch.cat([SA_side3, SP_side3], 1)
        Fuse_side3 = self.Fuse_fc3(fuse_vec3)

        Total_fuse = self.fuse_para[0]*Fuse_side3 + self.fuse_para[1]*Fuse_side2 + self.fuse_para[2]*Fuse_side1+\
                     self.fuse_para[3]*SP_side3 + self.fuse_para[4]*SP_side2 + self.fuse_para[5]*SP_side1+\
                     self.fuse_para[6]*SA_side3 + self.fuse_para[7]*SA_side2 + self.fuse_para[8]*SA_side1
        
        return [Total_fuse, Fuse_side3, Fuse_side2, Fuse_side1, SP_side3, SP_side2, SP_side1, SA_side3, SA_side2, SA_side1]

    # forward MLP
    def forward_mlp(self, x):
        b, c, h, w = x.size()

        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt_center = x[:,:,pth,ptw]

        # # Compute SAM weights
        # SAM_b = compute_SAM(SI_center, SI, norm=True)
        # SAM_W1 = self.SAM_fc1(SAM_b).view(b, -1, 1, 1)
        # SAM_W2 = self.SAM_fc2(SAM_b).view(b, -1, 1, 1)
        # SAM_W3 = self.SAM_fc3(SAM_b).view(b, -1, 1, 1)

        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        x_3d = torch.unsqueeze(x, 1)
        SA1 = self.SA_model.SA_conv_1(x_3d)
        SP1 = self.SP_model.SP_conv_1(pt_center)
        SA1_2D = self.CVT3d_2dCompress(SA1, self.SA_model.SA_CPR1)

        SA_side1 = self.sideout2d(SA1_2D, self.SA_model.SA_Sideout1)
        SP_side1 = self.sideoutMLP(SP1, self.SP_model.SP_Sideout1)
        fuse_vec1 = torch.cat([SA_side1, SP_side1], 1)
        # fuse_vec1_w = fuse_vec1 * self.softmax((1+SAM_W1))  #融合权重
        Fuse_side1 = self.Fuse_fc1(fuse_vec1)

        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.SA_conv_2(SA1)
        SP2 = self.SP_model.SP_conv_2(SP1)
        SA2_2D = self.CVT3d_2dCompress(SA2, self.SA_model.SA_CPR2)

        SA_side2 = self.sideout2d(SA2_2D, self.SA_model.SA_Sideout2)
        SP_side2 = self.sideoutMLP(SP2, self.SP_model.SP_Sideout2)
        fuse_vec2 = torch.cat([SA_side2, SP_side2], 1)
        Fuse_side2 = self.Fuse_fc2(fuse_vec2)

        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.SA_conv_3(SA2)
        SP3 = self.SP_model.SP_conv_3(SP2)
        SA3_2D = self.CVT3d_2dCompress(SA3, self.SA_model.SA_CPR3)

        SA_side3 = self.sideout2d(SA3_2D, self.SA_model.SA_Sideout3)
        SP_side3 = self.sideoutMLP(SP3, self.SP_model.SP_Sideout3)
        fuse_vec3 = torch.cat([SA_side3, SP_side3], 1)
        Fuse_side3 = self.Fuse_fc3(fuse_vec3)

        Total_fuse = self.fuse_para[0]*Fuse_side3 + self.fuse_para[1]*Fuse_side2 + self.fuse_para[2]*Fuse_side1+\
                     self.fuse_para[3]*SP_side3 + self.fuse_para[4]*SP_side2 + self.fuse_para[5]*SP_side1+\
                     self.fuse_para[6]*SA_side3 + self.fuse_para[7]*SA_side2 + self.fuse_para[8]*SA_side1
        
        return [Total_fuse, Fuse_side3, Fuse_side2, Fuse_side1, SP_side3, SP_side2, SP_side1, SA_side3, SA_side2, SA_side1]

    def forward_base(self, x):
        b, c, h, w = x.size()

        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        x_3d = torch.unsqueeze(x, 1)
        SA1 = self.SA_model.SA_conv_1(x_3d)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.SA_conv_2(SA1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.SA_conv_3(SA2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4  
        SA4 = self.SA_model.SA_conv_4(SA3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5 = self.SA_model.SA_conv_5(SA4)
        SA5_2D = self.CVT3d_2dCompress(SA5, self.SA_model.SA_CPR5)
        Fuse_side5 = self.Fuse_fc5( self.GAP2D(SA5_2D).view(b, -1) )
        
        return [Fuse_side5]

class SpatialAtt(nn.Module):
    def __init__(self):
        super(SpatialAtt, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_ori = x
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max], 1)
        x = self.conv(x)
        return x_ori*self.sigmoid(x)

def compute_SAM(vec1, vec2, norm):
    b, c, h, w = vec2.size()
    #Torch 版本
    vec1 = vec1.view(b, 1, -1).repeat(1, c, 1).view(-1, h*w, 1)
    vec1_T = vec1.transpose(2, 1)
    vec2 = vec2.view(b, c, -1).view(-1, h*w, 1)
    vec2_T = vec2.transpose(2, 1)
    SAM_dot = torch.bmm(vec1_T,vec2) / ( torch.sqrt(torch.bmm(vec1_T,vec1)) * torch.sqrt(torch.bmm(vec2_T,vec2)))
    neg_index = (SAM_dot > 1)
    SAM_dot[neg_index] = 1
    SAM = torch.arccos(SAM_dot)
    SAM = SAM.view(b, c)
    SAM[:,60] = 0
    if norm == True:
        SAM = SAM/torch.max(SAM)
    
    return SAM

def compute_SAM_np(vec1, vec2):
    b, c, h, w = vec2.size()
    
    vec1 = vec1.view(b, -1)                      # b, band_num
    vec2 = vec2.view(b, c, -1).transpose(2, 1)   # b, band_num, pixel_num
    vec1 = vec1.cpu().detach().numpy()
    vec2 = vec2.cpu().detach().numpy()
    SAM_total = []
    for batch in range(b):
        vec1_batch = vec1[batch]                 # band_num
        vec2_batch = vec2[batch]                 # band_num * pixel_num
        SAM_b = []
        for pixel in range(c):
            vec2_batch_pixel = vec2_batch[:, pixel]
            SAM_dot = np.dot(vec1_batch.T,vec2_batch_pixel) / ( np.sqrt(np.dot(vec1_batch.T,vec1_batch)) * np.sqrt(np.dot(vec2_batch_pixel.T,vec2_batch_pixel)))
            if SAM_dot > 1:
                SAM_dot = 1
            SAM_bp = np.arccos(SAM_dot)
            SAM_b.append(SAM_bp)
        SAM_total.append(SAM_b)
    SAM_total = np.array(SAM_total)
    print(SAM_total[:, 60])
    return SAM_total

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