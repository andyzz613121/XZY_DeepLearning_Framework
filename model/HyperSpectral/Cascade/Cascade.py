from turtle import forward
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Self_Module.ASPP import ASPP
from model.HyperSpectral.Grid.SpectralImage_Grid import CVT3D_2D
from model.HyperSpectral.Grid.SpectralImage_GridaAtten import compute_ratio_withstep_entireimage
from model.HyperSpectral.Base_Network import SP_2D, SA3D2D
from model.HyperSpectral.Basic_Operation import compute_index, compute_ratio_withstep

class DilaConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilaConv, self).__init__()
        self.DilaConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
    def forward(self, x):
        return self.DilaConv(x)

class SP_2DDilaConv(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(SP_2DDilaConv, self).__init__()
        # SP 2D Net
        print('SP_2D: Using SP_2DDilaConv model')
        self.SP_conv_1 = DilaConv(1, 16, 1)
        self.SP_conv_2 = DilaConv(16, 16, 2)
        self.SP_conv_3 = DilaConv(16, 16, 4)
        self.SP_conv_4 = DilaConv(16, 16, 8)
        self.SP_conv_5 = DilaConv(16, 16, 16)
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



class Cascade_3D2D_BASE(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(Cascade_3D2D_BASE, self).__init__()
        #///////////////////////////////////////////////////////////////////////
        self.SA_model = SA3D2D(input_channels, out_channels)
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

    def sideout2d(self, feats, mlp):
        b, c, h, w = feats.size()
        return mlp(self.GAP2D(feats).view(b, -1))

    def forward(self, x):
        b, c, h, w = x.size()

        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.SA_model.l1_3D(x_3d)
        SA1_2D = self.SA_model.SA_CPR1(SA1)
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

class Act_Layer1D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(Act_Layer1D, self).__init__()
        self.Act_Conv1D = nn.Sequential(
            nn.Conv1d(input_channels, out_channels, 7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.Act_Conv1D(x)

class Cascade_3D2D(Cascade_3D2D_BASE):
    def __init__(self, input_channels, out_channels):
        print('Using Cascade_3D2D model')
        super(Cascade_3D2D, self).__init__(input_channels, out_channels)
        # SP & SA Model
        # self.SP_model = SP_2DDilaConv(1, out_channels)
        self.SP_model = SP_2D(1, out_channels)
        #///////////////////////////////////////////////////////////////////////
        # ATT MLP
        rate = math.ceil(input_channels/16)
        # ipt_c_re = int(rate * 16)
        ipt_c_re = input_channels
        self.mlp1_1 = get_mlp(2, [ipt_c_re, ipt_c_re*2, ipt_c_re])
        self.mlp1_2 = get_mlp(2, [ipt_c_re, ipt_c_re*2, ipt_c_re])
        # self.mlp2_1 = get_mlp(2, [int(ipt_c_re/2), int(ipt_c_re/2)])
        # self.mlp2_2 = get_mlp(2, [int(ipt_c_re/2), int(ipt_c_re/2)])
        # self.mlp3_1 = get_mlp(2, [int(ipt_c_re/4), int(ipt_c_re/4)])
        # self.mlp3_2 = get_mlp(2, [int(ipt_c_re/4), int(ipt_c_re/4)])
        # self.mlp4_1 = get_mlp(2, [int(ipt_c_re/8), int(ipt_c_re/8)])
        # self.mlp4_2 = get_mlp(2, [int(ipt_c_re/8), int(ipt_c_re/8)])
        # self.mlp5_1 = get_mlp(2, [int(ipt_c_re/16), int(ipt_c_re/16)])
        # self.mlp5_2 = get_mlp(2, [int(ipt_c_re/16), int(ipt_c_re/16)])
        # self.mlp1_1 = Act_Layer1D(1, 1)
        # self.mlp1_2 = Act_Layer1D(1, 1)
        # self.mlp2_1 = Act_Layer1D(1, 1)
        # self.mlp2_2 = Act_Layer1D(1, 1)
        # self.mlp3_1 = Act_Layer1D(1, 1)
        # self.mlp3_2 = Act_Layer1D(1, 1)
        # self.mlp4_1 = Act_Layer1D(1, 1)
        # self.mlp4_2 = Act_Layer1D(1, 1)
        # self.mlp5_1 = Act_Layer1D(1, 1)
        # self.mlp5_2 = Act_Layer1D(1, 1)
        self.mlp2_1 = get_mlp(2, [ipt_c_re, ipt_c_re])
        self.mlp2_2 = get_mlp(2, [ipt_c_re, ipt_c_re])
        self.mlp3_1 = get_mlp(2, [ipt_c_re, ipt_c_re])
        self.mlp3_2 = get_mlp(2, [ipt_c_re, ipt_c_re])
        self.mlp4_1 = get_mlp(2, [ipt_c_re, ipt_c_re])
        self.mlp4_2 = get_mlp(2, [ipt_c_re, ipt_c_re])
        self.mlp5_1 = get_mlp(2, [ipt_c_re, ipt_c_re])
        self.mlp5_2 = get_mlp(2, [ipt_c_re, ipt_c_re])
        self.mlp_list1 = [self.mlp1_1, self.mlp2_1, self.mlp3_1, self.mlp4_1, self.mlp5_1]
        self.mlp_list2 = [self.mlp1_2, self.mlp2_2, self.mlp3_2, self.mlp4_2, self.mlp5_2]
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
    def center_pixel(self, x):
        b, l, h, w = x.size()

        # 得到光谱图
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt = x[:,:,pth,ptw].view(b, -1)
        return pt

    def compute_spatt_mlp(self, vector, mlp1, mlp2):
        b, n, l = vector.size()
        att_maps = []
        for img_num in range(n):
            vect_patch = vector[:,img_num,:]
            # print(vect_patch.shape)
            vector_emb1 = mlp1(vect_patch).view(b, -1, l).permute(0, 2, 1)
            vector_emb2 = mlp2(vect_patch).view(b, -1, l)
            attention_s = self.softmax(torch.bmm(vector_emb1, vector_emb2)).view(b, 1, l, l)
            # vector_emb1, vector_emb2 = mlp1(vect_patch), mlp2(vect_patch)
            # attention_s = self.compute_ratio_withstep(vector_emb1, vector_emb2)
            att_maps.append(attention_s)
        
        att_maps = torch.cat([x for x in att_maps], 1)
        return att_maps

    # def compute_ratio_withstep(self, vector1, vector2):
    #     b, l = vector1.size()

    #     # 处理0值，将其处理为非0值里面的最小值
    #     zero_index = (vector1 == 0)
    #     nonzero_index = (vector1!=0)
    #     vector1[zero_index] = torch.min(vector1[nonzero_index])
    #     zero_index = (vector2 == 0)
    #     nonzero_index = (vector2!=0)
    #     vector2[zero_index] = torch.min(vector2[nonzero_index])

    #     vect_squ1 = vector1.view(b, 1, l).repeat(1, l, 1)
    #     vect_squ2 = vector2.view(b, l, 1).repeat(1, 1, l)

    #     # step1 - step2之后，步长对角线的值为0，做除数会出错，因此再生成一个单位矩阵，加到step1 - step2之后
    #     step1 = torch.tensor([i for i in range(l)]).view(1, 1, l).repeat(b, l, 1)
    #     step2 = torch.tensor([i for i in range(l)]).view(1, l, 1).repeat(b, 1, l)
    #     step_diag = torch.eye(l).view(1, l, l).repeat(b, 1, 1)
    #     step = (step1 - step2 + step_diag).cuda()
    #     return torch.abs((vect_squ1 - vect_squ2)/step).unsqueeze(1)
    #     # return ((vect_squ1 - vect_squ2)/step).unsqueeze(1)

    def compute_spatt_conv1D(self, vector, conv1D_1, conv1D_2):
        b, n, l = vector.size()
        att_maps = []
        for img_num in range(n):
            vect_patch = vector[:,img_num,:]
            vector_emb1 = conv1D_1(vect_patch.view(b, -1, l)).permute(0, 2, 1)
            vector_emb2 = conv1D_2(vect_patch.view(b, -1, l))
            attention_s = self.softmax(torch.bmm(vector_emb1, vector_emb2)).view(b, 1, l, l)
            att_maps.append(attention_s)
        
        att_maps = torch.cat([x for x in att_maps], 1)
        return att_maps

    def patch_cascade(self, vector, act_list1, act_list2, cascade_level=[1, 2, 4, 8, 16]):
        b, l = vector.size()
        # 计算波段长度并重采样, 5层级联，因此除以16
        rate = math.ceil(l/16)
        l_re = rate * 16
        # l_re = l
        vector = vector.view(b, 1, l)
        vector_re = F.interpolate(vector, size=(l_re), mode='linear', align_corners=False)
        
        patch_cascade = []
        layer = 0
        for split_para in cascade_level:
            img_split = torch.split(vector_re, int(l_re/split_para), dim=2)
            img_split = torch.cat([x for x in img_split], 1)
            img_split = self.compute_spatt_mlp(img_split, act_list1[layer], act_list2[layer])
            patch_cascade.append(img_split)
            layer += 1
        return patch_cascade

    def Spectral_Weighted_MulChannel(self, Feats_3D, ratio_img):
        torch.cuda.empty_cache()
        b, c, l, h, w = Feats_3D.size()
        
        Feats_3D_Ori = Feats_3D
        att_map = self.softmax(ratio_img)

        Feats_3D = Feats_3D.view(b, c, l, h*w).permute(0, 1, 3, 2)
        att_map = att_map.view(b, c, l, l)
        result = torch.matmul(Feats_3D, att_map).permute(0, 1, 3, 2).view(b, c, l, h, w)

        return result + Feats_3D_Ori

    # forward cascade
    def forward(self, x):
        b, c, h, w = x.size()

        pt_center = self.center_pixel(x)
        # pt_img = self.patch_cascade(pt_center, self.mlp_list1, self.mlp_list2, [1, 2, 4, 8, 16])
        # pt_img = self.patch_cascade(pt_center, self.mlp_list1, self.mlp_list2, [1, 1, 1, 1, 1])
        # pt_img = self.compute_spatt_mlp(pt_center.view(b, 1, -1), self.mlp1_1, self.mlp1_2)
        pt_img = compute_index(pt_center)

        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.SA_model.l1_3D(x_3d)
        SP1 = self.SP_model.SP_conv_1(pt_img)

        SA1_W = self.Spectral_Weighted_MulChannel(SA1, SP1)
        SA1_2D = self.SA_model.SA_CPR1(SA1_W)
        SA_side1 = self.sideout2d(SA1_2D, self.SA_model.SA_Sideout1)  # 是否加上SP1的sideout
        SP_side1 = self.sideout2d(SP1, self.SP_model.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.l2_3D(SA1)
        SP2 = self.SP_model.SP_conv_2(SP1)
        SA2_W = self.Spectral_Weighted_MulChannel(SA2, SP2)
        SA2_2D = self.SA_model.SA_CPR2(SA2_W)
        SA_side2 = self.sideout2d(SA2_2D, self.SA_model.SA_Sideout2)
        SP_side2 = self.sideout2d(SP2, self.SP_model.SP_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.l3_3D(SA2)
        SP3 = self.SP_model.SP_conv_3(SP2)
        SA3_W = self.Spectral_Weighted_MulChannel(SA3, SP3)
        SA3_2D = self.SA_model.SA_CPR3(SA3_W)
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

        # Total_fuse = self.fuse_para[5]*SP_side5 + self.fuse_para[6]*SP_side4 + self.fuse_para[7]*SP_side3+\
        #              self.fuse_para[8]*SP_side2 + self.fuse_para[9]*SP_side1+\
        #              self.fuse_para[10]*SA_side5 + self.fuse_para[11]*SA_side4 + self.fuse_para[12]*SA_side3+\
        #              self.fuse_para[13]*SA_side2 + self.fuse_para[14]*SA_side1
        
        # return [Total_fuse, SA_side5, SA_side4, SA_side3, SA_side2, SA_side1, 
        #                     SP_side5, SP_side4, SP_side3, SP_side2, SP_side1]

        Total_fuse = self.fuse_para[10]*SA_side5 + self.fuse_para[11]*SA_side4 + self.fuse_para[12]*SA_side3+\
                     self.fuse_para[13]*SA_side2 + self.fuse_para[14]*SA_side1

        # return [Total_fuse, SA_side5, SA_side4, SA_side3, SA_side2, SA_side1]
        return [Total_fuse]

    # forward swin
    def forwardswin(self, x):
        b, c, h, w = x.size()
        pt_center = self.center_pixel(x)
        pt_img = self.patch_cascade(pt_center, self.mlp_list1, self.mlp_list2, [1, 2, 4, 8, 16])

        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.SA_model.l1_3D(x_3d)
        SP1 = self.SP_model.SP_conv_1(pt_img[0])
        # SA1_W = self.Spectral_Weighted_MulChannel(SA1, SP1)
        SA1_2D = self.SA_model.SA_CPR1(SA1)
        SA_side1 = self.sideout2d(SA1_2D, self.SA_model.SA_Sideout1)  # 是否加上SP1的sideout
        SP_side1 = self.sideout2d(SP1, self.SP_model.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.l2_3D(SA1)
        SP2 = self.SP_model.SP_conv_2(torch.cat([SP1, pt_img[1]], 1))
        # SA2_W = self.Spectral_Weighted_MulChannel(SA2, SP2)
        SA2_2D = self.SA_model.SA_CPR2(SA2)
        SA_side2 = self.sideout2d(SA2_2D, self.SA_model.SA_Sideout2)
        SP_side2 = self.sideout2d(SP2, self.SP_model.SP_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.l3_3D(SA2)
        SP3 = self.SP_model.SP_conv_3(torch.cat([SP2, pt_img[2]], 1))
        # SA3_W = self.Spectral_Weighted_MulChannel(SA3, SP3)
        SA3_2D = self.SA_model.SA_CPR3(SA3)
        SA_side3 = self.sideout2d(SA3_2D, self.SA_model.SA_Sideout3)
        SP_side3 = self.sideout2d(SP3, self.SP_model.SP_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.SA_model.l4_2D(SA3_2D)
        SP4 = self.SP_model.SP_conv_4(torch.cat([SP3, pt_img[3]], 1))

        SA_side4 = self.sideout2d(SA4_2D, self.SA_model.SA_Sideout4)
        SP_side4 = self.sideout2d(SP4, self.SP_model.SP_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.SA_model.l5_2D(SA4_2D)
        SP5 = self.SP_model.SP_conv_5(torch.cat([SP4, pt_img[4]], 1))

        SA_side5 = self.sideout2d(SA5_2D, self.SA_model.SA_Sideout5)
        SP_side5 = self.sideout2d(SP5, self.SP_model.SP_Sideout5)

        # Total_fuse = self.fuse_para[5]*SP_side5 + self.fuse_para[6]*SP_side4 + self.fuse_para[7]*SP_side3+\
        #              self.fuse_para[8]*SP_side2 + self.fuse_para[9]*SP_side1+\
        #              self.fuse_para[10]*SA_side5 + self.fuse_para[11]*SA_side4 + self.fuse_para[12]*SA_side3+\
        #              self.fuse_para[13]*SA_side2 + self.fuse_para[14]*SA_side1
        
        # return [Total_fuse, SA_side5, SA_side4, SA_side3, SA_side2, SA_side1, 
        #                     SP_side5, SP_side4, SP_side3, SP_side2, SP_side1]

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


