import imp
from telnetlib import X3PAD
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Self_Module.Deform_Conv import DeformableConv2d, DeformConv2D
from model.Self_Module.ASPP import ASPP
from model.HyperSpectral.Grid.SpectralImage_Grid import CVT3D_2D, SA_3D, SP_2D
class SP_2DATT(SP_2D):
    def __init__(self, input_channels, out_channels):
        super().__init__(input_channels, out_channels)
        self.SP_2DCRP1 = CVT2D_1C(16)
        self.SP_2DCRP2 = CVT2D_1C(16)
        self.SP_2DCRP3 = CVT2D_1C(32)
        self.SP_2DCRP4 = CVT2D_1C(32)
        self.SP_2DCRP5 = CVT2D_1C(64)

class CVT2D_1C(nn.Module):
    def __init__(self, in_channels):
        super(CVT2D_1C, self).__init__()
        self.s1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.s1(input)
        return self.bn(self.relu(out))


class HS_SI_3D_Grid_CONV(nn.Module):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI_3D_Grid_CONV model')
        super(HS_SI_3D_Grid_CONV,self).__init__()
        #///////////////////////////////////////////////////////////////////////
        # SP & SA Model
        self.SP_model = SP_2D(1, out_channels)
        print('Using SP_2D')
        self.SA_model = SA_3D(input_channels, out_channels)
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

    def pixel2image(self, x):
        b, c, h, w = x.size()

        SI = x.transpose(2, 1).transpose(2, 3).view(b, h*w, 1, c)  
        SI = F.interpolate(SI, size=(1, 121), mode='bilinear', align_corners=False)
        SI = SI.view(b, h*w, 11, 11)

        return SI
    
    # forward 2_Channels
    def forward(self, x):
        b, c, h, w = x.size()

        pt_center = self.center_pixel(x)
        pt_img_step = compute_index(pt_center)  # 步长要不要取绝对值
        # pt_img_grad = compute_grad(pt_center)
        # pt_img_ratio = compute_ratio(pt_center)
        # pt_img = torch.cat([pt_img_step, pt_img_grad, pt_img_ratio], 1)
        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.SA_model.SA_conv_1(x_3d)
        SP1 = self.SP_model.SP_conv_1(pt_img_step)

        SA1_2D = self.SA_model.SA_CPR1(SA1)
        SA_side1 = self.sideout2d(SA1_2D, self.SA_model.SA_Sideout1)  # 是否加上SP1的sideout
        SP_side1 = self.sideout2d(SP1, self.SP_model.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.SA_conv_2(SA1)
        SP2 = self.SP_model.SP_conv_2(SP1)

        SA2_2D = self.SA_model.SA_CPR2(SA2)
        SA_side2 = self.sideout2d(SA2_2D, self.SA_model.SA_Sideout2)
        SP_side2 = self.sideout2d(SP2, self.SP_model.SP_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.SA_conv_3(SA2)
        SP3 = self.SP_model.SP_conv_3(SP2)

        SA3_2D = self.SA_model.SA_CPR3(SA3)
        SA_side3 = self.sideout2d(SA3_2D, self.SA_model.SA_Sideout3)
        SP_side3 = self.sideout2d(SP3, self.SP_model.SP_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4 = self.SA_model.SA_conv_4(SA3)
        SP4 = self.SP_model.SP_conv_4(SP3)

        SA4_2D = self.SA_model.SA_CPR4(SA4)
        SA_side4 = self.sideout2d(SA4_2D, self.SA_model.SA_Sideout4)
        SP_side4 = self.sideout2d(SP4, self.SP_model.SP_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5 = self.SA_model.SA_conv_5(SA4)
        SP5 = self.SP_model.SP_conv_5(SP4)

        SA5_2D = self.SA_model.SA_CPR5(SA5)
        SA_side5 = self.sideout2d(SA5_2D, self.SA_model.SA_Sideout5)
        SP_side5 = self.sideout2d(SP5, self.SP_model.SP_Sideout5)

        Total_fuse = self.fuse_para[5]*SP_side5 + self.fuse_para[6]*SP_side4 + self.fuse_para[7]*SP_side3+\
                     self.fuse_para[8]*SP_side2 + self.fuse_para[9]*SP_side1+\
                     self.fuse_para[10]*SA_side5 + self.fuse_para[11]*SA_side4 + self.fuse_para[12]*SA_side3+\
                     self.fuse_para[13]*SA_side2 + self.fuse_para[14]*SA_side1
        
        return [Total_fuse, SA_side5, SA_side4, SA_side3, SA_side2, SA_side1, 
                            SP_side5, SP_side4, SP_side3, SP_side2, SP_side1]

    def forward_base(self, x):
        b, c, h, w = x.size()

        pt_center = self.center_pixel(x)
        pt_img = compute_ratio_withstep(pt_center)  # 步长要不要取绝对值
        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.SA_model.SA_conv_1(x_3d)
        SA1_2D = self.SA_model.SA_CPR1(SA1)
        SA_side1 = self.sideout2d(SA1_2D, self.SA_model.SA_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.SA_conv_2(SA1)
        SA2_2D = self.SA_model.SA_CPR2(SA2)
        SA_side2 = self.sideout2d(SA2_2D, self.SA_model.SA_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.SA_conv_3(SA2)
        SA3_2D = self.SA_model.SA_CPR3(SA3)
        SA_side3 = self.sideout2d(SA3_2D, self.SA_model.SA_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4 = self.SA_model.SA_conv_4(SA3)
        SA4_2D = self.SA_model.SA_CPR4(SA4)
        SA_side4 = self.sideout2d(SA4_2D, self.SA_model.SA_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5 = self.SA_model.SA_conv_5(SA4)
        SA5_2D = self.SA_model.SA_CPR5(SA5)
        SA_side5 = self.sideout2d(SA5_2D, self.SA_model.SA_Sideout5)

        
        return [SA_side5, SA_side4, SA_side3, SA_side2, SA_side1]
        # return [SA_side5]


def compute_ratio(vector):
    b, l = vector.size()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    vector[zero_index] = torch.min(vector[nonzero_index])

    vect_squ1 = vector.view(b, 1, l).repeat(1, l, 1)
    vect_squ2 = vector.view(b, l, 1).repeat(1, 1, l)

    return (vect_squ1 / vect_squ2).unsqueeze(1)

def compute_index(vector):
    b, l = vector.size()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    vector[zero_index] = torch.min(vector[nonzero_index])

    vect_squ1 = vector.view(b, 1, l).repeat(1, l, 1)
    vect_squ2 = vector.view(b, l, 1).repeat(1, 1, l)

    return torch.abs(((vect_squ1-vect_squ2) / (vect_squ1 + vect_squ2))).unsqueeze(1)

def compute_ratio_withstep(vector):
    b, l = vector.size()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    vector[zero_index] = torch.min(vector[nonzero_index])

    vect_squ1 = vector.view(b, 1, l).repeat(1, l, 1)
    vect_squ2 = vector.view(b, l, 1).repeat(1, 1, l)

    # step1 - step2之后，步长对角线的值为0，做除数会出错，因此再生成一个单位矩阵，加到step1 - step2之后
    step1 = torch.tensor([i for i in range(l)]).view(1, 1, l).repeat(b, l, 1)
    step2 = torch.tensor([i for i in range(l)]).view(1, l, 1).repeat(b, 1, l)
    step_diag = torch.eye(l).view(1, l, l).repeat(b, 1, 1)
    step = (step1 - step2 + step_diag).cuda()
    return torch.abs((vect_squ1 - vect_squ2)/step).unsqueeze(1)
    # return ((vect_squ1 - vect_squ2)/step).unsqueeze(1)

def compute_ratio_withstep_entireimage(vector):
    b, n, l = vector.size()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    vector[zero_index] = torch.min(vector[nonzero_index])

    vect_squ1 = vector.view(b, n, 1, l).repeat(1, 1, l, 1)
    vect_squ2 = vector.view(b, n, l, 1).repeat(1, 1, 1, l)

    # step1 - step2之后，步长对角线的值为0，做除数会出错，因此再生成一个单位矩阵，加到step1 - step2之后
    step1 = torch.tensor([i for i in range(l)]).view(1, 1, 1, l).repeat(b, n, l, 1)
    step2 = torch.tensor([i for i in range(l)]).view(1, 1, l, 1).repeat(b, n, 1, l)
    step_diag = torch.eye(l).view(1, 1, l, l).repeat(b, n, 1, 1)
    step = (step1 - step2 + step_diag).cuda()
    return torch.abs((vect_squ1 - vect_squ2)/step)

def compute_grad(vector):
    b, l = vector.size()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    vector[zero_index] = torch.min(vector[nonzero_index])

    vect_squ1 = vector.view(b, 1, l).repeat(1, l, 1)
    vect_squ2 = vector.view(b, l, 1).repeat(1, 1, l)

    return (vect_squ1 - vect_squ2).unsqueeze(1)

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