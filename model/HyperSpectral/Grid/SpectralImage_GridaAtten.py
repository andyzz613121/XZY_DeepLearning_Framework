import imp
from telnetlib import X3PAD
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Self_Module.Deform_Conv import DeformableConv2d, DeformConv2D
from model.Self_Module.ASPP import ASPP
from model.HyperSpectral.Base_Network import SP_2D, SA3D2D

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

class SelfAttention_Mapping(nn.Module):
    def __init__(self, input_channels):
        super(SelfAttention_Mapping, self).__init__()
        # self.conv3d = nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0))
        # self.bn = nn.BatchNorm2d(input_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.aspp = ASPP(input_channels, input_channels, [1, 1, 1])

    def forward(self, input):
        out = self.aspp(input)
        # out = out.squeeze(2)
        return out

class HS_SI_3D_Grid_ATTEN(nn.Module):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI_3D_Grid_ATTEN model')
        super(HS_SI_3D_Grid_ATTEN,self).__init__()
        #///////////////////////////////////////////////////////////////////////
        # SP & SA Model
        self.SP_model = SP_2D(1, out_channels)
        print('Using SP_2D')
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
        self.std_T = nn.Parameter(torch.tensor(0.5))
        # Att_alpha
        self.Att_alpha1 = nn.Parameter(torch.ones([1]))
        self.Att_alpha2 = nn.Parameter(torch.ones([1]))
        self.Att_alpha3 = nn.Parameter(torch.ones([1]))
        self.Att_alpha4 = nn.Parameter(torch.ones([1]))
        self.Att_alpha5 = nn.Parameter(torch.ones([1]))
        #///////////////////////////////////////////////////////////////////////
        # Self Attention
        self.CRPpt_img1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True))
        self.CRPpt_img2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.CRPpt_img3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.CRPpt_img4 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.CRPpt_img5 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
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
    
    def Spectral_Weighted_MulChannel(self, Feats_3D, ratio_img):
        torch.cuda.empty_cache()
        b, c, l, h, w = Feats_3D.size()
        Feats_3D_Ori = Feats_3D
        att_map = self.softmax(ratio_img)

        Feats_3D = Feats_3D.view(b, c, l, h*w).permute(0, 1, 3, 2)
        att_map = att_map.view(b, c, l, l)
        result = torch.matmul(Feats_3D, att_map).permute(0, 1, 3, 2).view(b, c, l, h, w)

        return result + Feats_3D_Ori  

    def Spectral_Weighted(self, Feats_3D, ratio_img):
        b, c, l, h, w = Feats_3D.size()
        Feats_3D_Ori = Feats_3D
        att_map = self.softmax(ratio_img)

        Feats_3D = Feats_3D.view(b, c, l, h*w).permute(0, 1, 3, 2)
        att_map = att_map.view(b, 1, l, l)
        result = torch.matmul(Feats_3D, att_map).permute(0, 1, 3, 2).view(b, c, l, h, w)

        return result + Feats_3D_Ori  # 可以尝试：out = self.alpha * feat_e + x 以及 softmax 以及 torch.matmul(Feats_3D, att_map)中att_map.permute(0, 2, 1)
                                      # 以及输入不一定是pt_img
    
    # forward 2D
    def forward2D(self, x):
        b, c, h, w = x.size()

        pt_center = self.center_pixel(x)
        pt_img = compute_ratio_withstep(pt_center)  # 步长要不要取绝对值
        # print(pt_img.shape)
        x_3d = torch.unsqueeze(x, 1)
        x3d_W = self.Spectral_Weighted(x_3d, pt_img)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.SA_model.SA_conv_1(x3d_W)
        SA1_W = self.Spectral_Weighted(SA1, pt_img)
        SA1_2D = self.SA_model.SA_CPR1(SA1)
        SA_side1 = self.sideout2d(SA1_2D, self.SA_model.SA_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.SA_conv_2(SA1)
        SA2_W = self.Spectral_Weighted(SA2, pt_img)
        SA2_2D = self.SA_model.SA_CPR2(SA2)
        SA_side2 = self.sideout2d(SA2_2D, self.SA_model.SA_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.SA_conv_3(SA2)
        SA3_W = self.Spectral_Weighted(SA3, pt_img)
        SA3_2D = self.SA_model.SA_CPR3(SA3)
        SA_side3 = self.sideout2d(SA3_2D, self.SA_model.SA_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4 = self.SA_model.SA_conv_4(SA3)
        SA4_W = self.Spectral_Weighted(SA4, pt_img)
        SA4_2D = self.SA_model.SA_CPR4(SA4)
        SA_side4 = self.sideout2d(SA4_2D, self.SA_model.SA_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5 = self.SA_model.SA_conv_5(SA4)
        SA5_W = self.Spectral_Weighted(SA5, pt_img)
        SA5_2D = self.SA_model.SA_CPR5(SA5)
        SA_side5 = self.sideout2d(SA5_2D, self.SA_model.SA_Sideout5)

        Total_fuse = self.fuse_para[10]*SA_side5 + self.fuse_para[11]*SA_side4 + self.fuse_para[12]*SA_side3+\
                     self.fuse_para[13]*SA_side2 + self.fuse_para[14]*SA_side1
        
        return [Total_fuse, SA_side5, SA_side4, SA_side3, SA_side2, SA_side1]

    # forward 用3D的中心像素进行注意力计算
    def forward_3Dcenter(self, x):
        b, c, h, w = x.size()

        pt_center = self.center_pixel(x)
        pt_img = compute_ratio_withstep(pt_center)  # 步长要不要取绝对值

        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.SA_model.SA_conv_1(x_3d)
        pt_center = self.center_pixel3D(SA1).view(b, -1, c, 1, 1)
        pt_Q1 = self.Q1(pt_center).view(b, -1, c)
        pt_img = compute_ratio_withstep_entireimage(pt_Q1)  

        SA1_W = self.Spectral_Weighted_MulChannel(SA1, pt_img)
        SA1_2D = self.SA_model.SA_CPR1(SA1_W)
        SA_side1 = self.sideout2d(SA1_2D, self.SA_model.SA_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.SA_conv_2(SA1)
        pt_center = self.center_pixel3D(SA2).view(b, -1, c, 1, 1)
        pt_Q2 = self.Q2(pt_center).view(b, -1, c)
        pt_img = compute_ratio_withstep_entireimage(pt_Q2)  

        SA2_W = self.Spectral_Weighted_MulChannel(SA2, pt_img)
        SA2_2D = self.SA_model.SA_CPR2(SA2_W)
        SA_side2 = self.sideout2d(SA2_2D, self.SA_model.SA_Sideout2)
        # ///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.SA_conv_3(SA2)
        pt_center = self.center_pixel3D(SA3).view(b, -1, c, 1, 1)
        pt_Q3 = self.Q3(pt_center).view(b, -1, c)
        pt_img = compute_ratio_withstep_entireimage(pt_Q3)  

        SA3_W = self.Spectral_Weighted_MulChannel(SA3, pt_img)
        SA3_2D = self.SA_model.SA_CPR3(SA3_W)
        SA_side3 = self.sideout2d(SA3_2D, self.SA_model.SA_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4 = self.SA_model.SA_conv_4(SA3)
        pt_center = self.center_pixel3D(SA4).view(b, -1, c, 1, 1)
        pt_Q4 = self.Q4(pt_center).view(b, -1, c)
        pt_img = compute_ratio_withstep_entireimage(pt_Q4)  

        SA4_W = self.Spectral_Weighted_MulChannel(SA4, pt_img)
        SA4_2D = self.SA_model.SA_CPR4(SA4_W)
        SA_side4 = self.sideout2d(SA4_2D, self.SA_model.SA_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5 = self.SA_model.SA_conv_5(SA4)
        pt_center = self.center_pixel3D(SA5).view(b, -1, c, 1, 1)
        pt_Q5 = self.Q5(pt_center).view(b, -1, c)
        pt_img = compute_ratio_withstep_entireimage(pt_Q5)  

        SA5_W = self.Spectral_Weighted_MulChannel(SA5, pt_img)
        SA5_2D = self.SA_model.SA_CPR5(SA5_W)
        SA_side5 = self.sideout2d(SA5_2D, self.SA_model.SA_Sideout5)

        Total_fuse = self.fuse_para[10]*SA_side5 + self.fuse_para[11]*SA_side4 + self.fuse_para[12]*SA_side3+\
                     self.fuse_para[13]*SA_side2 + self.fuse_para[14]*SA_side1
        
        return [Total_fuse, SA_side5, SA_side4, SA_side3, SA_side2, SA_side1]
        
    # forward 2_Channels
    def forward(self, x):
        b, c, h, w = x.size()

        pt_center = self.center_pixel(x)
        pt_img = compute_ratio_withstep(pt_center)  # 步长要不要取绝对值

        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.SA_model.l1_3D(x_3d)
        SP1 = self.SP_model.SP_conv_1(pt_img)
        SA1_W = self.Spectral_Weighted_MulChannel(SA1, SP1)

        SA1_2D = self.SA_model.SA_CPR1(SA1_W)
        SA_side1 = self.sideout2d(SA1_2D, self.SA_model.SA_Sideout1)  # 是否加上SP1的sideout
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.SA_model.l2_3D(SA1)
        SP2 = self.SP_model.SP_conv_2(SP1)
        SA2_W = self.Spectral_Weighted_MulChannel(SA2, SP2)

        SA2_2D = self.SA_model.SA_CPR2(SA2_W)
        SA_side2 = self.sideout2d(SA2_2D, self.SA_model.SA_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.SA_model.l3_3D(SA2)
        SP3 = self.SP_model.SP_conv_3(SP2)
        SA3_W = self.Spectral_Weighted_MulChannel(SA3, SP3)

        SA3_W = self.Spectral_Weighted(SA3, pt_img)
        SA3_2D = self.SA_model.SA_CPR3(SA3_W)
        SA_side3 = self.sideout2d(SA3_2D, self.SA_model.SA_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.SA_model.l4_2D(SA3_2D)
        SP4 = self.SP_model.SP_conv_4(SP3)

        SA_side4 = self.sideout2d(SA4_2D, self.SA_model.SA_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.SA_model.l5_2D(SA4_2D)
        SP5 = self.SP_model.SP_conv_5(SP4)

        SA_side5 = self.sideout2d(SA5_2D, self.SA_model.SA_Sideout5)

        Total_fuse = self.fuse_para[10]*SA_side5 + self.fuse_para[11]*SA_side4 + self.fuse_para[12]*SA_side3+\
                     self.fuse_para[13]*SA_side2 + self.fuse_para[14]*SA_side1
        
        return [Total_fuse, SA_side5, SA_side4, SA_side3, SA_side2, SA_side1]

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


def compute_ratio(vector):
    b, l = vector.size()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    vector[zero_index] = torch.min(vector[nonzero_index])

    vect_squ1 = vector.view(b, 1, l).repeat(1, l, 1)
    vect_squ2 = vector.view(b, l, 1).repeat(1, 1, l)

    return (vect_squ1 / vect_squ2).unsqueeze(1)

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