import imp
from telnetlib import X3PAD
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Self_Module.ASPP import ASPP
from model.HyperSpectral.Grid.SpectralImage_Grid import CVT3D_2D

class HS_SI_3D_Grid_3D2D(nn.Module):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI_3D_Grid_3D2D model')
        super(HS_SI_3D_Grid_3D2D,self).__init__()
        #///////////////////////////////////////////////////////////////////////
        # SP & SA Model
        self.l1_3D = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.l2_2D = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.l3_3D = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.l4_2D = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.l5_3D = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.cvt1 = CVT3D_2D(16, 16, input_channels)
        self.cvt2 = CVT3D_2D(16, 16, input_channels)
        self.cvt3 = CVT3D_2D(16, 16, input_channels)
        #///////////////////////////////////////////////////////////////////////
        # fuse side mlps
        self.sideout1 = get_mlp(2, [16, out_channels])
        self.sideout2 = get_mlp(2, [16, out_channels])
        self.sideout3 = get_mlp(2, [16, out_channels])
        self.sideout4 = get_mlp(2, [16, out_channels])
        self.sideout5 = get_mlp(2, [16, out_channels])
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
    
    # forward
    def forward2_Channels(self, x):
        b, c, h, w = x.size()

        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        L1_in = torch.unsqueeze(x, 1)
        L1_out3D = self.l1_3D(L1_in)
        L1_out2D = self.cvt1(L1_out3D)
        L1_sideout = self.sideout2d(L1_out2D, self.sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        L2_out2D = self.l2_2D(L1_out2D)
        L2_sideout = self.sideout2d(L2_out2D, self.sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3
        L3_in = torch.cat([L1_out3D, L2_out2D.unsqueeze(2)], 2) 

        L3_out3D = self.l3_3D(L3_in)
        L3_out2D = self.cvt2(L3_out3D)
        L3_sideout = self.sideout2d(L3_out2D, self.sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        L4_out2D = self.l4_2D(L3_out2D)
        L4_sideout = self.sideout2d(L4_out2D, self.sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        L5_in = torch.cat([L3_out3D, L4_out2D.unsqueeze(2)], 2)  
        L5_out3D = self.l5_3D(L5_in)
        L5_out2D = self.cvt3(L5_out3D)
        L5_sideout = self.sideout2d(L5_out2D, self.sideout5)

        Total_fuse = self.fuse_para[0]*L1_sideout + self.fuse_para[1]*L2_sideout + self.fuse_para[2]*L3_sideout+\
                     self.fuse_para[3]*L4_sideout + self.fuse_para[4]*L5_sideout
        
        return [Total_fuse, L1_sideout, L2_sideout, L3_sideout, L4_sideout, L5_sideout]

    def forward(self, x):
        b, c, h, w = x.size()

        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        L1_in = torch.unsqueeze(x, 1)
        L1_out3D = self.l1_3D(L1_in)
        L1_out2D = self.cvt1(L1_out3D)
        L1_sideout = self.sideout2d(L1_out2D, self.sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        L2_out3D = self.l3_3D(L1_out3D)
        L2_out2D = self.cvt2(L2_out3D)
        L2_sideout = self.sideout2d(L2_out2D, self.sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3
        L3_out3D = self.l5_3D(L2_out3D)
        L3_out2D = self.cvt3(L3_out3D)
        L3_sideout = self.sideout2d(L3_out2D, self.sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        L4_out2D = self.l2_2D(L3_out2D)
        L4_sideout = self.sideout2d(L4_out2D, self.sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        L5_out2D = self.l4_2D(L4_out2D)
        L5_sideout = self.sideout2d(L5_out2D, self.sideout5)

        Total_fuse = self.fuse_para[0]*L1_sideout + self.fuse_para[1]*L2_sideout + self.fuse_para[2]*L3_sideout+\
                     self.fuse_para[3]*L4_sideout + self.fuse_para[4]*L5_sideout
        
        return [Total_fuse, L1_sideout, L2_sideout, L3_sideout, L4_sideout, L5_sideout]


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