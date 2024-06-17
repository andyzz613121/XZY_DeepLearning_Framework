import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.SA_conv_2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.SA_conv_3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.SA_conv_4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.SA_conv_5 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
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
        self.SA_CPR1 = CVT3D_2D(16, 16, input_channels)
        self.SA_CPR2 = CVT3D_2D(16, 16, input_channels)
        self.SA_CPR3 = CVT3D_2D(32, 32, input_channels)
        self.SA_CPR4 = CVT3D_2D(32, 32, input_channels)
        self.SA_CPR5 = CVT3D_2D(64, 64, input_channels)

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
    '''
    当输入in_channels个3D特征图的时候，每个特征图深度为k时，对每一个3D特征图用一个深度为k的卷积核卷积为深度为1，即变为2D
    '''
    def __init__(self, in_channels, out_channels, k=49, bias=True):
        super(CVT3D_2D, self).__init__()
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k,1,1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.s1(input)
        out = out.squeeze(2)
        return self.bn(self.relu(out))

class HS_SI_3D_Grid(nn.Module):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI_3D_Grid model')
        super(HS_SI_3D_Grid,self).__init__()
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
        self.Fuse_fc1 = get_mlp(2, [2*out_channels, out_channels])
        self.Fuse_fc2 = get_mlp(2, [2*out_channels, out_channels])
        self.Fuse_fc3 = get_mlp(2, [2*out_channels, out_channels])
        self.Fuse_fc4 = get_mlp(2, [2*out_channels, out_channels])
        self.Fuse_fc5 = get_mlp(2, [2*out_channels, out_channels])
        # self.Fuse_fc1 = self.get_mlp(3, [out_channels*2, 256, out_channels])
        # self.Fuse_fc2 = self.get_mlp(3, [out_channels*2, 256, out_channels])
        # self.Fuse_fc3 = self.get_mlp(3, [out_channels*2, 256, out_channels])
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

    def center_pixel(self, x):
        b, c, h, w = x.size()

        # 得到光谱图
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt = x[:,:,pth,ptw].view(b, -1)
        return pt

    def pixel2image(self, x):
        b, c, h, w = x.size()

        SI = x.transpose(2, 1).transpose(2, 3).view(b, h*w, 1, c)  
        SI = F.interpolate(SI, size=(1, 121), mode='bilinear', align_corners=False)
        SI = SI.view(b, h*w, 11, 11)

        return SI
    
    # forward 2D
    def forward(self, x):
        b, c, h, w = x.size()

        pt_center = self.center_pixel(x)
        pt_img = compute_ratio_withstep(pt_center)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        x_3d = torch.unsqueeze(x, 1)
        SA1 = self.SA_model.SA_conv_1(x_3d)
        SP1 = self.SP_model.SP_conv_1(pt_img)
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

def compute_ratio(vector):
    b, l = vector.size()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    vector[zero_index] = torch.min(vector[nonzero_index])

    # 计算比例
    ratio_matrix = torch.ones([b, l, l]).cuda()
    for batch in range(b):
        for band in range(l):
            ratio_matrix[batch][band] = vector[batch]/vector[batch][band]
    ratio_matrix = ratio_matrix.unsqueeze(1)
    return ratio_matrix

def compute_ratio_withstep(vector):
    b, l = vector.size()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    vector[zero_index] = torch.min(vector[nonzero_index])

    # 计算比例
    grid_matrix = torch.ones([b, l, l]).cuda()
    step_list = torch.tensor([x for x in range(l)]).cuda()
    for batch in range(b):
        for band in range(l):
            grid_matrix[batch][band] = vector[batch]-vector[batch][band]
            steplist_tmp = step_list - band
            steplist_tmp[band] = 1
            grid_matrix[batch][band] = grid_matrix[batch][band] / steplist_tmp

    grid_matrix = grid_matrix.unsqueeze(1)
    grid_matrix = torch.abs(grid_matrix)
    return grid_matrix

def compute_grad(vector):
    b, l = vector.size()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    vector[zero_index] = torch.min(vector[nonzero_index])

    # 计算比例
    grid_matrix = torch.ones([b, l, l]).cuda()
    for batch in range(b):
        for band in range(l):
            grid_matrix[batch][band] = vector[batch]-vector[batch][band]
    grid_matrix = grid_matrix.unsqueeze(1)
    return grid_matrix

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