from ast import If
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from model.HyperSpectral.Base_Network import HS_Base, HS_Base3D
from model.Self_Module.Attention import SPA_Att
class HS_SI_CM_3D_Similar(HS_Base3D):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI_CM_3D_Similar model')
        super(HS_SI_CM_3D_Similar,self).__init__(input_channels, out_channels)
        #///////////////////////////////////////////////////////////////////////
        # SI Net
        self.SI_conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SI_conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SI_conv_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.GAP2D = torch.nn.AdaptiveAvgPool2d(1)
        #///////////////////////////////////////////////////////////////////////
        # SA sideout mlps & compress conv空间
        self.SA_fc1 = self.get_mlp(3, [16, 256, out_channels])
        self.SA_fc2 = self.get_mlp(3, [32, 256, out_channels])
        self.SA_fc3 = self.get_mlp(3, [64, 256, out_channels])
        # self.SA_fc1 = self.get_mlp(3, [16*11*11, 256, out_channels])
        # self.SA_fc2 = self.get_mlp(3, [32*11*11, 256, out_channels])
        # self.SA_fc3 = self.get_mlp(3, [64*11*11, 256, out_channels])
        # Houston18
        self.SA_CPR1 = nn.Sequential(
            nn.Conv2d(16*42, 16, kernel_size=1),
            nn.BatchNorm2d(16))
        self.SA_CPR2 = nn.Sequential(
            nn.Conv2d(32*36, 32, kernel_size=1),
            nn.BatchNorm2d(32))
        self.SA_CPR3 = nn.Sequential(
            nn.Conv2d(64*30, 64, kernel_size=1),
            nn.BatchNorm2d(64))
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
        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_fc1 = self.get_mlp(3, [16, 256, out_channels])
        self.SP_fc2 = self.get_mlp(3, [32, 256, out_channels])
        self.SP_fc3 = self.get_mlp(3, [64, 256, out_channels])
        #///////////////////////////////////////////////////////////////////////
        # fuse side convs & mlps 融合
        self.Fuse_conv1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16))
        self.Fuse_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32))
        self.Fuse_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64))
        self.Fuse_fc1 = self.get_mlp(3, [16*2, 256, out_channels])
        self.Fuse_fc2 = self.get_mlp(3, [32*2, 256, out_channels])
        self.Fuse_fc3 = self.get_mlp(3, [64*2, 256, out_channels])
        # self.Fuse_fc1 = self.get_mlp(3, [out_channels*2, 256, out_channels])
        # self.Fuse_fc2 = self.get_mlp(3, [out_channels*2, 256, out_channels])
        # self.Fuse_fc3 = self.get_mlp(3, [out_channels*2, 256, out_channels])
        #///////////////////////////////////////////////////////////////////////
        # CM mlps 融合
        self.CM_epoch = torch.ones([out_channels, out_channels]).cuda()
        self.CM_iter = torch.zeros([out_channels, out_channels]).cuda()

        self.CM_fc2 = self.get_mlp(3, [out_channels*out_channels, 64, 64*2])
        self.CM_fc3 = self.get_mlp(3, [out_channels*out_channels, 128, 128*2])
        #///////////////////////////////////////////////////////////////////////
        # Fusion Paras
        self.fuse_para = nn.Parameter(torch.ones([9]))
        #///////////////////////////////////////////////////////////////////////
        # SAM 融合权重
        self.SAM_fc1 = self.get_mlp(3, [121, 16, 16*2])
        self.SAM_fc2 = self.get_mlp(3, [121, 32, 32*2])
        self.SAM_fc3 = self.get_mlp(3, [121, 64, 64*2])
        # 光谱SI 权重
        self.SA_ATT1 = SpatialAtt()
        self.SA_ATT2 = SpatialAtt()
        self.SA_ATT3 = SpatialAtt()
        # 空间SP 权重
        self.SP_ATT1 = SpatialAtt()
        self.SP_ATT2 = SpatialAtt()
        self.SP_ATT3 = SpatialAtt()
        #///////////////////////////////////////////////////////////////////////
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def get_mlp(self, layer_num, node_list, drop_rate=0.2):
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

    def sideout2d(self, feats, mlp):
        b, c, h, w = feats.size()
        return mlp(self.GAP2D(feats).view(b, -1))
    
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

    def forward(self, x):
        b, c, h, w = x.size()

        SI = self.pixel2image(x)
        SI_center = SI[:,60,:,:].unsqueeze(1)

        # Compute SAM weights (弧度还是度？)
        SAM_b = compute_SAM(SI_center, SI, norm=True)
        # SAM_b2 = compute_SAM_np(SI_center, SI)
        # print(torch.max(SAM_b), torch.min(SAM_b))
        SAM_W1 = self.SAM_fc1(SAM_b).view(b, -1, 1, 1)
        SAM_W2 = self.SAM_fc2(SAM_b).view(b, -1, 1, 1)
        SAM_W3 = self.SAM_fc3(SAM_b).view(b, -1, 1, 1)

        # Layer 1
        x_3d = torch.unsqueeze(x, 1)
        SA1 = self.conv_1(x_3d)
        SP1 = self.SI_conv_1(SI_center)
        SA1_2D = self.CVT3d_2dCompress(SA1, self.SA_CPR1)
        # SA1_2D = self.SA_ATT1(SA1_2D)   #空间注意力
        # SP1 = self.SP_ATT1(SP1)         #空间注意力
        
        SA_side1 = self.sideout2d(SA1_2D, self.SA_fc1)
        SP_side1 = self.sideout2d(SP1, self.SP_fc1)
        fuse_vec1 = torch.cat([SA1_2D, SP1], 1)
        fuse_vec1_w = fuse_vec1 * self.softmax((1+SAM_W1))  #融合权重
        Fuse_side1 = self.Fuse_fc1( self.GAP2D(fuse_vec1).view(b, -1) )
        # Fuse_side1 = self.Fuse_fc1(torch.cat([SA_side1, SP_side1], 1))

        # Layer 2
        SA2 = self.conv_2(SA1)
        SP2 = self.SI_conv_2(SP1)
        SA2_2D = self.CVT3d_2dCompress(SA2, self.SA_CPR2)
        # SA2_2D = self.SA_ATT2(SA2_2D)   #空间注意力
        # SP2 = self.SP_ATT2(SP2)         #空间注意力

        SA_side2 = self.sideout2d(SA2_2D, self.SA_fc2)
        SP_side2 = self.sideout2d(SP2, self.SP_fc2)
        fuse_vec2 = torch.cat([SA2_2D, SP2], 1)
        fuse_vec2_w = fuse_vec2 * self.softmax((1+SAM_W2))  #融合权重
        Fuse_side2 = self.Fuse_fc2( self.GAP2D(fuse_vec2).view(b, -1) )
        # Fuse_side2 = self.Fuse_fc2(torch.cat([SA_side2, SP_side2], 1))

        # Layer 3       
        SA3 = self.conv_3(SA2)
        SP3 = self.SI_conv_3(SP2)
        SA3_2D = self.CVT3d_2dCompress(SA3, self.SA_CPR3)
        # SA3_2D = self.SA_ATT3(SA3_2D)   #空间注意力
        # SP3 = self.SP_ATT3(SP3)         #空间注意力

        SA_side3 = self.sideout2d(SA3_2D, self.SA_fc3)
        SP_side3 = self.sideout2d(SP3, self.SP_fc3)
        fuse_vec3 = torch.cat([SA3_2D, SP3], 1)
        fuse_vec3_w = fuse_vec3 * self.softmax((1+SAM_W3))   #融合权重
        # Fuse_side3 = self.Fuse_fc3( self.GAP2D(self.Fuse_conv3(fuse_vec3_w)).view(b, -1) )
        Fuse_side3 = self.Fuse_fc3( self.GAP2D(fuse_vec3).view(b, -1) )
        # Fuse_side3 = self.Fuse_fc3(torch.cat([SA_side3, SP_side3], 1))

        # Total_fuse = self.fuse_para[0]*Fuse_side3 + self.fuse_para[1]*Fuse_side2 + self.fuse_para[2]*Fuse_side1+\
        #              self.fuse_para[3]*SP_side3 + self.fuse_para[4]*SP_side2 + self.fuse_para[5]*SP_side1+\
        #              self.fuse_para[6]*SA_side3 + self.fuse_para[7]*SA_side2 + self.fuse_para[8]*SA_side1
        
        # return [Total_fuse, Fuse_side3, Fuse_side2, Fuse_side1, SP_side3, SP_side2, SP_side1, SA_side3, SA_side2, SA_side1]
        return [SA_side3]

        # return [self.sideout3d_noGAP(SA3, self.SA_fc3)]

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