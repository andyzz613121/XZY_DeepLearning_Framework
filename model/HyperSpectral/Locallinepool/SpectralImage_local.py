from ast import Return
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from model.HyperSpectral.Base_Network import HS_Base

class HS_SI_LOCAL(HS_Base):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI_LOCAL model')
        super(HS_SI_LOCAL,self).__init__(input_channels, out_channels)
        self.SI_conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.SI_conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.SI_conv_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.MaxPool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.dot_fc1 = self.get_mlp(3, [96, 256, out_channels])
        self.dot_fc2 = self.get_mlp(3, [192, 256, out_channels])
        self.dot_fc3 = self.get_mlp(3, [384, 512, out_channels])
        self.dot_fc31 = self.get_mlp(3, [256, 512, out_channels])
        self.dot_fc32 = self.get_mlp(3, [128, 256, out_channels])

        self.conv_dot11 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.conv_dot12 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.conv_dot21 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.conv_dot22 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.conv_dot31 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.conv_dot32 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.GAP = torch.nn.AdaptiveAvgPool2d(1)

        self.SI_fc3 = self.get_mlp(3, [64, 512, out_channels])

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

    def spectral2img_blocks(self, pt):
        b, _, _, c = pt.size()
        c1, c2, c3, c4 =  int(c/4), int(2*c/4), int(3*c/4), c

        if c != 144:
            pt = F.interpolate(pt, size=(1, 144), mode='bilinear')

        pt_1 = pt[:,:,:,0:c1].view(b, -1, 6, 6)
        pt_2 = pt[:,:,:,c1:c2].view(b, -1, 6, 6)
        pt_3 = pt[:,:,:,c2:c3].view(b, -1, 6, 6)
        pt_4 = pt[:,:,:,c3:c4].view(b, -1, 6, 6)
        # print(pt_1.shape, pt_2.shape, pt_3.shape, pt_4.shape)

        pt_img_up = torch.cat([pt_1, pt_2], 2)
        pt_img_down = torch.cat([pt_4, pt_3], 2)
        pt_img = torch.cat([pt_img_up, pt_img_down], 3)
        # print(pt_img.shape)

        return  pt_img


    def local_pool(self, feat_map):
        b, c, h, w = feat_map.size()

        lu = feat_map[:,:,0:int(h/2),0:int(w/2)].reshape(b, c, -1)
        ld = feat_map[:,:,int(h/2):h,0:int(w/2)].reshape(b, c, -1)
        ru = feat_map[:,:,0:int(h/2),int(w/2):w].reshape(b, c, -1)
        rd = feat_map[:,:,int(h/2):h,int(w/2):h].reshape(b, c, -1)
        # print(lu.shape, ld.shape, ru.shape, rd.shape)

        lu_mean = torch.mean(lu, 2).view(b, c, 1, 1)
        ld_mean = torch.mean(ld, 2).view(b, c, 1, 1)
        ru_mean = torch.mean(ru, 2).view(b, c, 1, 1)
        rd_mean = torch.mean(rd, 2).view(b, c, 1, 1)
        # print(lu_mean.shape, ld_mean.shape, ru_mean.shape, rd_mean.shape)

        pool_img_l = torch.cat([lu_mean, ld_mean], 2)
        pool_img_r = torch.cat([ru_mean, rd_mean], 2)
        pool_img = torch.cat([pool_img_l, pool_img_r], 3)
        # print(pool_img.shape)

        return pool_img

    def local_linepool(self, feat_map):
        b, c, h, w = feat_map.size()

        m1, m2, m3, m4 = int(h*w/4), int(2*h*w/4), int(3*h*w/4), int(4*h*w/4)

        feat_map_flat = feat_map.view(b, c, -1)

        mean1 = torch.mean(feat_map_flat[:,:,0:m1], 2).view(b, c, 1, 1)
        mean2 = torch.mean(feat_map_flat[:,:,m1:m2], 2).view(b, c, 1, 1)
        mean3 = torch.mean(feat_map_flat[:,:,m2:m3], 2).view(b, c, 1, 1)
        mean4 = torch.mean(feat_map_flat[:,:,m3:m4], 2).view(b, c, 1, 1)

        return torch.cat([mean1, mean2, mean3, mean4], 3)

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 得到光谱图
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt = x[:,:,pth,ptw].view(b, 1, 1, -1)
        # pt_img = pt.view(b, 1, 12, 12)
        pt_img_block = self.spectral2img_blocks(pt)

        # 计算空间图卷积结果
        x1 = self.conv_1(x)
        x1_GAP = self.GAP(x1)
        x2 = self.conv_2(x1)
        x2_GAP = self.GAP(x2)
        x3 = self.conv_3(x2)
        x3_GAP = self.GAP(x3)
        
        # 计算光谱图卷积结果
        p1 = self.SI_conv_1(pt_img_block)
        p1_MaxPool = self.MaxPool(p1)
        # p1_localpool = self.local_pool(p1)
        p2 = self.SI_conv_2(p1_MaxPool)
        p2_MaxPool = self.MaxPool(p2)
        # p2_localpool = self.local_pool(p2)
        p3 = self.SI_conv_3(p2_MaxPool)
        # p3_MaxPool = self.MaxPool(p3)
        # p3_pool = self.local_pool(p3)
        # p3_pool = self.local_linepool(p3)
        p3_pool = self.GAP(p3)

        # # 计算点击注意力(空间[1, n], 光谱[1, 4n], 结果为[n, 4n])
        # # x1_GAP.shape: torch.Size([64, 32, 1, 1]); p1_localpool.shape: torch.Size([64, 32, 2, 2])
        # # dot_1: torch.Size([64, 32, 128]); dot_2:torch.Size([64, 64, 128]); dot_3:torch.Size([64, 128, 256])
        # dot_1 = torch.bmm(x1_GAP.view(b, -1, 1), p1_localpool.view(b, -1, 1).permute(0, 2, 1))  #torch.Size([64, 32, 1]) torch.Size([64, 1, 128])
        # dot_2 = torch.bmm(x2_GAP.view(b, -1, 1), p2_localpool.view(b, -1, 1).permute(0, 2, 1))
        # dot_3 = torch.bmm(x3_GAP.view(b, -1, 1), p3_localpool.view(b, -1, 1).permute(0, 2, 1))
        # # print(dot_1.shape, dot_2.shape, dot_3.shape)

        # # 将点击注意力矩阵转换为图，然后在进行conv，比直接相加增加非线性
        # # _, h_dot1, w_dot1 = dot_1.size()
        # # dot_11 = dot_1.view(b, h_dot1, 1, w_dot1)
        # # dot_12 = dot_11.reshape(b, w_dot1, 1, h_dot1)
        # # conv_dot_11 = self.conv_dot11(dot_11).view(b, -1)
        # # conv_dot_12 = self.conv_dot12(dot_12).view(b, -1)
        # # out_dot1 = self.dot_fc1(torch.cat([conv_dot_11, conv_dot_12], 1))  #out_dot1: torch.Size([64, 96])

        # # _, h_dot2, w_dot2 = dot_2.size()
        # # dot_21 = dot_2.view(b, h_dot2, 1, w_dot2)
        # # dot_22 = dot_21.reshape(b, w_dot2, 1, h_dot2)
        # # conv_dot_21 = self.conv_dot21(dot_21).view(b, -1)
        # # conv_dot_22 = self.conv_dot22(dot_22).view(b, -1)
        # # out_dot2 = self.dot_fc2(torch.cat([conv_dot_21, conv_dot_22], 1))  #out_dot2: torch.Size([64, 192])

        # _, h_dot3, w_dot3 = dot_3.size()
        # dot_31 = dot_3.view(b, h_dot3, 1, w_dot3)
        # dot_32 = dot_31.reshape(b, w_dot3, 1, h_dot3)
        # conv_dot_31 = self.conv_dot31(dot_31).view(b, -1)
        # conv_dot_32 = self.conv_dot32(dot_32).view(b, -1)
        # # out_dot3 = self.dot_fc3(torch.cat([conv_dot_31, conv_dot_32], 1))  #out_dot1: torch.Size([64, 384])
        # out_dot3 = self.dot_fc31(conv_dot_31) + self.dot_fc32(conv_dot_32)


        x3_GAP = x3_GAP.view(b, -1)
        out1 = self.spatial_fc(x3_GAP)

        p3 = p3_pool.view(b, -1)
        out2 = self.SI_fc3(p3)


        return [out1 + out2]
    
    def forward1(self, x):
        b, c, h, w = x.size()
        
        half_wind = 1
        wind = (half_wind*2+1)
        # 得到光谱图
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)

        pt = x[:,:,pth-1:pth+2,ptw-1:ptw+2].reshape(b, 1, wind*wind, -1) #64, 144, 1, 9
        print(pt.shape)
        pt_img_block = self.spectral2img_blocks(pt)
        print(pt_img_block.shape)
        # 计算空间图卷积结果
        x1 = self.conv_1(x)
        x1_GAP = self.GAP(x1)
        x2 = self.conv_2(x1)
        x2_GAP = self.GAP(x2)
        x3 = self.conv_3(x2)
        x3_GAP = self.GAP(x3)
        
        # 计算光谱图卷积结果
        p1 = self.SI_conv_1(pt_img_block)
        p1_MaxPool = self.GloMaxPool(p1)
        # p1_localpool = self.local_pool(p1)
        p2 = self.SI_conv_2(p1_MaxPool)
        p2_MaxPool = self.MaxPool(p2)
        # p2_localpool = self.local_pool(p2)
        p3 = self.SI_conv_3(p2_MaxPool)
        # p3_MaxPool = self.MaxPool(p3)
        p3_localpool = self.local_pool(p3)


        # 特征输入MLP
        x3_GAP = x3_GAP.view(b, -1)
        out_SP = self.spatial_fc(x3_GAP)

        p3_localpool = self.GAP(p3).view(b, -1)
        out_SI = self.SI_fc3(p3_localpool)
        return [out_SP + out_SI]

