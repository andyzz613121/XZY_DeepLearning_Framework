import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HyperSpectral.Base_Network import HS_Base
from model.Self_Module.Deform_Conv import DeformConv2D, DeformableConv2d
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()

        modules = []
        
        # 增加1*1卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))
        # 增加3*3空洞卷积
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        # 增加可变卷积
        # modules.append(DeformableConv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.convs = nn.ModuleList(modules)

        # 空洞卷积结果降维（1个1*1，3个3*3，1个池化，一个可变卷积）
        self.project = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class HS_SI_LOCAL_ASPP(HS_Base):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI_LOCAL_ASPP model')
        super(HS_SI_LOCAL_ASPP,self).__init__(input_channels, out_channels)
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

        self.ASPP = ASPP(64, 256, [2, 4, 8])

        self.softmax = nn.Softmax(dim=-1)
        self.GAP = torch.nn.AdaptiveAvgPool2d(1)

        self.SI_fc3 = self.get_mlp(3, [256, 512, out_channels])

        self.fc_weight1 = self.get_mlp(3, [256+32, 512, 256+32])
        self.fc_weight2 = self.get_mlp(3, [256+64, 512, 256+64])
        self.fc_weight3 = self.get_mlp(3, [256+128, 512, 256+128])

        self.sideout1 = self.get_mlp(3, [256+32, 512, out_channels])
        self.sideout2 = self.get_mlp(3, [256+64, 512, out_channels])
        self.sideout3 = self.get_mlp(3, [256+128, 512, out_channels])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias != None:
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

        pt_1 = pt[:,:,:,0:c1].view(b, 1, 6, 6)
        pt_2 = pt[:,:,:,c1:c2].view(b, 1, 6, 6)
        pt_3 = pt[:,:,:,c2:c3].view(b, 1, 6, 6)
        pt_4 = pt[:,:,:,c3:c4].view(b, 1, 6, 6)
        # print(pt_1.shape, pt_2.shape, pt_3.shape, pt_4.shape)

        pt_img_up = torch.cat([pt_1, pt_2], 2)
        pt_img_down = torch.cat([pt_4, pt_3], 2)
        pt_img = torch.cat([pt_img_up, pt_img_down], 3)
        # print(pt_img.shape)

        return  pt_img
    def spectral2img_lines(self, pt, lines_num=4):
        b, _, _, c = pt.size()
        c1, c2, c3, c4 =  int(c/4), int(2*c/4), int(3*c/4), c

        if c != 144:
            pt = F.interpolate(pt, size=(1, 144), mode='bilinear')

        pt_1 = pt[:,:,:,0:c1].view(b, 1, 6, 6)
        pt_2 = pt[:,:,:,c1:c2].view(b, 1, 6, 6)
        pt_3 = pt[:,:,:,c2:c3].view(b, 1, 6, 6)
        pt_4 = pt[:,:,:,c3:c4].view(b, 1, 6, 6)
        # print(pt_1.shape, pt_2.shape, pt_3.shape, pt_4.shape)

        pt_img_up = torch.cat([pt_1, pt_2], 2)
        pt_img_down = torch.cat([pt_4, pt_3], 2)
        pt_img = torch.cat([pt_img_up, pt_img_down], 3)
        # print(pt_img.shape)

        return  pt_img
    
    def local_linepool(self, feat_map):
        b, c, h, w = feat_map.size()

        m1, m2, m3, m4 = int(h*w/4), int(2*h*w/4), int(3*h*w/4), int(4*h*w/4)

        feat_map_flat = feat_map.view(b, c, -1)

        mean1 = torch.mean(feat_map_flat[:,:,0:m1], 2).view(b, c, 1, 1)
        mean2 = torch.mean(feat_map_flat[:,:,m1:m2], 2).view(b, c, 1, 1)
        mean3 = torch.mean(feat_map_flat[:,:,m2:m3], 2).view(b, c, 1, 1)
        mean4 = torch.mean(feat_map_flat[:,:,m3:m4], 2).view(b, c, 1, 1)

        return torch.cat([mean1, mean2, mean3, mean4], 3)

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
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 得到光谱图
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        # pt_ensamble = x[:,:,pth-1:pth+2,ptw-1:ptw+2].reshape(b, 9, 12, 12)
        pt_img = self.spectral2img_blocks(x[:,:,pth,ptw].view(b, 1, 1, -1))

        # 计算空间图卷积结果
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        
        # 计算光谱图卷积结果
        p1 = self.SI_conv_1(pt_img)
        p1 = self.MaxPool(p1)
        p2 = self.SI_conv_2(p1)
        p2 = self.MaxPool(p2)
        p3 = self.SI_conv_3(p2)
        p3 = self.MaxPool(p3)
        aspp = self.ASPP(p3)
        
        aspp_up1 = F.interpolate(aspp, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        aspp_up2 = F.interpolate(aspp, size=(x2.shape[2], x2.shape[3]), mode='bilinear')
        aspp_up3 = F.interpolate(aspp, size=(x3.shape[2], x3.shape[3]), mode='bilinear')

        #sideout
        SISA_fuse1 = torch.cat([x1, aspp_up1], 1)
        w1 = self.fc_weight1(self.GAP(SISA_fuse1).view(b, -1)).view(b, -1, 1, 1).repeat(1, 1, SISA_fuse1.shape[2], SISA_fuse1.shape[3])
        # print(w1.shape, SISA_fuse1.shape, SISA_fuse1*w1)
        # sideout1 = self.sideout1(self.GAP(SISA_fuse1 + SISA_fuse1*w1).view(b, -1))
        sideout1 = self.sideout1(self.GAP(SISA_fuse1).view(b, -1))

        SISA_fuse2 = torch.cat([x2, aspp_up2], 1)
        w2 = self.fc_weight2(self.GAP(SISA_fuse2).view(b, -1)).view(b, -1, 1, 1).repeat(1, 1, SISA_fuse2.shape[2], SISA_fuse2.shape[3])
        sideout2 = self.sideout2(self.GAP(SISA_fuse2).view(b, -1))

        SISA_fuse3 = torch.cat([x3, aspp_up3], 1)
        w3 = self.fc_weight3(self.GAP(SISA_fuse3).view(b, -1)).view(b, -1, 1, 1).repeat(1, 1, SISA_fuse3.shape[2], SISA_fuse3.shape[3])
        sideout3 = self.sideout3(self.GAP(SISA_fuse3).view(b, -1))
        return [sideout3, sideout2, sideout1]

        # # 特征输入MLP
        # x3_GAP = x3_GAP.view(b, -1)
        # out_SP = self.spatial_fc(x3_GAP)

        # p3_localpool = self.GAP(aspp).view(b, -1)
        # # p3_localpool = self.local_linepool(aspp).view(b, -1)
        # out_SI = self.SI_fc3(p3_localpool)
        # return [out_SP + out_SI]

