import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HyperSpectral.Base_Network import HS_Base

class HS_SI_ENSEMBLE(HS_Base):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI_ENSEMBLE model')
        super(HS_SI_ENSEMBLE,self).__init__(input_channels, out_channels)

        self.half_wind = 1
        self.wind = (self.half_wind*2+1)
        self.SI_conv_1 = nn.Sequential(
            nn.Conv2d(self.wind*self.wind, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        self.SI_conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        self.SI_conv_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

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

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 得到光谱图
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)

        pt = x[:,:,pth-1:pth+2,ptw-1:ptw+2].reshape(b, self.wind*self.wind, 12, 12) #64, 9, 1, 144
        # pt_avg = torch.mean(pt, 1).view(b, 1, 1, -1)
        # print(pt_avg.shape, pt.shape)

        # pt = pt.view(b*self.wind*self.wind, 1, 12, 12)

        # 计算光谱图卷积结果
        p1 = self.SI_conv_1(pt)
        p2 = self.SI_conv_2(p1)
        p3 = self.SI_conv_3(p2)
        p3_GAP = self.GAP(p3)

        # 计算空间图卷积结果
        x1 = self.conv_1(x)
        x1_GAP = self.GAP(x1)
        x2 = self.conv_2(x1)
        x2_GAP = self.GAP(x2)
        x3 = self.conv_3(x2)
        x3_GAP = self.GAP(x3)
        

        # 特征输入MLP
        x3_GAP = x3_GAP.view(b, -1)
        out_SP = self.spatial_fc(x3_GAP)

        p3_GAP = p3_GAP.view(b, -1)
        out_SI_batch = self.SI_fc3(p3_GAP)
        # out_SI_batch = out_SI_batch.view(b, self.wind*self.wind, -1)
        # out_SI_batch = torch.mean(out_SI_batch, 1)
        return [out_SP + out_SI_batch]
