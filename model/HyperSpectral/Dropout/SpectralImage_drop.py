import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from model.HyperSpectral.Base_Network import HS_Base

class HS_SI_drop(HS_Base):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI_drop model')
        super(HS_SI_drop,self).__init__(input_channels, out_channels)
        self.fc = self.get_mlp(3, [128, 256, out_channels])
        self.drop_rate = 0.5
        self.drop = nn.Dropout(self.drop_rate)

        self.w = torch.zeros([input_channels]).cuda()      #累计权重
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
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
        c=144 
        self.tmp_w = torch.zeros([c]).cuda()

        w_softmax = self.softmax(self.w) * c
        w_softmax = w_softmax.view(1, c, 1, 1).repeat(b, 1, h, w)
        x = x*w_softmax

        x_drop = x
        zero_list = random.sample(range(0, c), int(c*self.drop_rate))
        for item in zero_list:
            x_drop[:,item,:,:] *= 0
            self.tmp_w[item] += 1

        out_list = []
        for input in [x, x_drop]:
            x1 = self.conv_1(input)
            x2 = self.conv_2(x1)
            x3 = self.conv_3(x2)

            x3_GAP = torch.nn.AdaptiveAvgPool2d(1)(x3)
            x3_GAP = x3_GAP.view(b, -1)
            out = self.fc(x3_GAP)
            out_list.append(out)

        return out_list

    def update_w(self):
        for i in range(self.w.shape[0]):
            self.w[i] += self.tmp_w[i]
