from locale import normalize
import re
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from model.ResNet.ResNet import resnet101
from model.Self_Module.CARB import CARB_Block

class Mean_Std_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Mean_Std_Attention,self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        #到底是1个MLP输入为2*n维，还是2个MLP输入为1*n
        self.mean_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )
        self.std_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x_std = self.global_std(x)
        x_means = self.avg_pool(x).view(b, c)
        std_w = self.std_fc(x_std)
        mean_w = self.mean_fc(x_means)
        
        #两者如何融合
        w_sum = (std_w + mean_w).view(b, c, 1, 1)

        return x*w_sum

    def global_std(self, x):

        b, c, h, w = x.size()
        x = x.reshape([b,c,h*w])
        std = torch.std(x, 2)
        std = std.view(b, c)
        return std

class CM_Attention(nn.Module):
    def __init__(self, layer_num, node_list):
        super(CM_Attention, self).__init__()
        assert layer_num == len(node_list), "ERROR at Weight_MLP: Layer_num != len(node_list)"
        self.MLP = self.get_mlp(layer_num, node_list)

    def get_mlp(self, layer_num, node_list, drop_rate=0.2):
        layers = []
        for layer in range(layer_num-1):
            layers.append(nn.Linear(node_list[layer], node_list[layer+1]))
            if layer+1 != (layer_num-1):  #Last layer
                layers.append(nn.Dropout(drop_rate))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
        mlp = nn.Sequential(*layers)
        for m in mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        return mlp

    def forward(self, x, confuse_matrix):
        b, c, _, _ = x.size()
        confuse_matrix_flatten = torch.reshape(confuse_matrix, (1, -1))
        #混淆矩阵是一个Batch统计全部，还是每个Batch的图像分别统计
        CM_weight = self.MLP(confuse_matrix_flatten)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(x, CM_weight).permute(0, 3, 1, 2).contiguous()
        return x


class ResU_Net_Difference(nn.Module):
    def __init__(self, input_channels, num_classes, pre_train=True):
        super(ResU_Net_Difference, self).__init__()
        
        self.resnet = resnet101(input_channels, True)

        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4

        self.decoder4 = DecoderBlock(2048, 1024)
        self.decoder3 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder1 = DecoderBlock(256, 64)
        
        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 3, padding=1)
        )

        self.MSA4 = Mean_Std_Attention(2048)
        self.MSA3 = Mean_Std_Attention(1024)
        self.MSA2 = Mean_Std_Attention(512)
        self.MSA1 = Mean_Std_Attention(256)
        self.MSA0 = Mean_Std_Attention(64)

        self.Sideout4 = CARB_Block(1024, 6)
        self.Sideout3 = CARB_Block(512, 6)
        self.Sideout2 = CARB_Block(256, 6)
        self.Sideout1 = CARB_Block(64, 6)

        self.CWA4 = CM_Attention(3, [36, 1024//2, 1024])
        self.CWA3 = CM_Attention(3, [36, 512//2, 512])
        self.CWA2 = CM_Attention(3, [36, 256//2, 256])
        self.CWA1 = CM_Attention(3, [36, 64//2, 64])

        self.MSA_CWA_Fuse4 = CARB_Block(1024*2, 1024)
        self.MSA_CWA_Fuse3 = CARB_Block(512*2, 512)
        self.MSA_CWA_Fuse2 = CARB_Block(256*2, 256)
        self.MSA_CWA_Fuse1 = CARB_Block(64*2, 64)

    def forward(self, x, label):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_unpool = x
        x = self.resnet.maxpool(x)      #64

        e1 = self.encoder1(x)           #256
        e2 = self.encoder2(e1)          #512
        e3 = self.encoder3(e2)          #1024
        e4 = self.encoder4(e3)          #2048

        # d4_in = self.MSA4(e4)
        # d4_out = self.decoder4(d4_in)                       
        # d4_map = self.Sideout4(d4_out)
        # d4_map = F.interpolate(d4_map, size=(label.shape[1], label.shape[2]), mode='bilinear')
        # d4_pre = torch.argmax(d4_map, 1)
        # CM4 = self.cal_confuse_matrix(d4_pre, label, normalize=True)
        # d4 = self.CWA4(d4_out, CM4) + self.MSA3(e3)

        # d3_in = d4
        # d3_out = self.decoder3(d3_in)                       
        # d3_map = self.Sideout3(d3_out)
        # d3_map = F.interpolate(d3_map, size=(label.shape[1], label.shape[2]), mode='bilinear')
        # d3_pre = torch.argmax(d3_map, 1)
        # CM3 = self.cal_confuse_matrix(d3_pre, label, normalize=True)
        # d3 = self.CWA3(d3_out, CM3) + self.MSA2(e2)

        # d2_in = d3
        # d2_out = self.decoder2(d2_in)                       
        # d2_map = self.Sideout2(d2_out)
        # d2_map = F.interpolate(d2_map, size=(label.shape[1], label.shape[2]), mode='bilinear')
        # d2_pre = torch.argmax(d2_map, 1)
        # CM2 = self.cal_confuse_matrix(d2_pre, label, normalize=True)
        # d2 = self.CWA2(d2_out, CM2) + self.MSA1(e1)

        # d1_in = d2
        # d1_out = self.decoder1(d1_in)                       
        # d1_map = self.Sideout1(d1_out)
        # d1_map = F.interpolate(d1_map, size=(label.shape[1], label.shape[2]), mode='bilinear')
        # d1_pre = torch.argmax(d1_map, 1)
        # CM1 = self.cal_confuse_matrix(d1_pre, label, normalize=True)
        # d1 = self.CWA1(d1_out, CM1) + self.MSA0(x_unpool)
        
        # out = self.classifier(d1)       

        # return [out, d1_map, d2_map, d3_map, d4_map], [CM1, CM2, CM3, CM4]
        
        d4_in = e4
        d4_out = self.decoder4(d4_in)                       
        d4_map = self.Sideout4(d4_out)
        d4_map = F.interpolate(d4_map, size=(label.shape[1], label.shape[2]), mode='bilinear')
        d4_pre = torch.argmax(d4_map, 1)
        CM4 = self.cal_confuse_matrix(d4_pre, label, normalize=True)
        d4 = d4_out + e3

        d3_in = d4
        d3_out = self.decoder3(d3_in)                       
        d3_map = self.Sideout3(d3_out)
        d3_map = F.interpolate(d3_map, size=(label.shape[1], label.shape[2]), mode='bilinear')
        d3_pre = torch.argmax(d3_map, 1)
        CM3 = self.cal_confuse_matrix(d3_pre, label, normalize=True)
        d3 = d3_out + e2

        d2_in = d3
        d2_out = self.decoder2(d2_in)                       
        d2_map = self.Sideout2(d2_out)
        d2_map = F.interpolate(d2_map, size=(label.shape[1], label.shape[2]), mode='bilinear')
        d2_pre = torch.argmax(d2_map, 1)
        CM2 = self.cal_confuse_matrix(d2_pre, label, normalize=True)
        d2 = d2_out + e1

        d1_in = d2
        d1_out = self.decoder1(d1_in)                       
        d1_map = self.Sideout1(d1_out)
        d1_map = F.interpolate(d1_map, size=(label.shape[1], label.shape[2]), mode='bilinear')
        d1_pre = torch.argmax(d1_map, 1)
        CM1 = self.cal_confuse_matrix(d1_pre, label, normalize=True)
        d1 = d1_out + x_unpool
        
        out = self.classifier(d1)       

        return [out, d1_map, d2_map, d3_map, d4_map], [CM1, CM2, CM3, CM4]
    
    def cal_confuse_matrix(self, predict, label, normalize=False):
        pre_pos_list = [] #predict等于各个类的下标数组
        label_pos_list = [] #label等于各个类的下标数组
        confuse_matrix = torch.zeros([6,6]).float().cuda()
        # label = label[:,0,:,:]
        for pre_class in range(6):
            pos_index = (predict == pre_class)
            pre_pos_list.append(pos_index)
        for label_class in range(6):
            pos_index = (label == label_class)
            label_pos_list.append(pos_index)
        
        for pre_class in range(6):
            for label_class in range(6):
                pos_index = pre_pos_list[pre_class]*label_pos_list[label_class]
                confuse_matrix[pre_class][label_class] = (pos_index.sum())
        if normalize == True:
            confuse_matrix = confuse_matrix/torch.max(confuse_matrix)
        return  confuse_matrix

    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


