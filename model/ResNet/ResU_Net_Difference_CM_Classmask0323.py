from locale import normalize
import re
from turtle import forward
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from model.ResNet.ResNet import resnet101, resnet50, resnet34
from model.Self_Module.CARB import CARB_Block

class PAM(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(PAM, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out

class PAM_Maps(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(PAM_Maps, self).__init__()
        self.conv_b = nn.Conv2d(6, 6, 1)
        self.conv_c = nn.Conv2d(6, 6, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, map1, map2, feature_map):
        batch_size, _, height, width = map1.size()
        #还要不要conv_b?
        feat_b = self.conv_b(map1).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(map2).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(feature_map).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + feature_map

        return out

class PAM_Maps_Diff(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(PAM_Maps_Diff, self).__init__()
        self.conv_b = nn.Conv2d(6, 6, 1)
        self.conv_c = nn.Conv2d(6, 6, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, map1, map2, feature_map):
        batch_size, _, height, width = map1.size()
        #还要不要conv_b?
        diff = map1 - map2
        feat_b = self.conv_b(diff).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(diff).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))

        feat_d = self.conv_d(feature_map).view(batch_size, -1, height * width)
        print(feat_d.shape)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        print(feat_e.shape)
        out = self.alpha * feat_e + feature_map

        return out

class Global_Error_Attention(nn.Module):
    def __init__(self, out_channels):
        super(Global_Error_Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.proj = nn.Conv2d(6, out_channels, 1)
    def forward(self, map, cm, feature_map):
        batch_size, _, height, width = map.size()
        map = self.softmax(map).view(batch_size, -1, height * width).permute(0, 2, 1)
        cm = cm.sum(0)
        #这步是Precision或Recall中的其中一个
        cm = cm.view(1, 1, 6).permute(0, 2, 1)
        cm = cm.repeat(batch_size, 1, 1)
        
        category_weight = torch.bmm(map, cm)
        category_weight = category_weight.permute(0, 2, 1).view(batch_size, -1, height, width)
        out = self.alpha * feature_map * category_weight + feature_map
        #这个地方可以用Conv(1, out_channels)增加维度
        print('category_weight', category_weight.shape)
        print('feature_map', feature_map)
        print('feature_map * category_weight', feature_map * category_weight)

        return out

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


class ResU_Net_Diff_Global_Attention(nn.Module):
    def __init__(self, input_channels, num_classes, pre_train=True):
        super(ResU_Net_Diff_Global_Attention, self).__init__()
        
        self.resnet = resnet34(input_channels, True)

        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)
        
        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 3, padding=1)
        )

        self.Sideout5 = CARB_Block(512, 6)
        self.Sideout4 = CARB_Block(256, 6)
        self.Sideout3 = CARB_Block(128, 6)
        self.Sideout2 = CARB_Block(64, 6)
        self.Sideout1 = CARB_Block(64, 6)

        self.PAM_Map4 = PAM_Maps_Diff(256)
        self.PAM_Map3 = PAM_Maps_Diff(128)
        self.PAM_Map2 = PAM_Maps_Diff(64)
        self.PAM_Map1 = PAM_Maps_Diff(64)

        self.GEA4 = Global_Error_Attention(256)
        self.GEA3 = Global_Error_Attention(128)
        self.GEA2 = Global_Error_Attention(64)
        self.GEA1 = Global_Error_Attention(64)

    def forward(self, x, cm):
        b, _, h, w = x.size()

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      #64

        e1 = self.encoder1(x)           #256
        e2 = self.encoder2(e1)          #512
        e3 = self.encoder3(e2)          #1024
        e4 = self.encoder4(e3)          #2048

        d4 = self.decoder4(e4)
        d4_map = self.Sideout4(d4)
        d4_map_upsample = F.interpolate(d4_map, size=(h, w), mode='bilinear')
        d4 = self.GEA4(d4_map, cm, d4)

        d3 = self.decoder3(d4)
        d3_map = self.Sideout3(d3)
        d3_map_upsample = F.interpolate(d3_map, size=(h, w), mode='bilinear')
        d3 = self.GEA3(d3_map, cm, d3)

        d2 = self.decoder2(d3)
        d2_map = self.Sideout2(d2)
        d2_map_upsample = F.interpolate(d2_map, size=(h, w), mode='bilinear')
        d2 = self.GEA2(d2_map, cm, d2)

        d1 = self.decoder1(d2)                       
        d1_map = self.Sideout1(d1)
        d1_map_upsample = F.interpolate(d1_map, size=(h, w), mode='bilinear')
        d1 = self.GEA1(d1_map, cm, d1)

        out = self.classifier(d1)     

        return out, d1_map_upsample, d2_map_upsample, d3_map_upsample, d4_map_upsample
        
        
    
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
                # if pre_class != label_class:
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


