from locale import normalize
import re
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
        map1 = self.softmax(map1)
        map2 = self.softmax(map2)
        #还要不要conv_b?
        # feat_b = self.conv_b(map1).view(batch_size, -1, height * width).permute(0, 2, 1)
        # feat_c = self.conv_c(map2).view(batch_size, -1, height * width)
        feat_b = map1.view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = map2.view(batch_size, -1, height * width)
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
        diff = torch.abs(self.softmax(self.softmax(map1) - self.softmax(map2)))        
        feat_b = self.conv_b(diff).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(diff).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(feature_map).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + feature_map

        return out

    def forward1(self, map1, map2, feature_map):
        batch_size, channel_size, height, width = map1.size()
        
        similarity = torch.zeros((batch_size, height*width, height*width)).cuda()
        # for channel in range(channel_size):
        #     feat_b = map1[:,channel,:,:].view(batch_size, -1, height * width, 1)
        #     feat_c = map2[:,channel,:,:].view(batch_size, -1, height * width, 1)
        #     print('feat_bc1', channel, feat_b, feat_c)
        #     feat_b = feat_b.repeat(1, 1, 1, height * width)
        #     feat_b = feat_b.permute(0, 1, 3, 2)
        #     feat_c = feat_c.repeat(1, 1, 1, height * width)
        #     similarity_single = torch.abs((feat_b - feat_c)).sum(1)
        #     similarity += similarity_single
        # similarity = torch.sqrt(similarity)
        feat_b = map1.view(batch_size, -1, height * width, 1)
        feat_c = map2.view(batch_size, -1, height * width)
        feat_b = feat_b.repeat(1, 1, 1, height * width)
        feat_b = feat_b.permute(0, 1, 3, 2)
        for n in range(height*width):
            similarity_single = torch.abs((feat_b[:,:,:,n] - feat_c)).sum(1)
            similarity[:,:,n] = similarity_single
        
        attention_s = self.softmax(similarity)
        feat_d = self.conv_d(feature_map).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + feature_map
        print('out', out)
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

class ResU_Net_Difference_Attention(nn.Module):
    def __init__(self, input_channels, num_classes, pre_train=True):
        super(ResU_Net_Difference_Attention, self).__init__()
        
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

        self.MSA4 = Mean_Std_Attention(512)
        self.MSA3 = Mean_Std_Attention(256)
        self.MSA2 = Mean_Std_Attention(128)
        self.MSA1 = Mean_Std_Attention(64)
        self.MSA0 = Mean_Std_Attention(64)

        self.Sideout4 = CARB_Block(256, 6)
        self.Sideout3 = CARB_Block(128, 6)
        self.Sideout2 = CARB_Block(64, 6)
        self.Sideout1 = CARB_Block(64, 6)

        self.PAM_Map4 = PAM_Maps_Diff(256)
        self.PAM_Map3 = PAM_Maps_Diff(128)
        self.PAM_Map2 = PAM_Maps_Diff(64)
        self.PAM_Map1 = PAM_Maps_Diff(64)

    def forward(self, x):
        batch_size, channel_size, height, width = x.size()

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_unpool = x
        x = self.resnet.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + e3
        d4_map = self.Sideout4(d4)
        
        d3 = self.decoder3(d4) + e2
        d3_map = self.Sideout3(d3)
        d4_map = F.interpolate(d4_map, size=(d3_map.shape[2], d3_map.shape[3]), mode='bilinear')
        d3 = self.PAM_Map3(d3_map, d4_map, d3)
        
        d2 = self.decoder2(d3) + e1
        d2_map = self.Sideout2(d2)
        d3_map = F.interpolate(d3_map, size=(d2_map.shape[2], d2_map.shape[3]), mode='bilinear')       
        d2 = self.PAM_Map2(d2_map, d3_map, d2)

        d1 = self.decoder1(d2) + x_unpool
        d1_map = self.Sideout1(d1)
        d2_map = F.interpolate(d2_map, size=(d1_map.shape[2], d1_map.shape[3]), mode='bilinear')
        d1 = self.PAM_Map1(d1_map, d2_map, d1) 
        
        out = self.classifier(d1)

        d1_map = F.interpolate(d1_map, size=(height, width), mode='bilinear')
        d2_map = F.interpolate(d2_map, size=(height, width), mode='bilinear')
        d3_map = F.interpolate(d3_map, size=(height, width), mode='bilinear')
        d4_map = F.interpolate(d4_map, size=(height, width), mode='bilinear')
        return out, d1_map, d2_map, d3_map, d4_map
        
        
    
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


