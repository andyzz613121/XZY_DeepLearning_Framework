import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Self_Module.Conv4D import Conv4d

def sideout2d(feats, mlp):
    '''
    Input: feat: feature maps
           mlp: mlp used for classification
    '''
    b, c, h, w = feats.size()
    GAP2D = torch.nn.AdaptiveAvgPool2d(1)
    return mlp(GAP2D(feats).view(b, -1))

def sideout3d(feats, mlp, compress_conv):
    '''
    Input: feat: feature maps
           mlp: mlp used for classification
           compress_conv: convs for compress 3D->2D
    '''
    b, c, _, h, w = feats.size()
    GAP2D = torch.nn.AdaptiveAvgPool2d(1)
    return mlp(GAP2D(compress_conv(feats)).view(b, -1))

def sideout4d(feats, mlp, compress_conv):
    '''
    Input: feat: feature maps
           mlp: mlp used for classification
           compress_conv: convs for compress 4D->2D
    '''
    b, c, _, h, w = feats.size()
    GAP2D = torch.nn.AdaptiveAvgPool2d(1)
    return mlp(GAP2D(compress_conv(feats)).view(b, -1))

class CVT4D_2D_SA(nn.Module):
    '''
    用于压缩光谱维将4D变为2D
    当输入in_channels个4D特征图的时候，每个特征图深度为k时，对每一个4D特征图用一个深度为k的卷积核卷积为深度为1，即变为2D
    '''
    def __init__(self, in_channels, out_channels, k=(1, 1, 103, 103), bias=True):
        super(CVT4D_2D_SA, self).__init__()
        self.s1 = Conv4d(in_channels, out_channels, kernel_size=k, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.s1(input)
        out = out.squeeze(4).squeeze(4)
        return self.relu(out)

class CVT3D_2D_SA(nn.Module):
    '''
    用于压缩光谱维将3D变为2D
    当输入in_channels个3D特征图的时候，每个特征图深度为k时，对每一个3D特征图用一个深度为k的卷积核卷积为深度为1，即变为2D
    '''
    def __init__(self, in_channels, out_channels, k=49, bias=True):
        super(CVT3D_2D_SA, self).__init__()
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k,1,1), bias=False)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.s1(input)
        out = out.squeeze(2)
        out = self.relu(out)
        # out = self.bn(out)
        return out

class CVT3D_2D_SP(nn.Module):
    '''
    用于压缩空间维将3D变为2D
    当输入in_channels个3D特征图的时候，每个特征图深度为k时，对每一个3D特征图用一个深度为k的卷积核卷积为深度为1，即变为2D
    '''
    def __init__(self, in_channels, out_channels, h=49, bias=True):
        super(CVT3D_2D_SP, self).__init__()

        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1,h,h), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):

        out = self.s1(input)
        out = out.squeeze(3)
        return self.bn(self.relu(out)).squeeze(3)

class CVT2D_Channels(nn.Module):
    '''
    使用1*1卷积改变2D Feats的通道数
    '''
    def __init__(self, in_channels, out_channels):
        super(CVT2D_Channels, self).__init__()
        self.s1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.s1(input)
        return self.bn(self.relu(out))

class CVT1D_Channels(nn.Module):
    '''
    使用1*1卷积改变1D Feats的通道数
    '''
    def __init__(self, in_channels, out_channels):
        super(CVT1D_Channels, self).__init__()
        self.s1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.s1(input)
        return self.bn(self.relu(out))