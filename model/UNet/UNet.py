import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from collections import Counter
import torch.nn.functional as F
class UNet(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)           # 16*512*512
        maxpool1 = self.maxpool(conv1)       # 16*256*256
        
        conv2 = self.conv2(maxpool1)         # 32*256*256
        maxpool2 = self.maxpool(conv2)       # 32*128*128

        conv3 = self.conv3(maxpool2)         # 64*128*128
        maxpool3 = self.maxpool(conv3)       # 64*64*64

        conv4 = self.conv4(maxpool3)         # 128*64*64
        maxpool4 = self.maxpool(conv4)       # 128*32*32

        center = self.center(maxpool4)       # 256*32*32
        up4 = self.up_concat4(center,conv4)  # 128*64*64
        up3 = self.up_concat3(up4,conv3)     # 64*128*128
        up2 = self.up_concat2(up3,conv2)     # 32*256*256
        up1 = self.up_concat1(up2,conv1)     # 16*512*512

        final = self.final(up1)

        return final


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size+(n_concat-2)*out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                 nn.UpsamplingBilinear2d(scale_factor=2),
                 nn.Conv2d(in_size, out_size, 1))
           
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)

### initalize the module
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def UNet_LOSS(input, target):
    n, c, h, w = input.size()
    # assert(max(target) == 1)
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, c, -1)
    target = torch.unsqueeze(target, 1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()
    pos_index = (target_t > 0)
    neg_index = (target_t == 0)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    weight = torch.Tensor(target_t.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num

    weight[pos_index] = neg_num*1.0 / sum_num
    weight[neg_index] = pos_num*1.0 / sum_num
    weight = weight/np.max(weight)

    weights_dict = Counter(weight[0])
    Cross_Entropy = 0
    for key in weights_dict:
        CE_pos_index = (weight == key)
        print(CE_pos_index.sum(), key)
        CE_pos_index = CE_pos_index.astype(bool)
        Cal_Pixel = log_p[:,:,CE_pos_index[0]]
        Cal_target = target_t[:,CE_pos_index[0]]
        
        cross_Entropy0 = nn.CrossEntropyLoss()(Cal_Pixel, Cal_target)
        cross_Entropy0 *= key
        Cross_Entropy += cross_Entropy0


    return Cross_Entropy