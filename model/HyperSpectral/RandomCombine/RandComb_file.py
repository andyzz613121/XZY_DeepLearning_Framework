import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

import random
import torch
import torch.nn as nn
import numpy as np
from model.HyperSpectral.Base_Network import get_mlp
from model.HyperSpectral.Basic_Operation import *
from model.HyperSpectral.Auto_Multiply.Backbone import *
from model.HyperSpectral.Baseline import HybridSN
class RandComb_model(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        print('Using RandComb_model model')
        # self.backbone = SP_1D(input_channels*10, out_channels)
        self.backbone = HybridSN.HybridSN(100, input_channels, out_channels)
        # self.backbone = SP_2D(2, out_channels)
        # self.backbone = SP_4D(1, out_channels)
        
        #///////////////////////////////////////////////////////////////////////
        self.out_fc = get_mlp(2, [16, out_channels])
        
        self.fuse_para = nn.Parameter(torch.ones([15]))
        # self.GAP2D = torch.nn.AdaptiveAvgPool2d(1)
        self.GAP1D = torch.nn.AdaptiveAvgPool1d(1)
        self.rand_list = torch.nn.Parameter(self.generate_randlist(input_channels, input_channels*10).cuda(), requires_grad=False)
        # self.rand_list = self.generate_randlist_group(input_channels, input_channels*10, 10).cuda()

    def generate_randlist(self, band_num, rand_num):
        print('generate_randlist')
        total = torch.tensor(range(0, band_num)).unsqueeze(0)
        for i in range(rand_num-1):
            tmp_list = list(range(0, band_num))
            tmp_list = torch.tensor(random.sample(tmp_list, int(len(tmp_list)))).unsqueeze(0)
            total = torch.cat([total, tmp_list], 0)
        return total

    def generate_randlist_group(self, band_num, rand_num, group_num):
        print('generate_randlist_group')
        total = torch.tensor(range(0, band_num)).unsqueeze(0)
        group_idx = []
        group_size = band_num / group_num
        for i in range(group_num):
            begin = int(group_size*i)
            end = int(group_size*(i+1))  if int(group_size*(i+1)) < band_num else band_num
            group_idx.append(np.arange(begin, end, 1))

        for i in range(rand_num-1):
            tmp_list = group_idx
            tmp_list = random.sample(tmp_list, int(len(group_idx)))
            tmp_list = np.concatenate([x for x in tmp_list])
            tmp_list = torch.tensor(np.array(tmp_list)).unsqueeze(0)
            total = torch.cat([total, tmp_list], 0)
        # print(total.shape, total)
        return total

    def generate_randimg(self, vect, rand_list):
        '''
            Input:  vect(b, c)
                    rand_list(N, c)
            Output: 
                    randimg(b, N, c)
        '''
        b, c = vect.size()
        l, _ = rand_list.size()
        
        vect = vect.unsqueeze(1)
        vect = vect.repeat(1, l, 1)
        rand_list = rand_list.unsqueeze(0).repeat(b, 1, 1)

        randimg = vect.gather(2, rand_list)
        return randimg
    
    def generate_randcube(self, cube, rand_list):
        '''
            将Input的cube中每一个像素按照rand_list里面的一种随机排列方式进行重排
            Input:  cube(b, c, h, w)
                    rand_list(N, c)
            Output: 
                    randimg(b, N, c, h, w)
        '''
        b, c, h, w = cube.size()
        cube = cube.view(b, c, -1)
        recube_list = []
        for px_idx in range(cube.shape[2]):
            randimg_tmp = self.generate_randimg(cube[:,:,px_idx], rand_list).unsqueeze(3)
            recube_list.append(randimg_tmp)
        recube = torch.cat([x for x in recube_list], 3).view(b, -1, c, h, w)
        return recube

    def forward(self, x):
        
        b, c, h, w = x.size()

        # rand_img = compute_local_feature(pt_ct)
        # rand_img = self.generate_randimg(pt_ct, self.rand_list)
        # rand_img = compute_local_feature(rand_img.view(-1, c))
        # rand_img = rand_img.view(b, -1, rand_img.shape[2], rand_img.shape[2])
        #------------------------------------------------------------------------------
        # For Rand Select first K
        # Band_num = 50
        # Feat_num = 50
        # pt_ct = center_pixel(x)
        # rand_img = self.generate_randimg(pt_ct, self.rand_list).unsqueeze(1)
        # rand_img = rand_img[:,:,10:10+Feat_num, 0:Band_num]
        # pt_img = compute_local_feature(pt_ct)
        # fuse_img = torch.cat([rand_img, pt_img], 1)
        #------------------------------------------------------------------------------
        # For Rand Select 1D
        # pt_ct = center_pixel(x)
        # rand_img = self.generate_randimg(pt_ct, self.rand_list).unsqueeze(1)
        # rand_img = rand_img.view(b, -1, c)
        #------------------------------------------------------------------------------
        # For 4D Conv
        # wind_size = 5
        # img_ct = center_img(x, wind_size).view(b, c, -1).permute(0, 2, 1)
        # rand_img = compute_local_feature_batch(img_ct)
        # rand_img = rand_img.view(b, -1, 2*(wind_size)+1, 2*(wind_size)+1, rand_img.shape[2], rand_img.shape[3])
        #------------------------------------------------------------------------------
        # For 3D Conv
        cube = center_img(x, 5)
        rand_img = self.generate_randcube(cube, self.rand_list)
        out = self.backbone(rand_img[:, 0:100, :,:,:])
        # cube = center_img(x, 5)
        # out = self.backbone(cube)
        #------------------------------------------------------------------------------
        # x1 = self.backbone.SP_conv_1(rand_img)
        # x2 = self.backbone.SP_conv_2(x1)
        # x3 = self.backbone.SP_conv_3(x2)
        # x4 = self.backbone.SP_conv_4(x3)
        # x5 = self.backbone.SP_conv_5(x4)

        # SP5_GAP = self.GAP1D(x5).view(b, -1)
        # out = self.out_fc(SP5_GAP)

        return [out]

if __name__ == '__main__':
    aa = RandComb_model(11, 11)
    # bb = torch.tensor([5, 6, 7, 8, 9]).unsqueeze(0)
    # bb = torch.cat([bb, bb, bb, bb, bb], 0)
    # list1 = aa.generate_randlist(5)
    # cc = bb.gather(1, list1)
    # print(list1)
    # print(bb.shape)
    # print(cc)