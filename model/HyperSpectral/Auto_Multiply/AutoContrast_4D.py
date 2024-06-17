import torch
import torch.nn as nn
from model.HyperSpectral.Base_Network import get_mlp
from model.HyperSpectral.Basic_Operation import *
from model.Self_Module.Layer_operations import *
from model.HyperSpectral.Auto_Multiply.Backbone import SP_2D_ASPP, SP_4D
from model.HyperSpectral.Auto_Multiply.Act_Layer import Conv1D_Act, Conv2D_Act

class AutoContrast_4D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        print('Using AutoContrast_4D model in AutoContrast_4D')
        self.backbone = SP_4D(1, out_channels)
        #///////////////////////////////////////////////////////////////////////
        # self.mlp1_1 = get_mlp(3, [2, 2, 1])
        # self.mlp1_2 = get_mlp(3, [2, 2, 1])
        # self.mlp2_1 = get_mlp(3, [2, 2, 1])
        # self.mlp2_2 = get_mlp(3, [2, 2, 1])
        # self.mlp3_1 = get_mlp(3, [2, 2, 1])
        # self.mlp3_2 = get_mlp(3, [2, 2, 1])
        # self.mlp4_1 = get_mlp(3, [2, 2, 1])
        # self.mlp4_2 = get_mlp(3, [2, 2, 1])
        # self.mlp5_1 = get_mlp(3, [2, 2, 1])
        # self.mlp5_2 = get_mlp(3, [2, 2, 1])

        ratio = 4
        scale_num = 1
        kernel_size = 3
        padding = 1
        ipt_c_re = input_channels
        self.scale_list = [1]
        print('ratio: %s, scale_list: %s' %(str(ratio), str(self.scale_list)))
        # self.mlp1_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp1_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp1_3 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # # self.mlp2_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # # self.mlp2_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # # self.mlp3_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # # self.mlp3_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp2_1 = get_mlp(2, [int((ipt_c_re)/2+0.99), int((ipt_c_re)/2+0.99)*ratio])
        # self.mlp2_2 = get_mlp(2, [int((ipt_c_re)/2+0.99), int((ipt_c_re)/2+0.99)*ratio])
        # self.mlp2_3 = get_mlp(2, [int((ipt_c_re)/2+0.99), int((ipt_c_re)/2+0.99)*ratio])
        # self.mlp3_1 = get_mlp(2, [int((ipt_c_re)/4+0.99), int((ipt_c_re)/4+0.99)*ratio])
        # self.mlp3_2 = get_mlp(2, [int((ipt_c_re)/4+0.99), int((ipt_c_re)/4+0.99)*ratio])
        # self.mlp3_3 = get_mlp(2, [int((ipt_c_re)/4+0.99), int((ipt_c_re)/4+0.99)*ratio])
        # self.mlp_list1 = [self.mlp1_1, self.mlp2_1, self.mlp3_1]
        # self.mlp_list2 = [self.mlp1_2, self.mlp2_2, self.mlp3_2]
        # self.mlp_list3 = [self.mlp1_3, self.mlp2_3, self.mlp3_3]
        
        # self.conv1D1_1 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        # self.conv1D1_2 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        # self.conv1D2_1 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        # self.conv1D2_2 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        # self.conv1D3_1 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        # self.conv1D3_2 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        # self.conv1D4_1 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        # self.conv1D4_2 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        # self.conv1D5_1 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        # self.conv1D5_2 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        # self.conv1D_list1 = [self.conv1D1_1, self.conv1D2_1, self.conv1D3_1, self.conv1D4_1, self.conv1D5_1]
        # self.conv1D_list2 = [self.conv1D1_2, self.conv1D2_2, self.conv1D3_2, self.conv1D4_2, self.conv1D5_2]

        self.conv2D1_1 = Conv2D_Act(scale_num, ratio, kernel_size, padding)
        self.conv2D1_2 = Conv2D_Act(scale_num, ratio, kernel_size, padding)
        self.conv2D2_1 = Conv2D_Act(scale_num, ratio, kernel_size, padding)
        self.conv2D2_2 = Conv2D_Act(scale_num, ratio, kernel_size, padding)
        self.conv2D3_1 = Conv2D_Act(scale_num, ratio, kernel_size, padding)
        self.conv2D3_2 = Conv2D_Act(scale_num, ratio, kernel_size, padding)
        self.conv2D4_1 = Conv2D_Act(scale_num, ratio, kernel_size, padding)
        self.conv2D4_2 = Conv2D_Act(scale_num, ratio, kernel_size, padding)
        self.conv2D5_1 = Conv2D_Act(scale_num, ratio, kernel_size, padding)
        self.conv2D5_2 = Conv2D_Act(scale_num, ratio, kernel_size, padding)
        self.conv2D_list1 = [self.conv2D1_1, self.conv2D2_1, self.conv2D3_1, self.conv2D4_1, self.conv2D5_1]
        self.conv2D_list2 = [self.conv2D1_2, self.conv2D2_2, self.conv2D3_2, self.conv2D4_2, self.conv2D5_2]

        self.bandwise_mlp1 = get_mlp(2, [ratio*2, 1])
        self.bandwise_mlp2 = get_mlp(2, [ratio*2, 1])
        self.bandwise_mlp3 = get_mlp(2, [ratio*2, 1])
        self.bandwise_mlp_list = [self.bandwise_mlp1, self.bandwise_mlp2, self.bandwise_mlp3]
        #///////////////////////////////////////////////////////////////////////
        self.out_fc = get_mlp(2, [16, out_channels])
        self.cvt4D_2D = CVT4D_2D_SA(16, 16, k=(1, 1, input_channels, input_channels))

        self.fuse_para = nn.Parameter(torch.ones([15]))
        self.GAP2D = torch.nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        b, c, h, w = x.size()

        '''
        MultiScale Feature(Resample Spectral) 4D
        '''

        MS_Spectral = MS_spectral_entireimage(center_img(x, 5), rsp_rate_list=self.scale_list)
        MS_Feat = MS_spectral_FeatureExtract_bandwise4D(MS_Spectral, self.conv2D_list1, self.conv2D_list2, self.bandwise_mlp_list, act = '2D', samesize=True)
        MS_Feat = torch.cat([x for x in MS_Feat], 1)

        x1 = self.backbone.SP_conv_1(MS_Feat)
        x2 = self.backbone.SP_conv_2(x1)
        x3 = self.backbone.SP_conv_3(x2)
        # x4 = self.backbone.SP_conv_4(x3)
        # x5 = self.backbone.SP_conv_5(x4)
        # print(x3)
        x3_2d = self.cvt4D_2D(x3)
        SP3_GAP = self.GAP2D(x3_2d).view(b, -1)

        out = self.out_fc(SP3_GAP)
        return [out], [SP3_GAP]

