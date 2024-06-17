import torch
import torch.nn as nn
from model.HyperSpectral.Base_Network import get_mlp
from model.HyperSpectral.Basic_Operation import *
from model.Self_Module.Layer_operations import *
from model.HyperSpectral.Auto_Multiply.Backbone import SP_2D
from model.HyperSpectral.Auto_Multiply.Act_Layer import Conv1D_Act, Conv2D_Act

class AutoContrast_2D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()

        print('Using AutoContrast_2D model in AutoContrast_2D')
        self.backbone = SP_2D(1, out_channels) 
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
        ipt_c_re = input_channels
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
        
        scale_num = 3
        self.conv1D1_1 = Conv1D_Act(scale_num, ratio, 7, 3)
        self.conv1D1_2 = Conv1D_Act(scale_num, ratio, 7, 3)
        self.conv1D2_1 = Conv1D_Act(scale_num, ratio, 7, 3)
        self.conv1D2_2 = Conv1D_Act(scale_num, ratio, 7, 3)
        self.conv1D3_1 = Conv1D_Act(scale_num, ratio, 7, 3)
        self.conv1D3_2 = Conv1D_Act(scale_num, ratio, 7, 3)
        self.conv1D4_1 = Conv1D_Act(scale_num, ratio, 7, 3)
        self.conv1D4_2 = Conv1D_Act(scale_num, ratio, 7, 3)
        self.conv1D5_1 = Conv1D_Act(scale_num, ratio, 7, 3)
        self.conv1D5_2 = Conv1D_Act(scale_num, ratio, 7, 3)
        self.conv1D_list1 = [self.conv1D1_1, self.conv1D2_1, self.conv1D3_1, self.conv1D4_1, self.conv1D5_1]
        self.conv1D_list2 = [self.conv1D1_2, self.conv1D2_2, self.conv1D3_2, self.conv1D4_2, self.conv1D5_2]

        self.conv2D1_1 = Conv2D_Act(scale_num, ratio, 7, 3)
        self.conv2D1_2 = Conv2D_Act(scale_num, ratio, 7, 3)
        self.conv2D2_1 = Conv2D_Act(scale_num, ratio, 7, 3)
        self.conv2D2_2 = Conv2D_Act(scale_num, ratio, 7, 3)
        self.conv2D3_1 = Conv2D_Act(scale_num, ratio, 7, 3)
        self.conv2D3_2 = Conv2D_Act(scale_num, ratio, 7, 3)
        self.conv2D4_1 = Conv2D_Act(scale_num, ratio, 7, 3)
        self.conv2D4_2 = Conv2D_Act(scale_num, ratio, 7, 3)
        self.conv2D5_1 = Conv2D_Act(scale_num, ratio, 7, 3)
        self.conv2D5_2 = Conv2D_Act(scale_num, ratio, 7, 3)
        self.conv2D_list1 = [self.conv2D1_1, self.conv2D2_1, self.conv2D3_1, self.conv2D4_1, self.conv2D5_1]
        self.conv2D_list2 = [self.conv2D1_2, self.conv2D2_2, self.conv2D3_2, self.conv2D4_2, self.conv2D5_2]

        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.out_fc = get_mlp(2, [16, out_channels])
        self.fuse_para = nn.Parameter(torch.ones([15]))
        self.GAP2D = torch.nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        b, c, h, w = x.size()

        '''
        Auto Feature Extract Separate
        '''
        # codes, MS_Feat = self.model_AutoFE(x)

        '''
        MultiScale Feature(Resample Spectral)
        '''
        MS_Spectral = MS_spectral(center_pixel(x), rsp_rate_list=[1, 2 ,4])
        # MS_Feat = MS_spectral_FeatureExtract(MS_Spectral, self.mlp_list1, self.mlp_list2, samesize=True)
        MS_Feat = MS_spectral_FeatureExtract(MS_Spectral, self.conv1D_list1, self.conv2D_list1, samesize=True)
        # MS_Feat = torch.cat([x for x in MS_Feat], 1)

        '''
        MultiScale Feature(Interpolate Spectral)
        '''
        # MS_Feat = MultiScale_CT(center_pixel(x), self.act_list1, self.act_list2, scale_list=[1], samesize=True)
        # MS_Feat = torch.cat([x for x in MS_Feat], 1)

        '''
        MultiScale Feature Triple
        '''
        # Feat_Triple = compute_auto_mulply_triple(center_pixel(x), self.mlp1_1, self.mlp1_2, self.mlp1_3)
        # Feat_Triple = F.interpolate(Feat_Triple, size=(70, 70, 70), mode='trilinear')

        '''
        Local & Global Feature
        '''
        # pt_ct = center_pixel(x)
        # MS_pt_ct = MS_spectral(pt_ct, rsp_rate_list=[1])
        # local_feat = compute_local_feature(MS_pt_ct[0], samesize=True)

        # global_feat = MS_spectral_FeatureExtract(MS_pt_ct, self.mlp_list1, self.mlp_list2, samesize=True)
        # global_feat = torch.cat([x for x in global_feat], 1)
        # feat = torch.cat([local_feat, global_feat], 1)

        x1 = self.backbone.SP_conv_1(MS_Feat)
        x2 = self.backbone.SP_conv_2(x1)
        x3 = self.backbone.SP_conv_3(x2)
        x4 = self.backbone.SP_conv_4(x3)
        x5 = self.backbone.SP_conv_5(x4)

        SP5_GAP = self.GAP2D(x5).view(b, -1)
        out = self.out_fc(SP5_GAP)

        return [out], [SP5_GAP]
