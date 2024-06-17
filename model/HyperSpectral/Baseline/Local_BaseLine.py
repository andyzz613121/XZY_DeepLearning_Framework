import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HyperSpectral.Base_Network import *
from model.HyperSpectral.Basic_Operation import *
from model.Self_Module.Layer_operations import *
from model.HyperSpectral.FeatureExtract.AutoFeature import AutoFeature
from model.Self_Module.ASPP import ASPP


class SP_2DASPP(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP_2DASPP Net
        print('SP_2DASPP: Using SP_2DASPP model')
        self.SP_conv_1 = nn.Sequential(
            ASPP(input_channels, 16, [1, 2, 4]),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_2 = nn.Sequential(
            ASPP(16, 16, [1, 2, 4]),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_3 = nn.Sequential(
            ASPP(16, 16, [1, 2, 4]),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_4 = nn.Sequential(
            ASPP(16, 16, [1, 2, 4]),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_5 = nn.Sequential(
            ASPP(16, 16, [1, 2, 4]),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        #///////////////////////////////////////////////////////////////////////

class SP_2D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP 2D Net
        print('SP_2D: Using SP_2D model in Local_BaseLine')
        self.SP_conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout1 = get_mlp(2, [16, out_channels])
        self.SP_Sideout2 = get_mlp(2, [16, out_channels])
        self.SP_Sideout3 = get_mlp(2, [16, out_channels])
        self.SP_Sideout4 = get_mlp(2, [16, out_channels])
        self.SP_Sideout5 = get_mlp(2, [16, out_channels])

    def forward(self, x):
        # SP_CT = compute_ratio_withstep(center_pixel(x))
        pt_ct = center_pixel(x)
        pt_ct = compute_local_feature(pt_ct)

        x1 = self.SP_conv_1(pt_ct)
        x2 = self.SP_conv_2(x1)
        x3 = self.SP_conv_3(x2)
        x4 = self.SP_conv_4(x3)
        x5 = self.SP_conv_5(x4)

        out = sideout2d(x5, self.SP_Sideout1)
        return [out], []

class SP_2D_3Layer(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP 2D Net
        print('SP_2D: Using SP_2D_3Layer model')
        self.SP_conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout = get_mlp(2, [16, out_channels])


    def forward(self, x):
        # SP_CT = compute_ratio_withstep(center_pixel(x))
        pt_ct = center_pixel(x)
        pt_ct = compute_local_feature(pt_ct)

        x1 = self.SP_conv_1(pt_ct)
        x2 = self.SP_conv_2(x1)
        x3 = self.SP_conv_3(x2)

        out = sideout2d(x3, self.SP_Sideout)
        return [out], []


class SP_2D_AutoContrast(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP 2D Net
        print('Using SP_2D_AutoContrast model')
        # self.backbone = SP_2DDilaConv(1, out_channels) 
        # self.backbone = SP_2DASPP(2, out_channels)
        # self.backbone1 = SP_2DASPP(1, out_channels)
        # self.backbone2 = SP_2DASPP(1, out_channels)
        self.backbone3 = SP_2DASPP(3, out_channels)
        # self.backbone = SA_3D(1, out_channels)
        # self.model_AutoFE = AutoFeature(input_channels)
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
        
        self.conv1D1_1 = Conv1D_Act(1, ratio, 3)
        self.conv1D1_2 = Conv1D_Act(1, ratio, 3)
        self.conv1D2_1 = Conv1D_Act(1, ratio, 3)
        self.conv1D2_2 = Conv1D_Act(1, ratio, 3)
        self.conv1D3_1 = Conv1D_Act(1, ratio, 3)
        self.conv1D3_2 = Conv1D_Act(1, ratio, 3)
        self.conv1D4_1 = Conv1D_Act(1, ratio, 3)
        self.conv1D4_2 = Conv1D_Act(1, ratio, 3)
        self.conv1D5_1 = Conv1D_Act(1, ratio, 3)
        self.conv1D5_2 = Conv1D_Act(1, ratio, 3)
        self.conv1D_list1 = [self.conv1D1_1, self.conv1D2_1, self.conv1D3_1, self.conv1D4_1, self.conv1D5_1]
        self.conv1D_list2 = [self.conv1D1_2, self.conv1D2_2, self.conv1D3_2, self.conv1D4_2, self.conv1D5_2]

        self.conv2D1_1 = Conv2D_Act(1, ratio, 3)
        self.conv2D1_2 = Conv2D_Act(1, ratio, 3)
        self.conv2D2_1 = Conv2D_Act(1, ratio, 3)
        self.conv2D2_2 = Conv2D_Act(1, ratio, 3)
        self.conv2D3_1 = Conv2D_Act(1, ratio, 3)
        self.conv2D3_2 = Conv2D_Act(1, ratio, 3)
        self.conv2D4_1 = Conv2D_Act(1, ratio, 3)
        self.conv2D4_2 = Conv2D_Act(1, ratio, 3)
        self.conv2D5_1 = Conv2D_Act(1, ratio, 3)
        self.conv2D5_2 = Conv2D_Act(1, ratio, 3)
        self.conv2D_list1 = [self.conv2D1_1, self.conv2D2_1, self.conv2D3_1, self.conv2D4_1, self.conv2D5_1]
        self.conv2D_list2 = [self.conv2D1_2, self.conv2D2_2, self.conv2D3_2, self.conv2D4_2, self.conv2D5_2]

        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout5 = get_mlp(2, [16, out_channels])
        self.SP_Sideout4 = get_mlp(2, [16, out_channels])
        self.SP_Sideout3 = get_mlp(2, [16, out_channels])
        self.CVT3D_2D = CVT3D_2D_SA(16, 16, 70)
        self.fuse_para = nn.Parameter(torch.ones([15]))
        self.GAP2D = torch.nn.AdaptiveAvgPool2d(1)

        self.loss_para = nn.Parameter(torch.ones([2]))

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

        x1 = self.backbone1.SP_conv_1(MS_Feat[0])
        x2 = self.backbone1.SP_conv_2(x1)
        x3 = self.backbone1.SP_conv_3(x2)
        x4 = self.backbone1.SP_conv_4(x3)
        x5 = self.backbone1.SP_conv_5(x4)

        xx1 = self.backbone2.SP_conv_1(MS_Feat[1])
        xx2 = self.backbone2.SP_conv_2(xx1)
        xx3 = self.backbone2.SP_conv_3(xx2)
        xx4 = self.backbone2.SP_conv_4(xx3)
        xx5 = self.backbone2.SP_conv_5(xx4)

        xxx1 = self.backbone3.SP_conv_1(MS_Feat[2])
        xxx2 = self.backbone3.SP_conv_2(xxx1)
        xxx3 = self.backbone3.SP_conv_3(xxx2)
        xxx4 = self.backbone3.SP_conv_4(xxx3)
        xxx5 = self.backbone3.SP_conv_5(xxx4)

        # x1 = self.SP_conv_1(SP_MultiScale[0])
        # SP_MultiScale[1] = F.interpolate(SP_MultiScale[1], size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        # x2 = self.SP_conv_2(torch.cat([x1, SP_MultiScale[1]], 1))
        # SP_MultiScale[2] = F.interpolate(SP_MultiScale[2], size=(x2.shape[2], x2.shape[3]), mode='bilinear')
        # x3 = self.SP_conv_3(torch.cat([x2, SP_MultiScale[2]], 1))
        # SP_MultiScale[3] = F.interpolate(SP_MultiScale[3], size=(x3.shape[2], x3.shape[3]), mode='bilinear')
        # x4 = self.SP_conv_4(torch.cat([x3, SP_MultiScale[3]], 1))
        # SP_MultiScale[4] = F.interpolate(SP_MultiScale[4], size=(x4.shape[2], x4.shape[3]), mode='bilinear')
        # x5 = self.SP_conv_5(torch.cat([x4, SP_MultiScale[4]], 1))
    
        # out1 = sideout2d(x1, self.SP_Sideout1)
        # out2 = sideout2d(x2, self.SP_Sideout2)
        # out3 = sideout2d(x3, self.SP_Sideout3)
        # out4 = sideout2d(x4, self.SP_Sideout4)
        # out5 = sideout2d(x5, self.SP_Sideout5)
        # out5 = sideout3d(x5, self.SP_Sideout5, self.CVT3D_2D)

        SP1_GAP = self.GAP2D(x1).view(b, -1)
        SP2_GAP = self.GAP2D(x2).view(b, -1)
        SP3_GAP = self.GAP2D(x3).view(b, -1)
        SP4_GAP = self.GAP2D(x4).view(b, -1)
        # SP5_GAP = self.GAP2D(self.CVT3D_2D(x5)).view(b, -1)
        SP5_GAP = self.GAP2D(x5).view(b, -1)
        SP55_GAP = self.GAP2D(xx5).view(b, -1)
        SP555_GAP = self.GAP2D(xxx5).view(b, -1)

        out5 = self.SP_Sideout5(SP5_GAP)
        out55 = self.SP_Sideout4(SP55_GAP)
        out555 = self.SP_Sideout3(SP555_GAP)
        Total_fuse = self.fuse_para[0]*out5 + self.fuse_para[1]*out55 + self.fuse_para[2]*out555
        # return [Total_fuse, out1, out2, out3, out4, out5], [SP1_GAP, SP2_GAP, SP3_GAP, SP4_GAP, SP5_GAP]
        return [Total_fuse], [SP5_GAP, SP55_GAP, SP555_GAP]
        # return [out5], [codes]

class SP_1D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP 1D Net
        print('SP_1D: Using SP_1D model')
        self.SP_conv_1 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_2 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_4 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_5 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout1 = get_mlp(2, [16, out_channels])
        self.SP_Sideout2 = get_mlp(2, [16, out_channels])
        self.SP_Sideout3 = get_mlp(2, [16, out_channels])
        self.SP_Sideout4 = get_mlp(2, [16, out_channels])
        self.SP_Sideout5 = get_mlp(2, [16, out_channels])
    
    def forward(self, x):
        b, c, h, w = x.size()

        x_ct = center_pixel(x).view(b, -1, c)
        x1 = self.SP_conv_1(x_ct)
        x2 = self.SP_conv_2(x1)
        x3 = self.SP_conv_3(x2)
        x4 = self.SP_conv_4(x3)
        x5 = self.SP_conv_5(x4)
        
        out = sideout2d(x5.view(b, -1, c, 1), self.SP_Sideout1)
        return [out]

class SP_MLP(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP MLP Net
        print('SP_MLP: Using SP_MLP model')
        self.mlp = get_mlp(5, [input_channels*input_channels, int(input_channels*input_channels/2), int(input_channels*input_channels/4), int(input_channels*input_channels/8), out_channels])

        ratio = 2
        ipt_c_re = input_channels
        self.mlp1_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        self.mlp1_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp2_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp2_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp3_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp3_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        self.mlp2_1 = get_mlp(2, [int((ipt_c_re)/2+0.99), int((ipt_c_re)/2+0.99)*ratio])
        self.mlp2_2 = get_mlp(2, [int((ipt_c_re)/2+0.99), int((ipt_c_re)/2+0.99)*ratio])
        self.mlp3_1 = get_mlp(2, [int((ipt_c_re)/4+0.99), int((ipt_c_re)/4+0.99)*ratio])
        self.mlp3_2 = get_mlp(2, [int((ipt_c_re)/4+0.99), int((ipt_c_re)/4+0.99)*ratio])
        self.mlp_list1 = [self.mlp1_1, self.mlp2_1, self.mlp3_1]
        self.mlp_list2 = [self.mlp1_2, self.mlp2_2, self.mlp3_2]

        self.SP_Sideout1 = get_mlp(2, [16, out_channels])
        self.SP_Sideout2 = get_mlp(2, [16, out_channels])
        self.SP_Sideout3 = get_mlp(2, [16, out_channels])
        self.SP_Sideout4 = get_mlp(2, [16, out_channels])
        self.SP_Sideout5 = get_mlp(2, [16, out_channels])

    def forward(self, x):
        b, l, h, w = x.size()
        # x_ct = center_pixel(x)
        # # print(x_ct.shape)
        # return [self.mlp(x_ct)]
        MS_Feat = MultiScale_CT(center_pixel(x), self.mlp_list1, self.mlp_list2, scale_list=[1], samesize=True)
        MS_Feat = torch.cat([x for x in MS_Feat], 1).view(b, -1)
        
        return [self.mlp(MS_Feat)], []
        