import torch.nn as nn
from model.Self_Module.ASPP import ASPP
from model.HyperSpectral.Basic_Operation import *
from model.Self_Module.Layer_operations import *
from model.Self_Module.Conv4D import Conv4d, Conv4d1, Conv4d_seperate2X3D
from model.HyperSpectral.Base_Network import get_mlp

from model.HyperSpectral.Base_Network import get_mlp
from model.HyperSpectral.Basic_Operation import *
from model.Self_Module.Layer_operations import *
from model.HyperSpectral.Auto_Multiply.Act_Layer import Conv1D_Act, Conv2D_Act
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

class SP_2D_ASPP(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP_2DASPP Net
        print('SP_2D_ASPP: Using SP_2D_ASPP model in Backbone.py')
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

class SP_2D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP 2D Net
        print('SP_2D: Using SP_2D model in Backbone.py')
        self.SP_conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
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

class SP_3D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP 2D Net
        print('SP_3D: Using SP_3D model in Backbone.py')
        self.SP_conv_1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_4 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_5 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_6 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        #///////////////////////////////////////////////////////////////////////
        self.out_fc = get_mlp(2, [16, out_channels])
        self.cvt3D_2D = CVT3D_2D_SA(16, 16, k=9)

        ratio = 4
        scale_num = 1
        kernel_size = 3
        padding = 1
        ipt_c_re = input_channels
        self.scale_list = [1]
        print('ratio: %s, scale_list: %s' %(str(ratio), str(self.scale_list)))
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

        self.GAP2D = torch.nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x):
        MS_Spectral = MS_spectral_entireimage(center_img(x, 1), rsp_rate_list=self.scale_list)
        MS_Feat = MS_spectral_FeatureExtract_bandwise4D(MS_Spectral, self.conv2D_list1, self.conv2D_list2, self.bandwise_mlp_list, act = '2D', samesize=True)
        x = torch.cat([x for x in MS_Feat], 1)

        b, c, h, w, H, W = x.size()

        a1 = x.view(b, -1, h*w, H, W)
        b1 = x.view(b, -1, h, w, H*W).permute(0, 1, 4, 2, 3)
        x1 = self.SP_conv_1(a1).view(b, -1, h, w, H, W)
        y1 = self.SP_conv_4(b1).permute(0, 1, 3, 4, 2).view(b, -1, h, w, H, W)
        xy1 = x1*y1

        a2 = xy1.view(b, -1, h*w, H, W)
        b2 = xy1.view(b, -1, h, w, H*W).permute(0, 1, 4, 2, 3)
        x2 = self.SP_conv_2(a2).view(b, -1, h, w, H, W)
        y2 = self.SP_conv_5(b2).permute(0, 1, 3, 4, 2).view(b, -1, h, w, H, W)
        xy2 = x2*y2

        a3 = xy2.view(b, -1, h*w, H, W)
        b3 = xy2.view(b, -1, h, w, H*W).permute(0, 1, 4, 2, 3)
        x3 = self.SP_conv_3(a3).view(b, -1, h, w, H, W)
        y3 = self.SP_conv_6(b3).permute(0, 1, 3, 4, 2).view(b, -1, h, w, H, W)
        # print(x3.shape, y3.shape)
        xy3 = (x3*y3).view(b, -1, h*w, H, W)
        
        # y2 = self.SP_conv_5(y1)
        # y3 = self.SP_conv_6(y2)

        Feat_gap = self.GAP2D(self.cvt3D_2D(xy3)).view(b, -1)
        # print(xy3.shape)
        out = self.out_fc(Feat_gap)
        
        return [out], [Feat_gap]

class SP_4D(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP 4D Net
        print('SP_4D: Using SP_4D model in Backbone.py')
        self.SP_conv_1 = nn.Sequential(
            # Conv4d(1, 16, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True),
            Conv4d1(1, 16),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_2 = nn.Sequential(
            # Conv4d(16, 16, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True),
            Conv4d1(16, 16),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_3 = nn.Sequential(
            # Conv4d(16, 16, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True),
            Conv4d1(16, 16),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_4 = nn.Sequential(
            # Conv4d(16, 16, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True),
            Conv4d1(16, 16),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_5 = nn.Sequential(
            # Conv4d(16, 16, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True),
            Conv4d1(16, 16),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        #///////////////////////////////////////////////////////////////////////
        self.out_fc = get_mlp(2, [16, out_channels])

    def forward(self, x):
        # SP_CT = compute_ratio_withstep(center_pixel(x))
        pt_ct = center_pixel(x)
        pt_ct = compute_local_feature(pt_ct)

        x1 = self.SP_conv_1(pt_ct)
        x2 = self.SP_conv_2(x1)
        x3 = self.SP_conv_3(x2)
        x4 = self.SP_conv_4(x3)
        x5 = self.SP_conv_5(x4)

        out = sideout2d(x5, self.out_fc)
        return [out], []