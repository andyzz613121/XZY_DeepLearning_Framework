
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.HyperSpectral.Base_Network import *
from model.HyperSpectral.Basic_Operation import *
from model.Self_Module.Layer_operations import *
from model.HyperSpectral.SISP_2Branch.SISP_2Branch import SISP_2Branch

class SP_2D_MultiScale(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        # SP_2D_MultiScale Net
        print('SP_2D: Using SP_2D_MultiScale model')
        self.SP_conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SP_conv_5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
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

class SP_2DDilaConv_MultiScale(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(SP_2DDilaConv_MultiScale, self).__init__()
        # SP 2D Net
        print('SP_2D: Using SP_2DDilaConv model')
        dilate_rate = 8
        self.SP_conv_1 = DilaConv(input_channels, 16, 1)
        self.SP_conv_2 = DilaConv(16, 16, 2)
        self.SP_conv_3 = DilaConv(16, 16, 4)
        self.SP_conv_4 = DilaConv(16, 16, 8)
        self.SP_conv_5 = DilaConv(16, 16, 16)
        #///////////////////////////////////////////////////////////////////////
        # SP sideout mlps 光谱
        self.SP_Sideout1 = get_mlp(2, [16, out_channels])
        self.SP_Sideout2 = get_mlp(2, [16, out_channels])
        self.SP_Sideout3 = get_mlp(2, [16, out_channels])
        self.SP_Sideout4 = get_mlp(2, [16, out_channels])
        self.SP_Sideout5 = get_mlp(2, [16, out_channels])

class SISP_2Branch_MultiScale(SISP_2Branch):
    def __init__(self, in_channel, out_channel):
        super(SISP_2Branch_MultiScale, self).__init__(in_channel, out_channel)
        print('Using SISP_2Branch_MultiScale')
        #///////////////////////////////////////////////////////////////////////
        self.Net2D = SP_2D_MultiScale(4, out_channel)
        # self.Net2D = SP_2DDilaConv_MultiScale(1, out_channel)
        #///////////////////////////////////////////////////////////////////////
        # Cascade
        # rate = math.ceil(in_channel/16)
        # ipt_c_re = int(rate * 16)
        ipt_c_re = in_channel
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
        self.mlp1_1 = get_mlp(3, [2, 4, 4])
        self.mlp1_2 = get_mlp(3, [2, 4, 4])
        self.mlp2_1 = get_mlp(3, [2, 4, 4])
        self.mlp2_2 = get_mlp(3, [2, 4, 4])
        self.mlp3_1 = get_mlp(3, [2, 4, 4])
        self.mlp3_2 = get_mlp(3, [2, 4, 4])
        self.mlp4_1 = get_mlp(3, [2, 4, 4])
        self.mlp4_2 = get_mlp(3, [2, 4, 4])
        self.mlp5_1 = get_mlp(3, [2, 4, 4])
        self.mlp5_2 = get_mlp(3, [2, 4, 4])

        ratio = 4
        # self.mlp1_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio, ipt_c_re])
        # self.mlp1_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio, ipt_c_re])
        # # self.mlp1_1 = get_mlp(2, [ipt_c_re, ipt_c_re])
        # # self.mlp1_2 = get_mlp(2, [ipt_c_re, ipt_c_re])
        # # self.mlp2_1 = get_mlp(2, [int(ipt_c_re/2), int(ipt_c_re/2)*ratio, int(ipt_c_re/2)])
        # # self.mlp2_2 = get_mlp(2, [int(ipt_c_re/2), int(ipt_c_re/2)*ratio, int(ipt_c_re/2)])
        # # self.mlp3_1 = get_mlp(2, [int(ipt_c_re/4), int(ipt_c_re/4)*ratio, int(ipt_c_re/4)])
        # # self.mlp3_2 = get_mlp(2, [int(ipt_c_re/4), int(ipt_c_re/4)*ratio, int(ipt_c_re/4)])
        # # self.mlp4_1 = get_mlp(2, [int(ipt_c_re/8), int(ipt_c_re/8)*ratio, int(ipt_c_re/8)])
        # # self.mlp4_2 = get_mlp(2, [int(ipt_c_re/8), int(ipt_c_re/8)*ratio, int(ipt_c_re/8)])
        # # self.mlp5_1 = get_mlp(2, [int(ipt_c_re/16), int(ipt_c_re/16)*ratio, int(ipt_c_re/16)])
        # # self.mlp5_2 = get_mlp(2, [int(ipt_c_re/16), int(ipt_c_re/16)*ratio, int(ipt_c_re/16)])
        # self.mlp2_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp2_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp3_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp3_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp4_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp4_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp5_1 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        # self.mlp5_2 = get_mlp(2, [ipt_c_re, ipt_c_re*ratio])
        self.mlp_list1 = [self.mlp1_1, self.mlp2_1, self.mlp3_1, self.mlp4_1, self.mlp5_1]
        self.mlp_list2 = [self.mlp1_2, self.mlp2_2, self.mlp3_2, self.mlp4_2, self.mlp5_2]
        # # Global trans prob
        # unit_mat = torch.eye(out_channel, out_channel).unsqueeze(0)
        # self.transprob_glob = unit_mat.repeat(10, 1, 1).cuda()
        # self.class_num = out_channel

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight.data)
        #         if m.bias != None:
        #             m.bias.data.fill_(0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight.data)
        #         if m.bias != None:
        #             m.bias.data.fill_(0)

    def MultiScale_SP(self, SP_img, scale_list=[1, 2, 4, 8, 16]):
        '''
        Usage:
                First Compute SP, then compute MultiScale SP
        Input: 
                SP_img(step\MLP\Ratio......, B, 1, H, W) 
        Output:
                SP_MultiScale(A list, SP_img with same position, different scale)  [[B, 1, H, W], [B, 1, H/2, W/2]...]
        '''
        b, _, h, w = SP_img.size()
        
        # 计算5层多尺度级联
        SP_MultiScale = []
        for scale in scale_list:
            SP_scale = F.interpolate(SP_img, size=(int((h)/scale), int((w)/scale)), mode='bilinear')
            SP_MultiScale.append(SP_scale)
        
        return SP_MultiScale

    def MultiScale_SP_samesize(self, SP_img, scale_list=[1, 2, 4, 8, 16]):
        '''
        Usage:
                First Compute SP, then compute MultiScale SP
        Input: 
                SP_img(step\MLP\Ratio......, B, 1, H, W) 
        Output:
                SP_MultiScale(A list, SP_img with same position, different scale)  [[B, 1, H, W], [B, 1, H/2, W/2]...]
        '''
        b, _, h, w = SP_img.size()
        
        # 计算5层多尺度级联
        SP_MultiScale = []
        for scale in scale_list:
            SP_scale = F.interpolate(SP_img, size=(int((h)/scale), int((w)/scale)), mode='bilinear')
            SP_scale = F.interpolate(SP_scale, size=(int((h)), int((w))), mode='bilinear')
            SP_MultiScale.append(SP_scale)
        
        return SP_MultiScale

    def MultiScale_CT(self, vector_ct, scale_list=[1, 2, 4, 8, 16]):
        '''
        Usage:
                First Compute MultiScale vector_ct, then compute SP of MultiScale vector_ct
        Input: 
                vector_ct(center pixel of img)
        Output:
                SP_MultiScale(A list, SP_img with same position, different scale)  [[B, 1, H, W], [B, 1, H/2, W/2]...]
        '''
        b, l = vector_ct.size()
        vector_ct = vector_ct.view(b, 1, l)

        # 计算5层多尺度级联
        SP_MultiScale = []
        for scale in scale_list:
            vector_re = F.interpolate(vector_ct, size=(int(l/scale)), mode='linear', align_corners=False)
            SP_singlescale = compute_ratio_withstep(vector_re.view(b, -1))
            SP_MultiScale.append(SP_singlescale)
            # print(vector_ct.shape, vector_re.shape, SP_singlescale.shape)
        return SP_MultiScale

    def MultiScale_CT_mlp(self, vector_ct, act_list1, act_list2, scale_list=[1, 2, 4, 8, 16], samesize=False):
        '''
        Usage:
                First Compute MultiScale vector_ct, then compute SP of MultiScale vector_ct
        Input: 
                vector_ct(center pixel of img)
        Output:
                SP_MultiScale(A list, SP_img with same position, different scale)  [[B, 1, H, W], [B, 1, H/2, W/2]...]
        '''
        b, l = vector_ct.size()
        vector_ct = vector_ct.view(b, 1, l)

        # 计算5层多尺度级联
        SP_MultiScale = []
        layer = 0
        for scale in scale_list:
            vector_re = F.interpolate(vector_ct, size=(int(l/scale)), mode='linear', align_corners=False)
            # SP_singlescale = compute_auto_mulply(vector_re, act_list1[layer], act_list2[layer], 0)
            SP_singlescale = compute_pixelwise_relation(vector_re.view(b, -1), act_list1[layer])
            # SP_singlescale = compute_ratio_withstep(vector_re.view(b, -1))
            if samesize == True:
                SP_singlescale = F.interpolate(SP_singlescale, size=(l, l), mode='bilinear')
            SP_MultiScale.append(SP_singlescale)
            layer += 1
            # print(vector_ct.shape, vector_re.shape, SP_singlescale.shape)
        return SP_MultiScale

    def Cascade_SP(self, SP_img, scale_list=[1, 2, 4, 8, 16]):
        '''
        Compute Cascade image for SP image
        Input: 
                SP_img(step\MLP\Ratio......, B, 1, H, W) 
        Output:
                SP_Cascade(A list, SP_img with same scale, different position)  [[B, 1, 0:n0, 0:n0], [B, 1, n0:n1, 0:n0]...]
        '''
        b, _, h, w = SP_img.size()

        # 重采样图像大小，能被16整除
        SP_img = SP_img.view(b, 1, h*w)
        h_rate = math.ceil((h)/16)
        w_rate = math.ceil((w)/16)
        l_re = h_rate * w_rate * 16 * 16
        SPimg_re =  F.interpolate(SP_img, size=(l_re), mode='linear', align_corners=False)

        # 计算5层多尺度级联
        SP_cascade = []
        for scale in scale_list:
            img_split = torch.split(SPimg_re, int(l_re/split_para), dim=2)
            SP_cascade.append(SP_scale)
        
        return SP_cascade

    def Cascade_CT(self, vector, act_list1, act_list2, cascade_level=[1, 2, 4, 8, 16]):
        '''
        Compute Cascade image for vector
        Input: 
                SP_img(step\MLP\Ratio......, B, 1, H, W) 
        Output:
                SP_Cascade(A list, SP_img of center vector at different position)
        
        E.g.: Input: 
                    Scale_level: n;   ct_vector: [B, L]
             Output: 
                    [[B, 1, L, L],
                     [B, 2, L/2, L/2],
                     [B, 4, L/4, L/4]
                     .....]

        '''
        b, l = vector.size()
        # 计算波段长度并重采样, 5层级联，因此除以16
        rate = math.ceil(l/16)
        l_re = rate * 16

        vector = vector.view(b, 1, l)
        vector_re = F.interpolate(vector, size=(l_re), mode='linear', align_corners=False)
        
        SP_cascade = []
        layer = 0
        for split_para in cascade_level:
            img_split = torch.split(vector_re, int(l_re/split_para), dim=2)
            img_split = torch.cat([x for x in img_split], 1)
            # print(img_split.shape, act_list1[layer])
            img_split = compute_spatt_mlp(img_split, act_list1[layer], act_list2[layer])
            SP_cascade.append(img_split)
            layer += 1

        return SP_cascade

#///////////////////////////////////////////////////////////////////////

    # _MultiScale
    def forward(self, x):
        b, c, h, w = x.size()
        
        x_3d = torch.unsqueeze(x, 1)
        # SP_CT = compute_multi_features_mlp(center_pixel(x), self.mlp_list1, self.mlp_list2)
        # SP_CT = compute_ratio(center_pixel(x))
        # SP_MultiScale = self.MultiScale_SP_samesize(SP_CT, scale_list=[1, 2, 4, 8, 16])
        # SP_MultiScale = self.MultiScale_SP(SP_CT, scale_list=[1, 2, 4, 8, 16])
        # SP_MultiScale = torch.cat([x for x in SP_MultiScale], 1)
        SP_MultiScale = self.MultiScale_CT_mlp(center_pixel(x), self.mlp_list1, self.mlp_list2, scale_list=[1], samesize=True)
        
        # SP_MultiScale = compute_pixelwise_relation(center_pixel(x), self.mlp1_1)
        # SP_MultiScale = self.MultiScale_SP(SP_MultiScale, scale_list=[1, 2, 4, 8, 16])
        SP_MultiScale = torch.cat([x for x in SP_MultiScale], 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.Net3D.l1_3D(x_3d)
        SA1_2D = self.Net3D.SA_CPR1(SA1)
        SA1_sideout = sideout2d(SA1_2D, self.Net3D.SA_Sideout1)
        
        SP1 = self.Net2D.SP_conv_1(SP_MultiScale)
        SP1_sideout = sideout2d(SP1, self.Net2D.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.Net3D.l2_3D(SA1)
        SA2_2D = self.Net3D.SA_CPR2(SA2)
        SA2_sideout = sideout2d(SA2_2D, self.Net3D.SA_Sideout2)
        
        # SP_MultiScale[1] = F.interpolate(SP_MultiScale[1], size=(SP1.shape[2], SP1.shape[3]), mode='bilinear')
        # SP2 = self.Net2D.SP_conv_2(torch.cat([SP1, SP_MultiScale[1]], 1))
        # SP2 = self.Net2D.SP_conv_2(SP1+SP_MultiScale[1])
        # print(SP_MultiScale[1].shape)
        SP2 = self.Net2D.SP_conv_2(SP1)
        SP2_sideout = sideout2d(SP2, self.Net2D.SP_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.Net3D.l3_3D(SA2)
        SA3_2D = self.Net3D.SA_CPR3(SA3)
        SA3_sideout = sideout2d(SA3_2D, self.Net3D.SA_Sideout3)
    
        # SP_MultiScale[2] = F.interpolate(SP_MultiScale[2], size=(SP2.shape[2], SP2.shape[3]), mode='bilinear')
        # SP3 = self.Net2D.SP_conv_3(torch.cat([SP2, SP_MultiScale[2]], 1))
        # SP3 = self.Net2D.SP_conv_3(SP2+SP_MultiScale[2])
        SP3 = self.Net2D.SP_conv_3(SP2)
        SP3_sideout = sideout2d(SP3, self.Net2D.SP_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.Net3D.l4_2D(SA3_2D)
        SA4_sideout = sideout2d(SA4_2D, self.Net3D.SA_Sideout4)

        # SP_MultiScale[3] = F.interpolate(SP_MultiScale[3], size=(SP3.shape[2], SP3.shape[3]), mode='bilinear')
        # SP4 = self.Net2D.SP_conv_4(torch.cat([SP3, SP_MultiScale[3]], 1))
        # SP4 = self.Net2D.SP_conv_4(SP3+SP_MultiScale[3])
        SP4 = self.Net2D.SP_conv_4(SP3)
        SP4_sideout = sideout2d(SP4, self.Net2D.SP_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.Net3D.l5_2D(SA4_2D)
        SA5_sideout = sideout2d(SA5_2D, self.Net3D.SA_Sideout5)

        # SP_MultiScale[4] = F.interpolate(SP_MultiScale[4], size=(SP4.shape[2], SP4.shape[3]), mode='bilinear')
        # SP5 = self.Net2D.SP_conv_5(torch.cat([SP4, SP_MultiScale[4]], 1))
        # SP5 = self.Net2D.SP_conv_5(SP4+SP_MultiScale[4])
        SP5 = self.Net2D.SP_conv_5(SP4)
        SP5_sideout = sideout2d(SP5, self.Net2D.SP_Sideout5)
        
        # Total_fuse = self.prob_norm(self.fuse_para[0], self.fuse_para[1], SA1_sideout, SP1_sideout) + \
        #              self.prob_norm(self.fuse_para[2], self.fuse_para[3], SA2_sideout, SP2_sideout) + \
        #              self.prob_norm(self.fuse_para[4], self.fuse_para[5], SA3_sideout, SP3_sideout) + \
        #              self.prob_norm(self.fuse_para[6], self.fuse_para[7], SA4_sideout, SP4_sideout) + \
        #              self.prob_norm(self.fuse_para[8], self.fuse_para[9], SA5_sideout, SP5_sideout)

        # Total_fuse = self.fuse_para[0]*SA1_sideout + self.fuse_para[1]*SA2_sideout + self.fuse_para[2]*SA3_sideout+\
        #              self.fuse_para[3]*SA4_sideout + self.fuse_para[4]*SA5_sideout+ self.fuse_para[9]*SP5_sideout
                     
        # return [Total_fuse, SA1_sideout, SA2_sideout, SA3_sideout, SA4_sideout, SA5_sideout,SP5_sideout]

        Total_fuse = self.fuse_para[0]*SA1_sideout + self.fuse_para[1]*SA2_sideout + self.fuse_para[2]*SA3_sideout+\
                     self.fuse_para[3]*SA4_sideout + self.fuse_para[4]*SA5_sideout+\
                     self.fuse_para[5]*SP1_sideout + self.fuse_para[6]*SP2_sideout + self.fuse_para[7]*SP3_sideout+\
                     self.fuse_para[8]*SP4_sideout + self.fuse_para[9]*SP5_sideout
                     
        return [Total_fuse, SA1_sideout, SA2_sideout, SA3_sideout, SA4_sideout, SA5_sideout, 
                            SP1_sideout, SP2_sideout, SP3_sideout, SP4_sideout, SP5_sideout]

        # fuse_loss1 = SA1_sideout + SP1_sideout
        # fuse_loss2 = SA2_sideout + SP2_sideout
        # fuse_loss3 = SA3_sideout + SP3_sideout
        # fuse_loss4 = SA4_sideout + SP4_sideout
        # fuse_loss5 = SA5_sideout + SP5_sideout
        # Total_fuse = self.fuse_para[0]*SA1_sideout + self.fuse_para[1]*SA2_sideout + self.fuse_para[2]*SA3_sideout+\
        #              self.fuse_para[3]*SA4_sideout + self.fuse_para[4]*SA5_sideout+\
        #              self.fuse_para[5]*SP1_sideout + self.fuse_para[6]*SP2_sideout + self.fuse_para[7]*SP3_sideout+\
        #              self.fuse_para[8]*SP4_sideout + self.fuse_para[9]*SP5_sideout+\
        #              self.fuse_para[10]*fuse_loss1 + self.fuse_para[11]*fuse_loss2 + self.fuse_para[12]*fuse_loss3+\
        #              self.fuse_para[13]*fuse_loss4 + self.fuse_para[14]*fuse_loss5

        # return [Total_fuse, SA1_sideout, SA2_sideout, SA3_sideout, SA4_sideout, SA5_sideout, 
        #                     SP1_sideout, SP2_sideout, SP3_sideout, SP4_sideout, SP5_sideout,
        #                     fuse_loss1, fuse_loss2, fuse_loss3, fuse_loss4, fuse_loss5]

        # Total_fuse = self.fuse_para[3]*SA4_sideout + self.fuse_para[4]*SA5_sideout+\
        #              self.fuse_para[8]*SP4_sideout + self.fuse_para[9]*SP5_sideout+\
        #              self.fuse_para[13]*fuse_loss4 + self.fuse_para[14]*fuse_loss5
                    
        # return [Total_fuse, SA4_sideout, SA5_sideout, 
        #                     SP4_sideout, SP5_sideout,
        #                     fuse_loss4, fuse_loss5]

        '''
        transprob return
        '''

        # sideout_list = [SA1_sideout, SA2_sideout, SA3_sideout, SA4_sideout, SA5_sideout, 
        #                     SP1_sideout, SP2_sideout, SP3_sideout, SP4_sideout, SP5_sideout]

        # # pre_t = self.fuse_mlp(torch.cat([x for x in sideout_list], 1))
        # # print(self.transprob_glob.shape)
        # pre_t = torch.zeros(b, self.class_num).cuda()
        # for index in range(len(sideout_list)):
        #     outs = sideout_list[index].view(b, -1, 1)
        #     transprob = self.transprob_glob[index].view(1, self.class_num, self.class_num).repeat(b, 1, 1)
        #     # pre = torch.bmm(transprob, outs)[:,:,0]  #是否转置
        #     pre = (1 + transprob) * outs  #是否转置
        #     pre = pre.sum(1)
        #     pre_t = pre_t + pre
        
        # return [pre_t, SA1_sideout, SA2_sideout, SA3_sideout, SA4_sideout, SA5_sideout, 
        #                     SP1_sideout, SP2_sideout, SP3_sideout, SP4_sideout, SP5_sideout]

        '''
        Random return
        '''
        # out_num = 2
        # sideouts = [SA1_sideout, SA2_sideout, SA3_sideout, SA4_sideout, SA5_sideout, 
        #                     SP1_sideout, SP2_sideout, SP3_sideout, SP4_sideout, SP5_sideout]
        # sideouts = torch.cat([x.unsqueeze(1) for x in sideouts], 1)
        # random_weight = torch.randint(0, 2, size=(out_num, 1, 10, 1)).repeat(1, b, 1, SA1_sideout.shape[1]).cuda()
        # total_weight = torch.ones([1, 10, 1]).repeat(b, 1, SA1_sideout.shape[1]).cuda()
        # random_weight[0] = total_weight

        # Return_Outs = []
        # for out_time in range(out_num):
        #     one_index = (random_weight[out_time] == 1)
        #     if one_index.sum() == 0:
        #         # random_weight[out_time] = 1
        #         continue
        #     else:
        #         sideouts_w = sideouts * random_weight[out_time]
        #         sideout_sum = sideouts_w.sum(1)
        #         norm_max, _ = torch.max(torch.abs(sideout_sum), 1)

        #         # print('444', sideout_sum, random_weight[out_time], )
        #         # print('111', sideout_sum, norm_max)
        #         sideout_norm = sideout_sum/norm_max.unsqueeze(1)
        #         # print('222', sideout_norm)
        #         Return_Outs.append(sideout_norm)
            
        # return [x for x in Return_Outs]


    # # Cascade
    # def forward(self, x):
    #     b, c, h, w = x.size()
        
    #     x_3d = torch.unsqueeze(x, 1)
    #     SP_Cascade = self.Cascade_CT(center_pixel(x), self.mlp_list1, self.mlp_list2, [1, 2, 4, 8, 16])
    #     #///////////////////////////////////////////////////////////////////////
    #     # Layer 1
    #     SA1 = self.Net3D.l1_3D(x_3d)
    #     SA1_2D = self.Net3D.SA_CPR1(SA1)
    #     SA1_sideout = sideout2d(SA1_2D, self.Net3D.SA_Sideout1)

    #     SP1 = self.Net2D.SP_conv_1(SP_Cascade[0])
    #     SP1_sideout = sideout2d(SP1, self.Net2D.SP_Sideout1)
    #     #///////////////////////////////////////////////////////////////////////
    #     # Layer 2
    #     SA2 = self.Net3D.l2_3D(SA1)
    #     SA2_2D = self.Net3D.SA_CPR2(SA2)
    #     SA2_sideout = sideout2d(SA2_2D, self.Net3D.SA_Sideout2)
        
    #     SP2 = self.Net2D.SP_conv_2(torch.cat([SP1, SP_Cascade[1]], 1))
    #     SP2_sideout = sideout2d(SP2, self.Net2D.SP_Sideout2)
    #     #///////////////////////////////////////////////////////////////////////
    #     # Layer 3       
    #     SA3 = self.Net3D.l3_3D(SA2)
    #     SA3_2D = self.Net3D.SA_CPR3(SA3)
    #     SA3_sideout = sideout2d(SA3_2D, self.Net3D.SA_Sideout3)
    
    #     SP3 = self.Net2D.SP_conv_3(torch.cat([SP2, SP_Cascade[2]], 1))
    #     SP3_sideout = sideout2d(SP3, self.Net2D.SP_Sideout3)
    #     #///////////////////////////////////////////////////////////////////////
    #     # Layer 4
    #     SA4_2D = self.Net3D.l4_2D(SA3_2D)
    #     SA4_sideout = sideout2d(SA4_2D, self.Net3D.SA_Sideout4)

    #     SP4 = self.Net2D.SP_conv_4(torch.cat([SP3, SP_Cascade[3]], 1))
    #     SP4_sideout = sideout2d(SP4, self.Net2D.SP_Sideout4)
    #     #///////////////////////////////////////////////////////////////////////
    #     # Layer 5
    #     SA5_2D = self.Net3D.l5_2D(SA4_2D)
    #     SA5_sideout = sideout2d(SA5_2D, self.Net3D.SA_Sideout5)

    #     SP5 = self.Net2D.SP_conv_5(torch.cat([SP4, SP_Cascade[4]], 1))
    #     SP5_sideout = sideout2d(SP5, self.Net2D.SP_Sideout5)

    #     Total_fuse = self.fuse_para[0]*SA1_sideout + self.fuse_para[1]*SA2_sideout + self.fuse_para[2]*SA3_sideout+\
    #                  self.fuse_para[3]*SA4_sideout + self.fuse_para[4]*SA5_sideout+\
    #                  self.fuse_para[5]*SP1_sideout + self.fuse_para[6]*SP2_sideout + self.fuse_para[7]*SP3_sideout+\
    #                  self.fuse_para[8]*SP4_sideout + self.fuse_para[9]*SP5_sideout
                     
    #     return [Total_fuse, SA1_sideout, SA2_sideout, SA3_sideout, SA4_sideout, SA5_sideout, 
    #                         SP1_sideout, SP2_sideout, SP3_sideout, SP4_sideout, SP5_sideout]
    
