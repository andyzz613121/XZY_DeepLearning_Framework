
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HyperSpectral.Base_Network import *
from model.HyperSpectral.Basic_Operation import *
from model.Self_Module.Layer_operations import *

class SISP_2Branch(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SISP_2Branch, self).__init__()
        print('Using SISP_2Branch')
        #///////////////////////////////////////////////////////////////////////
        # Nets
        self.Net3D = SA3D2D(in_channel, out_channel)
        # self.Net2D = SP_2DDilaConv(1, out_channel)
        self.Net2D = SP_2D(3, out_channel)
        #///////////////////////////////////////////////////////////////////////
        # CPR Para
        self.CPR_SA1 = CVT3D_2D_SA(16, 16, in_channel)
        self.CPR_SA2 = CVT3D_2D_SA(16, 16, in_channel)
        self.CPR_SA3 = CVT3D_2D_SA(16, 16, in_channel)
        self.CPR_SA4 = CVT3D_2D_SA(16, 16, in_channel)
        self.CPR_SA5 = CVT3D_2D_SA(16, 16, in_channel)
        #///////////////////////////////////////////////////////////////////////
        # Fuse Para
        self.fuse_para = nn.Parameter(torch.ones([15]))
        self.mlp1_1 = get_mlp(2, [in_channel, in_channel])
        self.mlp1_2 = get_mlp(2, [in_channel, in_channel])
        self.mlp2_1 = get_mlp(2, [in_channel, in_channel])
        self.mlp2_2 = get_mlp(2, [in_channel, in_channel])
        self.mlp3_1 = get_mlp(2, [in_channel, in_channel])
        self.mlp3_2 = get_mlp(2, [in_channel, in_channel])
        self.mlp4_1 = get_mlp(2, [in_channel, in_channel])
        self.mlp4_2 = get_mlp(2, [in_channel, in_channel])
        self.mlp5_1 = get_mlp(2, [in_channel, in_channel])
        self.mlp5_2 = get_mlp(2, [in_channel, in_channel])
        self.mlp_list1 = [self.mlp1_1, self.mlp2_1, self.mlp3_1, self.mlp4_1, self.mlp5_1]
        self.mlp_list2 = [self.mlp1_2, self.mlp2_2, self.mlp3_2, self.mlp4_2, self.mlp5_2]
        #///////////////////////////////////////////////////////////////////////
        # Global trans prob
        unit_mat = torch.eye(out_channel, out_channel).unsqueeze(0)
        self.transprob_glob = unit_mat.repeat(10, 1, 1).cuda()

        self.fuse_mlp = get_mlp(2, [10*out_channel, out_channel])
        #///////////////////////////////////////////////////////////////////////
        # Init Para
        self.softmax = nn.Softmax(dim=-1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)

    def prob_norm(self, a, b, result_a, result_b):
        '''
            Output: result_a*(W1) + result_b*(1-W1)
            W1 = (a^2/(a^2+b^2))  W2 = (b^2/(a^2+b^2))
        '''
        W1 = a*a/(a*a+b*b)
        W2 = b*b/(a*a+b*b)
        return (W1*result_a + W2*result_b)

    def pos_pool2d(self, feats, masks, mlp):
        b, c, h, w = feats.size()
        PosNum = masks.view(b, 1, -1).sum(2)          #   激活的像元个数
        ActFeat_sum = (masks * feats).view(b, c, -1).sum(2)  #   激活的像元的光谱总和
        ActMean = ActFeat_sum / PosNum               #   激活的像元的光谱均值     b, 1, h, w

        return mlp(ActMean.view(b, -1))
#///////////////////////////////////////////////////////////////////////

    def forward(self, x):
        b, c, h, w = x.size()
        
        x_3d = torch.unsqueeze(x, 1)
        SP_CT = compute_ratio_withstep(center_pixel(x))
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.Net3D.l1_3D(x_3d)
        SA1_2D = self.Net3D.SA_CPR1(SA1)
        SA1_sideout = sideout2d(SA1_2D, self.Net3D.SA_Sideout1)

        SP1 = self.Net2D.SP_conv_1(SP_CT)
        SP1_sideout = sideout2d(SP1, self.Net2D.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.Net3D.l2_3D(SA1)
        SA2_2D = self.Net3D.SA_CPR2(SA2)
        SA2_sideout = sideout2d(SA2_2D, self.Net3D.SA_Sideout2)

        SP2 = self.Net2D.SP_conv_2(SP1)
        SP2_sideout = sideout2d(SP2, self.Net2D.SP_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.Net3D.l3_3D(SA2)
        SA3_2D = self.Net3D.SA_CPR3(SA3)
        SA3_sideout = sideout2d(SA3_2D, self.Net3D.SA_Sideout3)

        SP3 = self.Net2D.SP_conv_3(SP2)
        SP3_sideout = sideout2d(SP3, self.Net2D.SP_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.Net3D.l4_2D(SA3_2D)
        SA4_sideout = sideout2d(SA4_2D, self.Net3D.SA_Sideout4)

        SP4 = self.Net2D.SP_conv_4(SP3)
        SP4_sideout = sideout2d(SP4, self.Net2D.SP_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.Net3D.l5_2D(SA4_2D)
        SA5_sideout = sideout2d(SA5_2D, self.Net3D.SA_Sideout5)

        SP5 = self.Net2D.SP_conv_5(SP4)
        SP5_sideout = sideout2d(SP5, self.Net2D.SP_Sideout5)
        
        # Total_fuse = self.prob_norm(self.fuse_para[0], self.fuse_para[1], SA1_sideout, SP1_sideout) + \
        #              self.prob_norm(self.fuse_para[2], self.fuse_para[3], SA2_sideout, SP2_sideout) + \
        #              self.prob_norm(self.fuse_para[4], self.fuse_para[5], SA3_sideout, SP3_sideout) + \
        #              self.prob_norm(self.fuse_para[6], self.fuse_para[7], SA4_sideout, SP4_sideout) + \
        #              self.prob_norm(self.fuse_para[8], self.fuse_para[9], SA5_sideout, SP5_sideout)

        Total_fuse = self.fuse_para[0]*SA1_sideout + self.fuse_para[1]*SA2_sideout + self.fuse_para[2]*SA3_sideout+\
                     self.fuse_para[3]*SA4_sideout + self.fuse_para[4]*SA5_sideout+\
                     self.fuse_para[5]*SP1_sideout + self.fuse_para[6]*SP2_sideout + self.fuse_para[7]*SP3_sideout+\
                     self.fuse_para[8]*SP4_sideout + self.fuse_para[9]*SP5_sideout

        return [Total_fuse, SA1_sideout, SA2_sideout, SA3_sideout, SA4_sideout, SA5_sideout, 
                            SP1_sideout, SP2_sideout, SP3_sideout, SP4_sideout, SP5_sideout]

        
        # out_num = 10
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