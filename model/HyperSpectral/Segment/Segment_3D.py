from msilib.schema import Class
from re import S
import torch
import torch.nn as nn
from model.HyperSpectral.Base_Network import *
from model.HyperSpectral.Basic_Operation import *
from model.Self_Module.Layer_operations import *

class SA3D2D_seg(SA3D2D):
    def __init__(self, input_channels, out_channels, seg_num):
        super(SA3D2D_seg, self).__init__(input_channels, out_channels)

        self.SA_Sideout1 = get_mlp(2, [16*seg_num, out_channels])
        self.SA_Sideout2 = get_mlp(2, [16*seg_num, out_channels])
        self.SA_Sideout3 = get_mlp(2, [16*seg_num, out_channels])
        self.SA_Sideout4 = get_mlp(2, [16*seg_num, out_channels])
        self.SA_Sideout5 = get_mlp(2, [16*seg_num, out_channels])

class Segment_3D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Segment_3D, self).__init__()
        print('Using Segment_3D')
        self.class_num = out_channel
        #///////////////////////////////////////////////////////////////////////
        # Segment Para
        self.Seg_Num = 1
        #///////////////////////////////////////////////////////////////////////
        # Nets
        self.Net3D = SA3D2D_seg(in_channel, out_channel, self.Seg_Num)
        self.Net2D = SP_2D(1, out_channel)
        #///////////////////////////////////////////////////////////////////////
        # Segment Convs
        self.Seg_Conv1 = CVT2D_Channels(16, self.Seg_Num)
        self.Seg_Conv2 = CVT2D_Channels(16, self.Seg_Num)
        self.Seg_Conv3 = CVT2D_Channels(16, self.Seg_Num)
        self.Seg_Conv4 = CVT2D_Channels(16, self.Seg_Num)
        self.Seg_Conv5 = CVT2D_Channels(16, self.Seg_Num)
        #///////////////////////////////////////////////////////////////////////
        # CPR Para
        self.CPR_SA1 = CVT3D_2D_SA(16, 16, in_channel)
        self.CPR_SA2 = CVT3D_2D_SA(16, 16, in_channel)
        self.CPR_SA3 = CVT3D_2D_SA(16, 16, in_channel)
        #///////////////////////////////////////////////////////////////////////
        # Fuse Para
        self.fuse_para = nn.Parameter(torch.ones([10]))
        #///////////////////////////////////////////////////////////////////////
        # Global trans prob
        unit_mat = torch.eye(out_channel, out_channel).unsqueeze(0)
        self.transprob_glob = unit_mat.repeat(10, 1, 1).cuda()

        self.fuse_mlp = get_mlp(2, [10*out_channel, out_channel])
        #///////////////////////////////////////////////////////////////////////
        # Init Para
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)

    def gen_segmask(self, feats):
        feat_argmax = torch.argmax(feats, 1)
        # print(feat_argmax,feat_argmax.shape)
        # print(torch.nn.functional.one_hot(feat_argmax, self.Seg_Num),torch.nn.functional.one_hot(feat_argmax, self.Seg_Num).shape)
        return torch.nn.functional.one_hot(feat_argmax, self.Seg_Num).permute(0, 3, 1, 2)

    def seg_pool(self, feats, masks):
        '''
            Usage: 按照masks每个波段中的分割块进行池化，每个波段图中的分割块池化得到一个值
            Input: 
                   feats (待分类的特征图, b, c, h, w)
                   masks (分割的one-hot mask, b, self.Seg_Num, h, w)
            Output:
                   seg_pool(b, c*self.Seg_Num)
        '''
        b, c, h, w = feats.size()
        feats = feats.view(b, c, -1)
        masks = masks.view(b, self.Seg_Num, -1)
        means_list = []
        for i in range(self.Seg_Num):
            mask = masks[:, i, :]                       # b * (hw) 
            means_list_batch = [] 
            for batch in range(b):
                feat_mask_b = feats[batch]              # c * (hw) 

                mask_b = mask[batch]                    # (hw) 
                pos_index = (mask_b == 1)
                if pos_index.sum() != 0: 
                    feat_pos = feat_mask_b[:, pos_index]
                    # print('feat_pos', feat_pos.shape)
                    mean_b = torch.mean(feat_pos, 1)
                    # print(mean_b.shape)
                else:
                    mean_b = torch.zeros([c]).cuda()

                means_list_batch.append(mean_b.view(1, -1))

            means_batch = torch.cat([x for x in means_list_batch], 0)
            means_list.append(means_batch)

        seg_pool = torch.cat([x for x in means_list], 1)
        return seg_pool


#///////////////////////////////////////////////////////////////////////
    def forward1(self, x):
        b, c, h, w = x.size()

        # pt_center = center_pixel(x)
        # pt_img = compute_ratio_withstep(pt_center)
        pt_img = x.view(b, c, h*w)
        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.Net3D.l1_3D(x_3d)
        SA1_2D = self.Net3D.SA_CPR1(SA1)
        SA1_2D_mask = self.gen_segmask(self.Seg_Conv1(SA1_2D))
        SA1_seg_pool = self.seg_pool(SA1_2D, SA1_2D_mask)
        SA1_maps = self.Net3D.SA_Sideout1(SA1_seg_pool)          # SA的LULC maps
        # SA1_sideout = sideout2d(SA1_2D, self.Net3D.SA_Sideout1)
        
        # SP1 = self.Net2D.SP_conv_1(pt_img)
        # segmask_1 = self.gen_segmask(self.Seg_Conv1(SA1_2D))
        # SP1_sideout = self.Net2D.SP_Sideout1(SP1.permute(0, 2, 1).contiguous().view(b*h*w, -1))
        # print(SP1_sideout.shape, SP1_sideout.view(b, -1, h, w).shape, segmask_1.shape)
        # SP1_maps = segmask_1 * SP1_sideout.view(b, -1, h, w)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.Net3D.l2_3D(SA1)
        SA2_2D = self.Net3D.SA_CPR2(SA2)
        SA2_2D_mask = self.gen_segmask(self.Seg_Conv2(SA2_2D))
        SA2_seg_pool = self.seg_pool(SA2_2D, SA2_2D_mask)
        SA2_maps = self.Net3D.SA_Sideout2(SA2_seg_pool)          # SA的LULC maps
        # SA2_maps = self.CPR_SA2(SA2_2D)          # SA的LULC maps

        # SP2 = self.Net2D.SP_conv_2(SP1)
        # SP2_maps = self.CPR_SP2(SP2)             # SP的LULC maps
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.Net3D.l3_3D(SA2)
        SA3_2D = self.Net3D.SA_CPR3(SA3)
        SA3_2D_mask = self.gen_segmask(self.Seg_Conv3(SA3_2D))
        SA3_seg_pool = self.seg_pool(SA3_2D, SA3_2D_mask)
        SA3_maps = self.Net3D.SA_Sideout3(SA3_seg_pool)          # SA的LULC maps
        # SA3_maps = self.CPR_SA3(SA3_2D)          # SA的LULC maps

        # SP3 = self.Net2D.SP_conv_3(SP2)
        # SP3_maps = self.CPR_SP3(SP3)             # SP的LULC maps
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.Net3D.l4_2D(SA3_2D)
        SA4_2D_mask = self.gen_segmask(self.Seg_Conv4(SA4_2D))
        SA4_seg_pool = self.seg_pool(SA4_2D, SA4_2D_mask)
        SA4_maps = self.Net3D.SA_Sideout4(SA4_seg_pool)          # SA的LULC maps
        # SA4_maps = self.CPR_SA4(SA4_2D)          # SA的LULC maps

        # SP4 = self.Net2D.SP_conv_4(SP3)
        # SP4_maps = self.CPR_SP4(SP4)             # SP的LULC maps
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.Net3D.l5_2D(SA4_2D)
        # SA5_maps = self.Net3D.SA_Sideout5(local_pool(SA5_2D).view(b, -1))          # SA的LULC maps
        # SA5_maps = self.Net3D.SA_Sideout5(torch.nn.AdaptiveAvgPool2d(1)(local_pool(SA5_2D)).view(b, -1))          # SA的LULC maps
        SA5_2D_mask = self.gen_segmask(self.Seg_Conv5(SA5_2D))
        SA5_seg_pool = self.seg_pool(SA5_2D, SA5_2D_mask)
        SA5_maps = self.Net3D.SA_Sideout5(SA5_seg_pool)          # SA的LULC maps
        # print(SA5_maps)
        # SP5 = self.Net2D.SP_conv_5(SP4)
        # SP5_maps = self.CPR_SP5(SP5)             # SP的LULC maps
        #///////////////////////////////////////////////////////////////////////
        # # Reshape
        # SP1_maps = SP1_maps.view(b, -1, h, w)
        # SP2_maps = SP2_maps.view(b, -1, h, w)
        # SP3_maps = SP3_maps.view(b, -1, h, w)
        # SP4_maps = SP4_maps.view(b, -1, h, w)
        # SP5_maps = SP5_maps.view(b, -1, h, w)

        # print(SA1_maps.shape, SA2_maps.shape,SA3_maps.shape,SA4_maps.shape,SA5_maps.shape,SP1_maps.shape,SP2_maps.shape)
        # Total_fuse = self.fuse_para[0]*SA1_maps + self.fuse_para[1]*SA2_maps + self.fuse_para[2]*SA3_maps+\
        #              self.fuse_para[3]*SA4_maps + self.fuse_para[4]*SA5_maps+\
        #              self.fuse_para[5]*SP1_maps + self.fuse_para[6]*SP2_maps + self.fuse_para[7]*SP3_maps+\
        #              self.fuse_para[8]*SP4_maps + self.fuse_para[9]*SP5_maps
        
        # return [Total_fuse, SA1_maps, SA2_maps, SA3_maps, SA4_maps, SA5_maps, 
        #                     SP1_maps, SP2_maps, SP3_maps, SP4_maps, SP5_maps]

        # return [SA5_maps]

        Total_fuse = self.fuse_para[0]*SA1_maps + self.fuse_para[1]*SA2_maps + self.fuse_para[2]*SA3_maps+\
                     self.fuse_para[3]*SA4_maps + self.fuse_para[4]*SA5_maps
        
        return [Total_fuse, SA1_maps, SA2_maps, SA3_maps, SA4_maps, SA5_maps]
    
    def forwardbase(self, x):
        b, c, h, w = x.size()

        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.Net3D.l1_3D(x_3d)
        SA1_2D = self.Net3D.SA_CPR1(SA1)
        # SA1_sideout = sideout2d(SA1_2D, self.Net3D.SA_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.Net3D.l2_3D(SA1)
        SA2_2D = self.Net3D.SA_CPR2(SA2)
        # SA2_sideout = sideout2d(SA2_2D, self.Net3D.SA_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.Net3D.l3_3D(SA2)
        SA3_2D = self.Net3D.SA_CPR3(SA3)
        # SA3_sideout = sideout2d(SA3_2D, self.Net3D.SA_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.Net3D.l4_2D(SA3_2D)
        # SA4_sideout = sideout2d(SA4_2D, self.Net3D.SA_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.Net3D.l5_2D(SA4_2D)
        # print(SA5_2D.shape)
        # SA5_sideout = self.Seg_Conv1(SA5_2D)
        # SA5_sideout = sideout2d(SA5_2D, self.Net3D.SA_Sideout4)
        SA5_sideout = self.Net3D.SA_Sideout4(SA5_2D.view(b, -1))

        # Total_fuse = self.fuse_para[0]*SA1_maps + self.fuse_para[1]*SA2_maps + self.fuse_para[2]*SA3_maps+\
        #              self.fuse_para[3]*SA4_maps + self.fuse_para[4]*SA5_maps
        
        # return [Total_fuse, SA1_maps, SA2_maps, SA3_maps, SA4_maps, SA5_maps]
        return [SA5_sideout]
    
    # _transprob
    def forward(self, x):
        b, c, h, w = x.size()
        # print('_transprob')
        pt_center = center_pixel(x)
        pt_img = compute_ratio_withstep(pt_center)
        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.Net3D.l1_3D(x_3d)
        SA1_2D = self.Net3D.SA_CPR1(SA1)
        SA1_maps = sideout2d(SA1_2D, self.Net3D.SA_Sideout1)
        # SA1_2D_mask = self.gen_segmask(self.Seg_Conv1(SA1_2D)).view(b, -1, h*w)
        
        SP1 = self.Net2D.SP_conv_1(pt_img)
        # SP1 = (SP1 * SA1_2D_mask)
        SP1_maps = sideout2d(SP1, self.Net2D.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.Net3D.l2_3D(SA1)
        SA2_2D = self.Net3D.SA_CPR2(SA2)
        SA2_maps = sideout2d(SA2_2D, self.Net3D.SA_Sideout2)
        # SA2_2D_mask = self.gen_segmask(self.Seg_Conv2(SA2_2D)).view(b, -1, h*w)
        
        SP2 = self.Net2D.SP_conv_2(SP1)
        # SP2 = (SP2 * SA2_2D_mask)
        SP2_maps = sideout2d(SP2, self.Net2D.SP_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.Net3D.l3_3D(SA2)
        SA3_2D = self.Net3D.SA_CPR3(SA3)
        SA3_maps = sideout2d(SA3_2D, self.Net3D.SA_Sideout3)
        # SA3_2D_mask = self.gen_segmask(self.Seg_Conv3(SA3_2D)).view(b, -1, h*w)
        
        SP3 = self.Net2D.SP_conv_3(SP2)
        # SP3 = (SP3 * SA3_2D_mask)
        SP3_maps = sideout2d(SP3, self.Net2D.SP_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.Net3D.l4_2D(SA3_2D)
        SA4_maps = sideout2d(SA4_2D, self.Net3D.SA_Sideout4)
        # SA4_2D_mask = self.gen_segmask(self.Seg_Conv4(SA4_2D)).view(b, -1, h*w)
        
        SP4 = self.Net2D.SP_conv_4(SP3)
        # SP4 = (SP4 * SA4_2D_mask)
        SP4_maps = sideout2d(SP4, self.Net2D.SP_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.Net3D.l5_2D(SA4_2D)
        SA5_maps = sideout2d(SA5_2D, self.Net3D.SA_Sideout5)
        # SA5_2D_mask = self.gen_segmask(self.Seg_Conv5(SA5_2D)).view(b, -1, h*w)

        SP5 = self.Net2D.SP_conv_5(SP4)
        # SP5 = (SP5 * SA5_2D_mask)
        SP5_maps = sideout2d(SP5, self.Net2D.SP_Sideout5)


        sideout_list = [SA1_maps, SA2_maps, SA3_maps, SA4_maps, SA5_maps, 
                        SP1_maps, SP2_maps, SP3_maps, SP4_maps, SP5_maps]

        # pre_t = self.fuse_mlp(torch.cat([x for x in sideout_list], 1))
        # print(self.transprob_glob.shape)
        pre_t = torch.zeros(b, self.class_num).cuda()
        for index in range(len(sideout_list)):
            outs = sideout_list[index].view(b, -1, 1)
            transprob = self.transprob_glob[index].view(1, self.class_num, self.class_num).repeat(b, 1, 1)
            # pre = torch.bmm(transprob, outs)[:,:,0]  #是否转置
            pre = (1 + transprob) * outs  #是否转置
            pre = pre.sum(1)
            pre_t = pre_t + pre
        
        return [pre_t, SA1_maps, SA2_maps, SA3_maps, SA4_maps, SA5_maps, 
                        SP1_maps, SP2_maps, SP3_maps, SP4_maps, SP5_maps]
