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

class Segment_3DCenter(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Segment_3DCenter, self).__init__()
        print('Using Segment_3DCenter')
        self.class_num = out_channel
        #///////////////////////////////////////////////////////////////////////
        # Segment Para
        self.Seg_Num = 1
        #///////////////////////////////////////////////////////////////////////
        # Nets
        self.Net3D = SA3D2D_seg(in_channel, out_channel, self.Seg_Num)
        self.Net2D_L1 = SP_2DDilaConv(1, out_channel)
        self.Net2D_L2 = SP_2DDilaConv(1, out_channel)
        self.Net2D_L3 = SP_2DDilaConv(1, out_channel)
        self.Net2D_L4 = SP_2DDilaConv(1, out_channel)
        self.Net2D_L5 = SP_2DDilaConv(1, out_channel)
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

    def gen_segmask(self, feats):
        b, c, h, w = feats.size()
        feat_argmax = torch.argmax(feats, 1)
        # print(feat_argmax,feat_argmax.shape)
        # print(torch.nn.functional.one_hot(feat_argmax, self.Seg_Num),torch.nn.functional.one_hot(feat_argmax, self.Seg_Num).shape)
        return torch.nn.functional.one_hot(feat_argmax, c).permute(0, 3, 1, 2), feat_argmax

    def prob_norm(self, a, b, result_a, result_b):
        '''
            Output: result_a*(W1) + result_b*(1-W1)
            W1 = (a^2/(a^2+b^2))  W2 = (b^2/(a^2+b^2))
        '''
        W1 = a*a/(a*a+b*b)
        W2 = b*b/(a*a+b*b)
        return (W1*result_a + W2*result_b)
    
    def act_mask(self, feats):
        '''
            找到Feats中值大于中心像元值的地方，也就是激活的地方
            Output:  Mask
        '''
        feats = torch.abs(feats)
        b, l, h, w = feats.size()

        # 中心像素位置
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)

        center_value = center_pixel(feats).view(-1, 1, 1, 1).repeat(1, 1, 11, 11)
        feats_mask = F.relu(feats - center_value)
        feats_mask = self.softmax(feats_mask)

        # add = torch.zeros_like(feats_mask).cuda()
        # add[:,:, pth, ptw] = 1
        
        # feats_mask[:,:, pth, ptw] = feats_mask[:,:, pth, ptw] + 1     #是否要+1
        return feats_mask

    def act_mask_multichannel(self, feats_b):
        b, c, h, w = feats_b.size()
        a = []
        featmask_0 = self.act_mask(feats_b[:,0,:,:].unsqueeze(1))
        a.append(featmask_0)
        for channel in range(1, c, 1):
            feat_c = feats_b[:,channel,:,:].unsqueeze(1)
            featmask_0 = featmask_0 + self.act_mask(feat_c)
            a.append(self.act_mask(feat_c))
        # print(featmask_0)
        return featmask_0

    def similar_mask(self, x, feats, n=5):
        '''
            找到Feats中值与中心像元值相似的地方，取n个
            Input:   x(HS image)
                     feats(Seg 2D image)
            Output:  Mask
        '''
        b, l, h, w = feats.size()
        torch.set_printoptions(threshold=np.inf)
        center_value = center_pixel(feats).view(-1, 1, 1, 1)
        feats_mask = torch.abs(feats - center_value).view(b, -1)        # b, h*w
        
        # 自己实现的，和下面的topk一个作用，但是目前不能用在batch上
        # feats_sort = torch.sort(feats_mask, 1)
        # sort_index = feats_sort[1]
        # select_index = sort_index[:, 0:n]
        # feats_flat = feats.view(b, h*w)
        # similar = feats_flat[0][select_index[0]]

        select_index = torch.topk(feats_mask, 5, dim=1, largest=False)[1].unsqueeze(1).repeat(1, x.shape[1], 1)
        img_flat = x.view(b, -1, h*w)
        similar = torch.gather(img_flat, 2, select_index)
        
        return similar

    def pos_pool2d(self, feats, masks, mlp):
        b, c, h, w = feats.size()
        PosNum = masks.view(b, 1, -1).sum(2)          #   激活的像元个数
        ActFeat_sum = (masks * feats).view(b, c, -1).sum(2)  #   激活的像元的光谱总和
        ActMean = ActFeat_sum / PosNum               #   激活的像元的光谱均值     b, 1, h, w

        return mlp(ActMean.view(b, -1))
#///////////////////////////////////////////////////////////////////////
    
    # _ct
    def forward_act(self, x):
        b, c, h, w = x.size()
        
        x_3d = torch.unsqueeze(x, 1)
        SP_CT = compute_ratio_withstep(center_pixel(x))
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.Net3D.l1_3D(x_3d)
        SA1_2D = self.Net3D.SA_CPR1(SA1)
        SA1_sideout = sideout2d(SA1_2D, self.Net3D.SA_Sideout1)

        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act1 = self.Seg_Conv1(SA1_2D)
        # SP_act1 = SA1_2D.sum(1).unsqueeze(1)
        SP1_actmask = self.similar_mask(x, SP_act1)                    #   是用原来的值还是用1 !!!
        SP1_PosNum = SP1_actmask.view(b, 1, -1).sum(2)          #   激活的像元个数
        SP1_2DActSum = (SP1_actmask * x).view(b, c, -1).sum(2)  #   激活的像元的光谱总和
        SP1_2DActMean = SP1_2DActSum / SP1_PosNum               #   激活的像元的光谱均值     b, 1, h, w

        SP1_img = compute_ratio_withstep(SP1_2DActSum)
        # SP1_img = compute_spatt_conv1D(SP1_2DActMean.unsqueeze(1), self.mlp1_1, self.mlp1_2)
        SP1_img = torch.cat([SP1_img, SP_CT], 1)
        SP1_sideout = sideout2d(self.Net2D_L1(SP1_img), self.Net2D_L1.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.Net3D.l2_3D(SA1)
        SA2_2D = self.Net3D.SA_CPR2(SA2)
        SA2_sideout = sideout2d(SA2_2D, self.Net3D.SA_Sideout2)

        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act2 = self.Seg_Conv2(SA2_2D)
        # SP_act2 = SA2_2D.sum(1).unsqueeze(1)
        SP2_actmask = self.similar_mask(SP_act2)                    #   是用原来的值还是用1 !!!
        SP2_PosNum = SP2_actmask.view(b, 1, -1).sum(2)          #   激活的像元个数
        SP2_2DActSum = (SP2_actmask * x).view(b, c, -1).sum(2)  #   激活的像元的光谱总和
        SP2_2DActMean = SP2_2DActSum / SP2_PosNum               #   激活的像元的光谱均值     b, 1, h, w

        SP2_img = compute_ratio_withstep(SP2_2DActSum)
        # SP2_img = compute_spatt_conv1D(SP2_2DActMean.unsqueeze(1), self.mlp2_1, self.mlp2_2)
        SP2_img = torch.cat([SP2_img, SP_CT], 1)
        SP2_sideout = sideout2d(self.Net2D_L2(SP2_img), self.Net2D_L2.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.Net3D.l3_3D(SA2)
        SA3_2D = self.Net3D.SA_CPR3(SA3)
        SA3_sideout = sideout2d(SA3_2D, self.Net3D.SA_Sideout3)

        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act3 = self.Seg_Conv3(SA3_2D)
        # SP_act3 = SA3_2D.sum(1).unsqueeze(1)
        SP3_actmask = self.similar_mask(SP_act3)                    #   是用原来的值还是用1 !!!
        SP3_PosNum = SP3_actmask.view(b, 1, -1).sum(2)          #   激活的像元个数
        SP3_2DActSum = (SP3_actmask * x).view(b, c, -1).sum(2)  #   激活的像元的光谱总和
        SP3_2DActMean = SP3_2DActSum / SP3_PosNum               #   激活的像元的光谱均值     b, 1, h, w

        SP3_img = compute_ratio_withstep(SP3_2DActSum)
        # SP3_img = compute_spatt_conv1D(SP3_2DActMean.unsqueeze(1), self.mlp3_1, self.mlp3_2)
        SP3_img = torch.cat([SP3_img, SP_CT], 1)
        SP3_sideout = sideout2d(self.Net2D_L3(SP3_img), self.Net2D_L3.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.Net3D.l4_2D(SA3_2D)
        SA4_sideout = sideout2d(SA4_2D, self.Net3D.SA_Sideout4)

        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act4 = self.Seg_Conv4(SA4_2D)
        # SP_act4 = SA4_2D.sum(1).unsqueeze(1)
        SP4_actmask = self.similar_mask(SP_act4)                    #   是用原来的值还是用1 !!!
        SP4_PosNum = SP4_actmask.view(b, 1, -1).sum(2)          #   激活的像元个数
        SP4_2DActSum = (SP4_actmask * x).view(b, c, -1).sum(2)  #   激活的像元的光谱总和
        SP4_2DActMean = SP4_2DActSum / SP4_PosNum               #   激活的像元的光谱均值     b, 1, h, w

        SP4_img = compute_ratio_withstep(SP4_2DActSum)
        # SP4_img = compute_spatt_conv1D(SP4_2DActMean.unsqueeze(1), self.mlp4_1, self.mlp4_2)
        SP4_img = torch.cat([SP4_img, SP_CT], 1)
        SP4_sideout = sideout2d(self.Net2D_L4(SP4_img), self.Net2D_L4.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.Net3D.l5_2D(SA4_2D)
        SA5_sideout = sideout2d(SA5_2D, self.Net3D.SA_Sideout5)

        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act5 = self.Seg_Conv5(SA5_2D)
        # SP_act5 = SA5_2D.sum(1).unsqueeze(1)
        SP5_actmask = self.similar_mask(SP_act5)                    #   是用原来的值还是用1 !!!
        SP5_PosNum = SP5_actmask.view(b, 1, -1).sum(2)          #   激活的像元个数
        SP5_2DActSum = (SP5_actmask * x).view(b, c, -1).sum(2)  #   激活的像元的光谱总和
        SP5_2DActMean = SP5_2DActSum / SP5_PosNum               #   激活的像元的光谱均值     b, 1, h, w

        SP5_img = compute_ratio_withstep(SP5_2DActSum)
        # SP5_img = compute_spatt_conv1D(SP5_2DActMean.unsqueeze(1), self.mlp5_1, self.mlp5_2)
        SP5_img = torch.cat([SP5_img, SP_CT], 1)
        SP5_sideout = sideout2d(self.Net2D_L5(SP5_img), self.Net2D_L5.SP_Sideout1)
        
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
    
    # _pospool
    def forward_pospool(self, x):
        b, c, h, w = x.size()
        
        x_3d = torch.unsqueeze(x, 1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.Net3D.l1_3D(x_3d)
        SA1_2D = self.Net3D.SA_CPR1(SA1)
        
        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act1 = self.Seg_Conv1(SA1_2D)
        SP1_actmask = self.act_mask(SP_act1)                    #   是用原来的值还是用1 !!!
        SA1_sideout = self.pos_pool2d(SA1_2D, SP1_actmask, self.Net3D.SA_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.Net3D.l2_3D(SA1)
        SA2_2D = self.Net3D.SA_CPR2(SA2)
        
        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act2 = self.Seg_Conv2(SA2_2D)
        SP2_actmask = self.act_mask(SP_act2)                    #   是用原来的值还是用1 !!!
        SA2_sideout = self.pos_pool2d(SA2_2D, SP2_actmask, self.Net3D.SA_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.Net3D.l3_3D(SA2)
        SA3_2D = self.Net3D.SA_CPR3(SA3)
        
        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act3 = self.Seg_Conv3(SA3_2D)
        SP3_actmask = self.act_mask(SP_act3)                    #   是用原来的值还是用1 !!!
        SA3_sideout = self.pos_pool2d(SA3_2D, SP3_actmask, self.Net3D.SA_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.Net3D.l4_2D(SA3_2D)
        
        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act4 = self.Seg_Conv4(SA4_2D)
        SP4_actmask = self.act_mask(SP_act4)                    #   是用原来的值还是用1 !!!
        SA4_sideout = self.pos_pool2d(SA4_2D, SP4_actmask, self.Net3D.SA_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.Net3D.l5_2D(SA4_2D)
        
        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act5 = self.Seg_Conv5(SA5_2D)
        SP5_actmask = self.act_mask(SP_act5)                    #   是用原来的值还是用1 !!!
        SA5_sideout = self.pos_pool2d(SA5_2D, SP5_actmask, self.Net3D.SA_Sideout5)
        
        # Total_fuse = self.prob_norm(self.fuse_para[0], self.fuse_para[1], SA1_sideout, SP1_sideout) + \
        #              self.prob_norm(self.fuse_para[2], self.fuse_para[3], SA2_sideout, SP2_sideout) + \
        #              self.prob_norm(self.fuse_para[4], self.fuse_para[5], SA3_sideout, SP3_sideout) + \
        #              self.prob_norm(self.fuse_para[6], self.fuse_para[7], SA4_sideout, SP4_sideout) + \
        #              self.prob_norm(self.fuse_para[8], self.fuse_para[9], SA5_sideout, SP5_sideout)

        # print(SP5_PosNum, SP4_PosNum, SP3_PosNum, SP2_PosNum, SP1_PosNum)
        Total_fuse = self.fuse_para[0]*SA1_sideout + self.fuse_para[1]*SA2_sideout + self.fuse_para[2]*SA3_sideout+\
                     self.fuse_para[3]*SA4_sideout + self.fuse_para[4]*SA5_sideout

        
        return [Total_fuse, SA1_sideout, SA2_sideout, SA3_sideout, SA4_sideout, SA5_sideout]

    def forward_1(self, x):
        b, c, h, w = x.size()
        wind = 1

        x_3d = torch.unsqueeze(x, 1)

        pt_center = center_pixel(x)
        pt_img = compute_ratio_withstep(pt_center)
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.Net3D.l1_3D(x_3d)
        SA1_2D = self.Net3D.SA_CPR1(SA1)
        SA1_sideout = sideout2d(SA1_2D, self.Net3D.SA_Sideout1)

        # SA1_2D_mask, SA1_2D_argmax = self.gen_segmask(SA1_2D)
        # pos_channel = SA1_2D[SA1_2D_argmax[:,5,5]]

        # SP1_img = compute_ratio_withstep_entireimage(SA1[:,:,:,5,5])
        SP1_img = SA1_2D[:,:,5-wind:5+wind+1,5-wind:5+wind+1].view(b, 16, 2*wind+1, 2*wind+1,)
        SP1_sideout = sideout2d(SP1_img, self.Net2D_L1.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.Net3D.l2_3D(SA1)
        SA2_2D = self.Net3D.SA_CPR2(SA2)
        SA2_sideout = sideout2d(SA2_2D, self.Net3D.SA_Sideout2)

        SP2_img = SA2_2D[:,:,5-wind:5+wind+1,5-wind:5+wind+1].view(b, 16, 2*wind+1, 2*wind+1,)
        SP2_sideout = sideout2d(SP2_img, self.Net2D_L1.SP_Sideout2)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.Net3D.l3_3D(SA2)
        SA3_2D = self.Net3D.SA_CPR3(SA3)
        SA3_sideout = sideout2d(SA3_2D, self.Net3D.SA_Sideout3)

        SP3_img = SA3_2D[:,:,5-wind:5+wind+1,5-wind:5+wind+1].view(b, 16, 2*wind+1, 2*wind+1,)
        SP3_sideout = sideout2d(SP3_img, self.Net2D_L1.SP_Sideout3)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.Net3D.l4_2D(SA3_2D)
        SA4_sideout = sideout2d(SA4_2D, self.Net3D.SA_Sideout4)

        SP4_sideout = sideout2d(SA4_2D[:,:,5-wind:5+wind+1,5-wind:5+wind+1].view(b, 16, 2*wind+1, 2*wind+1,), self.Net2D_L1.SP_Sideout4)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.Net3D.l5_2D(SA4_2D)
        SA5_sideout = sideout2d(SA5_2D, self.Net3D.SA_Sideout5)
        SP5_sideout = sideout2d(SA5_2D[:,:,5-wind:5+wind+1,5-wind:5+wind+1].view(b, 16, 2*wind+1, 2*wind+1,), self.Net2D_L1.SP_Sideout5)
        
        Total_fuse = self.prob_norm(self.fuse_para[0], self.fuse_para[1], SA1_sideout, SP1_sideout) + \
                     self.prob_norm(self.fuse_para[2], self.fuse_para[3], SA2_sideout, SP2_sideout) + \
                     self.prob_norm(self.fuse_para[4], self.fuse_para[5], SA3_sideout, SP3_sideout) + \
                     self.prob_norm(self.fuse_para[6], self.fuse_para[7], SA4_sideout, SP4_sideout) + \
                     self.prob_norm(self.fuse_para[8], self.fuse_para[9], SA5_sideout, SP5_sideout)
        
        return [Total_fuse, SA1_sideout, SA2_sideout, SA3_sideout, SA4_sideout, SA5_sideout, 
                            SP1_sideout, SP2_sideout, SP3_sideout, SP4_sideout, SP5_sideout]
    
    # similar_mask
    def forward(self, x):
        b, c, h, w = x.size()
        
        similar_num = 10
        x_3d = torch.unsqueeze(x, 1)
        SP_CT = compute_ratio_withstep(center_pixel(x))
        #///////////////////////////////////////////////////////////////////////
        # Layer 1
        SA1 = self.Net3D.l1_3D(x_3d)
        SA1_2D = self.Net3D.SA_CPR1(SA1)
        SA1_sideout = sideout2d(SA1_2D, self.Net3D.SA_Sideout1)

        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act1 = self.Seg_Conv1(SA1_2D)
        # SP_act1 = SA1_2D.sum(1).unsqueeze(1)
        SP1_sim = self.similar_mask(x, SP_act1, similar_num).permute(0, 2, 1)        # b, c, similar_num -> b, similar_num, c
        # SP1_img = compute_ratio_withstep_entireimage(SP1_sim)
        SP1_img = compute_ratio_withstep(torch.mean(SP1_sim, 1))
        SP1_sideout = sideout2d(self.Net2D_L1(SP1_img), self.Net2D_L1.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 2
        SA2 = self.Net3D.l2_3D(SA1)
        SA2_2D = self.Net3D.SA_CPR2(SA2)
        SA2_sideout = sideout2d(SA2_2D, self.Net3D.SA_Sideout2)

        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act2 = self.Seg_Conv2(SA2_2D)
        # SP_act2 = SA2_2D.sum(1).unsqueeze(1)
        SP2_sim = self.similar_mask(x, SP_act2, similar_num).permute(0, 2, 1)        # b, c, similar_num -> b, similar_num, c
        # SP2_img = compute_ratio_withstep_entireimage(SP2_sim)
        SP2_img = compute_ratio_withstep(torch.mean(SP2_sim, 1))
        SP2_sideout = sideout2d(self.Net2D_L2(SP2_img), self.Net2D_L2.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 3       
        SA3 = self.Net3D.l3_3D(SA2)
        SA3_2D = self.Net3D.SA_CPR3(SA3)
        SA3_sideout = sideout2d(SA3_2D, self.Net3D.SA_Sideout3)

        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act3 = self.Seg_Conv3(SA3_2D)
        # SP_act3 = SA3_2D.sum(1).unsqueeze(1)
        SP3_sim = self.similar_mask(x, SP_act3, similar_num).permute(0, 2, 1)        # b, c, similar_num -> b, similar_num, c
        # SP3_img = compute_ratio_withstep_entireimage(SP3_sim)
        SP3_img = compute_ratio_withstep(torch.mean(SP3_sim, 1))
        SP3_sideout = sideout2d(self.Net2D_L3(SP3_img), self.Net2D_L3.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 4
        SA4_2D = self.Net3D.l4_2D(SA3_2D)
        SA4_sideout = sideout2d(SA4_2D, self.Net3D.SA_Sideout4)

        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act4 = self.Seg_Conv4(SA4_2D)
        # SP_act4 = SA4_2D.sum(1).unsqueeze(1)
        SP4_sim = self.similar_mask(x, SP_act4, similar_num).permute(0, 2, 1)        # b, c, similar_num -> b, similar_num, c
        # SP4_img = compute_ratio_withstep_entireimage(SP4_sim)
        SP4_img = compute_ratio_withstep(torch.mean(SP4_sim, 1))
        SP4_sideout = sideout2d(self.Net2D_L4(SP4_img), self.Net2D_L4.SP_Sideout1)
        #///////////////////////////////////////////////////////////////////////
        # Layer 5
        SA5_2D = self.Net3D.l5_2D(SA4_2D)
        SA5_sideout = sideout2d(SA5_2D, self.Net3D.SA_Sideout5)

        #   计算哪些像素是被激活的（值大于中心像元）
        SP_act5 = self.Seg_Conv5(SA5_2D)
        # SP_act5 = SA5_2D.sum(1).unsqueeze(1)
        SP5_sim = self.similar_mask(x, SP_act5, similar_num).permute(0, 2, 1)        # b, c, similar_num -> b, similar_num, c
        # SP5_img = compute_ratio_withstep_entireimage(SP5_sim)
        SP5_img = compute_ratio_withstep(torch.mean(SP5_sim, 1))
        SP5_sideout = sideout2d(self.Net2D_L5(SP5_img), self.Net2D_L5.SP_Sideout1)
        
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