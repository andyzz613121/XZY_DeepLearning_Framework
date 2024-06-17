import torch
import torch.nn as nn
from model.HyperSpectral.Base_Network import get_mlp
from model.HyperSpectral.Basic_Operation import *
from model.Self_Module.Layer_operations import *
from model.HyperSpectral.Auto_Multiply.Backbone import SP_2D_ASPP, SP_2D
from model.HyperSpectral.Auto_Multiply.Act_Layer import Conv1D_Act, Conv2D_Act, BandMLP

class AutoContrast_ASPP(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        print('Using AutoContrast_ASPP model in AutoContrast_ASPP')
        self.backbone = SP_2D_ASPP(input_channels, out_channels)
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
        
        self.conv1D1_1 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        self.conv1D1_2 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        self.conv1D2_1 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        self.conv1D2_2 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        self.conv1D3_1 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        self.conv1D3_2 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        self.conv1D4_1 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        self.conv1D4_2 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        self.conv1D5_1 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        self.conv1D5_2 = Conv1D_Act(scale_num, ratio, kernel_size, padding)
        self.conv1D_list1 = [self.conv1D1_1, self.conv1D2_1, self.conv1D3_1, self.conv1D4_1, self.conv1D5_1]
        self.conv1D_list2 = [self.conv1D1_2, self.conv1D2_2, self.conv1D3_2, self.conv1D4_2, self.conv1D5_2]

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

        self.bandwise_mlp1 = nn.Sequential(
            nn.Linear(2*ratio, ratio),
            nn.ReLU(inplace=True),
            nn.Linear(ratio, 1)

        )
        self.bandwise_mlp2 = nn.Sequential(
            nn.Linear(2*ratio, ratio),
            nn.ReLU(inplace=True),
            nn.Linear(ratio, 1)

        )
        self.bandwise_mlp3 = nn.Sequential(
            nn.Linear(2*ratio, ratio),
            nn.ReLU(inplace=True),
            nn.Linear(ratio, 1)
        )
        self.bandwise_mlp_list = [self.bandwise_mlp1, self.bandwise_mlp2, self.bandwise_mlp3]
        
        #///////////////////////////////////////////////////////////////////////
        self.out_fc = get_mlp(2, [16, out_channels])
        
        self.fuse_para = nn.Parameter(torch.ones([15]))
        self.GAP2D = torch.nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x, mode='train'):
        if mode == 'train':
            return self.forward_train(x)
        else:
            return self.forward_test(x)
        
    def forward_test(self, img_patch):
        b, c, patch_h, patch_w = img_patch.size()
        wind = 5
        feat_patch = MS_spectral_entireimage(img_patch, rsp_rate_list=self.scale_list)
        feat_patch = MS_spectral_FeatureExtract_bandwise4D(feat_patch, self.conv2D_list1, self.conv2D_list2, self.bandwise_mlp_list, act = '2D', samesize=True)[0]
        feat_patch = feat_patch[0][0] #[256, 256, 12, 12]
        feat_patch = feat_patch.view(patch_h, patch_w, -1) #[256, 256, 144]
        feat_patch = feat_patch.permute(2, 0, 1).unsqueeze(1) #[144, 1, 256, 256]
        avg_conv = torch.ones([b, 1, 2*wind+1, 2*wind+1]).cuda() / ((2*wind+1)*(2*wind+1))
        feat_avg = F.conv2d(feat_patch, avg_conv, padding=5) #[144, 1, 256, 256]
        feat_avg = feat_avg.view(c, c, patch_h*patch_w*b).permute(2, 0, 1).unsqueeze(1) #[256*256, 1, 12, 12]
        
        x1 = self.backbone.SP_conv_1(feat_avg)
        x2 = self.backbone.SP_conv_2(x1)
        x3 = self.backbone.SP_conv_3(x2)
        x4 = self.backbone.SP_conv_4(x3)
        x5 = self.backbone.SP_conv_5(x4)
        SP5_GAP = self.GAP2D(x5).view(patch_h*patch_w*b, -1)
        out = self.out_fc(SP5_GAP)
        out = out.transpose(0, 1).contiguous().view(b, -1, 256, 256)
        return out
    
    def forward_train(self, x):
        b, c, h, w = x.size()

        '''
        Auto Feature Extract Separate
        '''
        # codes, MS_Feat = self.model_AutoFE(x)

        '''
        MultiScale Feature(Resample Spectral)
        '''
        # MS_Spectral = MS_spectral(center_pixel(x), rsp_rate_list=self.scale_list)
        # # MS_Feat = MS_spectral_FeatureExtract(MS_Spectral, self.mlp_list1, self.mlp_list2, samesize=True)
        # MS_Feat = MS_spectral_FeatureExtract(MS_Spectral, self.conv1D_list1, self.conv1D_list2, act = '1D', samesize=True)
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

        '''
        MultiScale Feature(Resample Spectral entireimage)
        '''
        # img_patch = center_img(x, 1).contiguous().view(b, c, -1).mean(2).view(b, c, 1, 1)
        x1 = x[0:int(b/2),:,:,:]
        x2 = x[int(b/2):b,:,:,:]
        MS_Feat_list = []
        for x_batch in [x1, x2]:
            b_batch, _, _, _ = x_batch.size()
            img_patch = center_img(x_batch, 5)
            # img_patch = center_pixel(x_batch).view(b_batch, c, 1, 1)

            MS_Spectral = MS_spectral_entireimage(img_patch, rsp_rate_list=self.scale_list)
            # MS_Feat = MS_spectral_FeatureExtract_bandwise4D_LF(MS_Spectral, self.conv2D_list1, self.conv2D_list2, self.bandwise_mlp_list, act = '2D', samesize=True)
            # MS_Feat = torch.cat([torch.mean(x, 1) for x in MS_Feat], 1).view(b_batch, -1, 14, 14)
            MS_Feat = MS_spectral_FeatureExtract_bandwise4D(MS_Spectral, self.conv1D_list1, self.conv1D_list2, self.bandwise_mlp_list, act = '1D', samesize=True)
            # print(MS_Feat[0].shape)
            # MS_Feat = torch.cat([torch.mean(x.view(b_batch, -1, 121, c, c), 2) for x in MS_Feat], 1).view(b_batch, -1, c, c)
            # MS_Feat: [scale1, scale2, ...] -> [[B, 1, wid, wid, c, c]...] -> [[16, 1, 11, 11, 12, 12]]
            MS_Feat = torch.cat([torch.mean(x.view(b_batch, x.shape[1], -1, c, c), 2) for x in MS_Feat], 1).view(b_batch, -1, c, c)
            MS_Feat_list.append(MS_Feat)

        MS_Feat = torch.cat([x for x in MS_Feat_list], 0)

        x1 = self.backbone.SP_conv_1(MS_Feat)
        x2 = self.backbone.SP_conv_2(x1)
        x3 = self.backbone.SP_conv_3(x2)
        x4 = self.backbone.SP_conv_4(x3)
        x5 = self.backbone.SP_conv_5(x4)
    
        SP5_GAP = self.GAP2D(x5).view(b, -1)
        out = self.out_fc(SP5_GAP)
        # return [out], [SP5_GAP], [MS_Feat], [center_pixel(x)]
        return [out], [SP5_GAP]

class AutoContrast_ASPP_Branchs(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        print('Using AutoContrast_ASPP_3Branch model in AutoConstract_ASPP')
        
        self.branch1 = AutoContrast_ASPP(9, out_channels)
        self.branch2 = AutoContrast_ASPP(3, out_channels)
        self.branch3 = AutoContrast_ASPP(3, out_channels)

        self.fuse_para = nn.Parameter(torch.ones([10]))

    def forward(self, x):
        '''
        Input: 
                x: list [(B, 1, H, W), (B, 1, H, W) ... ]   len(x)==branch_num
        '''

        '''
        MultiScale Feature(Resample Spectral)
        '''
        MS_Spectral = MS_spectral(center_pixel(x), rsp_rate_list=self.branch1.scale_list)
        
        MS_Feat_1D = MS_spectral_FeatureExtract_bandwise(MS_Spectral, self.branch1.conv2D_list1, self.branch1.conv2D_list2, self.branch1.bandwise_mlp_list, act='2D', samesize=True)
        MS_Feat_2D = MS_spectral_FeatureExtract_bandwise(MS_Spectral, self.branch2.conv2D_list1, self.branch2.conv2D_list2, self.branch2.bandwise_mlp_list, act='2D', samesize=True)
        MS_Feat_3D = MS_spectral_FeatureExtract_bandwise(MS_Spectral, self.branch3.conv2D_list1, self.branch3.conv2D_list2, self.branch3.bandwise_mlp_list, act='2D', samesize=True)
        
        MS_Feat_1D = torch.cat([x for x in MS_Feat_1D], 1)
        MS_Feat_2D = torch.cat([x for x in MS_Feat_2D], 1)
        MS_Feat_3D = torch.cat([x for x in MS_Feat_3D], 1)
        MS_Feat = torch.cat([MS_Feat_1D, MS_Feat_2D, MS_Feat_3D], 1)
        out_1, feat_1 = self.branch1(MS_Feat)
        # out_2, feat_2 = self.branch2(MS_Feat_2D)
        
        # out = out_1[0]*self.fuse_para[0] + out_2[0]*self.fuse_para[1]
        # feat_list = [feat_1[0], feat_2[0]]

        # return [out], feat_list
        return out_1, feat_1

    def forward1(self, x):
        '''
        Input: 
                x: list [(B, 1, H, W), (B, 1, H, W) ... ]   len(x)==branch_num
        '''

        '''
        MultiScale Feature(Resample Spectral)
        '''
        MS_Spectral = MS_spectral(center_pixel(x), rsp_rate_list=self.branch1.scale_list)
        MS_Feat = MS_spectral_FeatureExtract(MS_Spectral, self.branch1.conv2D_list1, self.branch1.conv2D_list2, act='2D', samesize=True)

        # MS_Feat_1D = MS_spectral_FeatureExtract(MS_Spectral, self.branch1.conv2D_list1, self.branch1.conv2D_list2, act='2D', samesize=True)
        # MS_Feat_2D = MS_spectral_FeatureExtract(MS_Spectral, self.branch2.conv2D_list1, self.branch2.conv2D_list2, act='2D', samesize=True)
        # MS_Feat_1D = torch.cat([x for x in MS_Feat_1D], 1)
        # MS_Feat_2D = torch.cat([x for x in MS_Feat_2D], 1)

        out_1, feat_1 = self.branch1(MS_Feat[0])
        out_2, feat_2 = self.branch2(MS_Feat[1])
        out_3, feat_3 = self.branch3(MS_Feat[2])

        out = out_1[0]*self.fuse_para[0] + out_2[0]*self.fuse_para[1] + out_3[0]*self.fuse_para[2]
        feat_list = [feat_1[0], feat_2[0], feat_3[0]]
        # out = out_1[0]*self.fuse_para[0] + out_2[0]*self.fuse_para[1]
        # feat_list = [feat_1[0], feat_2[0]]

        return [out], feat_list

class AutoContrast_ASPP_Branchs_old(nn.Module):
    def __init__(self, input_channels, out_channels, branch_num=3):
        super().__init__()
        print('Using AutoContrast_ASPP_3Branch model in AutoConstract_ASPP')
        
        self.branch_num = branch_num
        self.net_list = []
        for i in range(branch_num):
            self.net_list.append(AutoContrast_ASPP(input_channels, out_channels).cuda())
        self.fuse_para = nn.Parameter(torch.ones([branch_num]))

    def forward(self, x):
        '''
        Input: 
                x: list [(B, 1, H, W), (B, 1, H, W) ... ]   len(x)==branch_num
        '''

        '''
        MultiScale Feature(Resample Spectral)
        '''
        MS_Spectral = MS_spectral(center_pixel(x), rsp_rate_list=[1, 2, 4])
        # MS_Feat = MS_spectral_FeatureExtract(MS_Spectral, self.mlp_list1, self.mlp_list2, samesize=True)
        MS_Feat = MS_spectral_FeatureExtract(MS_Spectral, self.net_list[0].conv1D_list1, self.net_list[0].conv2D_list1, samesize=True)
        # MS_Feat = torch.cat([x for x in MS_Feat], 1)
        
        assert len(MS_Feat) == self.branch_num
        out = 0
        feat_cst_list = []
        for branch in range(self.branch_num):
            img_branch = MS_Feat[branch]
            out_b, feat_cst = self.net_list[branch](img_branch)
            out = out + out_b[0] * self.fuse_para[branch]
            feat_cst_list.append(feat_cst[0])

        return [out], feat_cst_list