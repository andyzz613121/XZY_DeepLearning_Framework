from numpy.core.numeric import outer
import torch
from torch import nn
import numpy as np
from model import SE_Layer
from model import priori_knowledge
import torch.nn.functional as F
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        #shortcut
        self.shortcut = nn.Sequential()
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    def forward(self, x):
        return nn.ReLU(inplace=False)(self.residual_function(x) + self.shortcut(x))

class CARB_Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CARB_Block,self).__init__()
        self.bottlenect = BasicBlock(input_channels,input_channels)
        self.sideout_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels)
        )
        
    def forward(self, x):
        x_1 = self.bottlenect(x)
        return self.sideout_conv(x_1)
       
class Fuse_Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Fuse_Block,self).__init__()
        self.img_prio_fuse_conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channels//2),
            nn.ReLU(inplace=False)
        )
        # self.sideout1 = Sideout_Block(input_channels//2, output_channels)
        self.sideout1 = CARB_Block(input_channels, output_channels)
        self.img_prio_fuse_conv2 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channels//2),
            nn.ReLU(inplace=False)
        )
        # self.sideout2 = Sideout_Block(input_channels//2, output_channels)
        self.sideout2 = CARB_Block(input_channels, output_channels)
        self.img_prio_fuse_conv3 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channels//2),
            nn.ReLU(inplace=False)
        )
        # self.sideout3 = Sideout_Block(input_channels//2, output_channels)
        self.sideout3 = CARB_Block(input_channels, output_channels)
        self.img_prio_fuse_conv4 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channels//2),
            nn.ReLU(inplace=False)
        )
        # self.sideout4 = Sideout_Block(input_channels//2, output_channels)
        self.sideout4 = CARB_Block(input_channels, output_channels)
        self.img_prio_fuse_conv5 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channels//2),
            nn.ReLU(inplace=False)
        )
        # self.sideout5 = Sideout_Block(input_channels//2, output_channels)
        self.sideout5 = CARB_Block(input_channels, output_channels)
        

    def forward(self, img_feature, prio_features_list):
        # fuse_inputs1 = torch.cat([img_feature, prio_features_list[0]], 1)
        # fuse_outputs1 = self.img_prio_fuse_conv1(fuse_inputs1)
        # fuse_inputs2 = torch.cat([img_feature, prio_features_list[1]], 1)
        # fuse_outputs2 = self.img_prio_fuse_conv2(fuse_inputs2)
        # fuse_inputs3 = torch.cat([img_feature, prio_features_list[2]], 1)
        # fuse_outputs3 = self.img_prio_fuse_conv3(fuse_inputs3)
        # fuse_inputs4 = torch.cat([img_feature, prio_features_list[3]], 1)
        # fuse_outputs4 = self.img_prio_fuse_conv4(fuse_inputs4)
        # fuse_inputs5 = torch.cat([img_feature, prio_features_list[4]], 1)
        # fuse_outputs5 = self.img_prio_fuse_conv5(fuse_inputs5)

        fuse_outputs1 = prio_features_list[0]
        fuse_outputs2 = prio_features_list[1]
        fuse_outputs3 = prio_features_list[2]
        fuse_outputs4 = prio_features_list[3]
        fuse_outputs5 = prio_features_list[4]
        out_list = [self.sideout1(fuse_outputs1), self.sideout2(fuse_outputs2), 
        self.sideout3(fuse_outputs3), self.sideout4(fuse_outputs4), self.sideout5(fuse_outputs5)]
        return out_list
        
class SegNet_img(nn.Module):
    def __init__(self, input_channels, output_channels, pre_train=True):
        super(SegNet_img,self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        #decoder
        self.deconv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.deconv_1_half = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.deconv_1_classifier = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=False)
        )
        self.unpool = nn.MaxUnpool2d(2,2)

        self.CARB_block1 = CARB_Block(64, 6)
        self.CARB_block2 = CARB_Block(128, 6)
        self.CARB_block3 = CARB_Block(256, 6)
        self.CARB_block4 = CARB_Block(512, 6)
        self.CARB_block5 = CARB_Block(512, 6)

        self.fuse_block1 = Fuse_Block(64*2, 1)
        self.fuse_block2 = Fuse_Block(128*2, 1)
        self.fuse_block3 = Fuse_Block(256*2, 1)
        self.fuse_block4 = Fuse_Block(512, 1)
        # self.fuse_block5 = Fuse_Block(512*2, 6)
        self.fuse_block5 = Fuse_Block(512, 1)

        self.fuse_img_prio_conv1 = nn.Sequential(
            nn.Conv2d(6*6, 6, kernel_size=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=False)
        )
        self.fuse_img_prio_conv2 = nn.Sequential(
            nn.Conv2d(6*6, 6, kernel_size=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=False)
        )
        self.fuse_img_prio_conv3 = nn.Sequential(
            nn.Conv2d(6*6, 6, kernel_size=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=False)
        )
        # self.fuse_img_prio_conv4 = nn.Sequential(
        #     nn.Conv2d(6*6, 6, kernel_size=1),
        #     nn.BatchNorm2d(6),
        #     nn.ReLU(inplace=False)
        # )
        self.fuse_img_prio_conv4_c1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False)
        )
        self.fuse_img_prio_conv4_c2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False)
        )
        self.fuse_img_prio_conv4_c3 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False)
        )
        self.fuse_img_prio_conv4_c4 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False)
        )
        self.fuse_img_prio_conv4_c5 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False)
        )
        self.fuse_img_prio_conv5_c1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False)
        )
        self.fuse_img_prio_conv5_c2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False)
        )
        self.fuse_img_prio_conv5_c3 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False)
        )
        self.fuse_img_prio_conv5_c4 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False)
        )
        self.fuse_img_prio_conv5_c5 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False)
        )

        self.output_fuse = nn.Sequential(
            nn.Conv2d(6*2, 6, kernel_size=1),
            nn.BatchNorm2d(6),
            # nn.ReLU(inplace=False)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

        if pre_train==True:
            if input_channels == 3:
                print('SegNet param init: input channels == 3, IMAGE, init weight by Pascal model')
                SegNet_load_Pascal_model(self, 'D:\\Code\\LULC\\Hed_Seg\\pretrained\\SegNet\\segnet_pascal_params.npy')
            else:
                print('SegNet param init: Using xavier_normal')
        else:
            print('SegNet param init: Using xavier_normal')
        

    def forward_first(self, x):
        #encoder
        conv_1 = self.conv_1(x)
        conv_1_copy = conv_1
        conv_1, index_1 = self.pool(conv_1)

        conv_2 = self.conv_2(conv_1)
        conv_2_copy = conv_2
        conv_2, index_2 = self.pool(conv_2)

        conv_3 = self.conv_3(conv_2)
        conv_3_copy = conv_3
        conv_3, index_3 = self.pool(conv_3)

        conv_4 = self.conv_4(conv_3)
        conv_4_copy = conv_4
        conv_4, index_4 = self.pool(conv_4)

        conv_5 = self.conv_5(conv_4)
        conv_5_copy = conv_5
        conv_5, index_5 = self.pool(conv_5)

        conv_5 = self.unpool(conv_5,index_5,output_size=conv_5_copy.shape)
        conv_5 = self.deconv_5(conv_5)

        conv_4 = self.unpool(conv_5,index_4,output_size=conv_4_copy.shape)
        conv_4 = self.deconv_4(conv_4)

        conv_3 = self.unpool(conv_4,index_3,output_size=conv_3_copy.shape)
        conv_3 = self.deconv_3(conv_3)

        conv_2 = self.unpool(conv_3,index_2,output_size=conv_2_copy.shape)
        conv_2 = self.deconv_2(conv_2)

        conv_1 = self.unpool(conv_2,index_1,output_size=conv_1_copy.shape)
        conv_1 = self.deconv_1_half(conv_1)
        out_PRD = self.deconv_1_classifier(conv_1)

        return out_PRD

    def forward_second(self, x, prio_features_list1, prio_features_list2, prio_features_list3, prio_features_list4, prio_features_list5):
        #encoder
        # label = label.unsqueeze(1).float()

        conv_1 = self.conv_1(x)
        conv_1_copy = conv_1
        # sideout_1 = self.side_block1(conv_1)
        # fuse_1 = self.fuse_block1(conv_1, prio_features_list1)
        # out_1 = self.fuse_img_prio_conv1(torch.cat([sideout_1, fuse_1[0], fuse_1[1], fuse_1[2], fuse_1[3], fuse_1[4]], 1))
        # out_1 = F.interpolate(out_1, size=(x.shape[2], x.shape[3]), mode='bilinear')
        conv_1, index_1 = self.pool(conv_1)

        conv_2 = self.conv_2(conv_1)
        conv_2_copy = conv_2
        # sideout_2 = self.side_block2(conv_2)
        # fuse_2 = self.fuse_block2(conv_2, prio_features_list2)
        # out_2 = self.fuse_img_prio_conv2(torch.cat([sideout_2, fuse_2[0], fuse_2[1], fuse_2[2], fuse_2[3], fuse_2[4]], 1))
        # out_2 = F.interpolate(out_2, size=(x.shape[2], x.shape[3]), mode='bilinear')
        conv_2, index_2 = self.pool(conv_2)

        conv_3 = self.conv_3(conv_2)
        conv_3_copy = conv_3
        # sideout_3 = self.side_block3(conv_3)
        # fuse_3 = self.fuse_block3(conv_3, prio_features_list3)
        # out_3 = self.fuse_img_prio_conv3(torch.cat([sideout_3, fuse_3[0], fuse_3[1], fuse_3[2], fuse_3[3], fuse_3[4]], 1))
        # out_3 = F.interpolate(out_3, size=(x.shape[2], x.shape[3]), mode='bilinear')
        conv_3, index_3 = self.pool(conv_3)

        conv_4 = self.conv_4(conv_3)
        conv_4_copy = conv_4
        # sideout_4 = self.side_block4(conv_4)
        # fuse_4 = self.fuse_block4(conv_4, prio_features_list4)
        # # out_4 = self.fuse_img_prio_conv4(torch.cat([sideout_4, fuse_4[0], fuse_4[1], fuse_4[2], fuse_4[3], fuse_4[4]], 1))
        # # out_4 = F.interpolate(out_4, size=(x.shape[2], x.shape[3]), mode='bilinear')
        # # # class_prob = torch.cat([fuse_4[0], fuse_4[1], fuse_4[2], fuse_4[3], fuse_4[4]], 1)
        # # # out_4 = (sideout_4[:,0:5,:,:] + class_prob)/2
        # # # out_4 = torch.cat([out_4, sideout_4[:,5,:,:].unsqueeze(1)], 1)
        # # # # #background保持原先的预测
        # # # pre_sideout_4 = torch.argmax(sideout_4, 1).unsqueeze(1).repeat(1, 6, 1, 1)
        # # # background_index = ( pre_sideout_4== 5)
        # # # out_4[background_index] = sideout_4[background_index]
        # out_4_1 = self.fuse_img_prio_conv4_c1(torch.cat([sideout_4[:,0,:,:].unsqueeze(1), fuse_4[0]], 1))
        # out_4_2 = self.fuse_img_prio_conv4_c2(torch.cat([sideout_4[:,1,:,:].unsqueeze(1), fuse_4[1]], 1))
        # out_4_3 = self.fuse_img_prio_conv4_c3(torch.cat([sideout_4[:,2,:,:].unsqueeze(1), fuse_4[2]], 1))
        # out_4_4 = self.fuse_img_prio_conv4_c4(torch.cat([sideout_4[:,3,:,:].unsqueeze(1), fuse_4[3]], 1))
        # out_4_5 = self.fuse_img_prio_conv4_c5(torch.cat([sideout_4[:,4,:,:].unsqueeze(1), fuse_4[4]], 1))
        # out_4_6 = sideout_4[:,5,:,:].unsqueeze(1)
        # out_4 = torch.cat([out_4_1,out_4_2,out_4_3,out_4_4,out_4_5,out_4_6],1)
        # out_4 = F.interpolate(out_4, size=(x.shape[2], x.shape[3]), mode='bilinear')
        conv_4, index_4 = self.pool(conv_4)

        conv_5 = self.conv_5(conv_4)
        conv_5_copy = conv_5
        conv_5, index_5 = self.pool(conv_5)

        #decoder
        conv_5 = self.unpool(conv_5,index_5,output_size=conv_5_copy.shape)
        conv_5 = self.deconv_5(conv_5)

        conv_4 = self.unpool(conv_5,index_4,output_size=conv_4_copy.shape)
        conv_4 = self.deconv_4(conv_4)

        conv_3 = self.unpool(conv_4,index_3,output_size=conv_3_copy.shape)
        conv_3 = self.deconv_3(conv_3)

        conv_2 = self.unpool(conv_3,index_2,output_size=conv_2_copy.shape)
        conv_2 = self.deconv_2(conv_2)

        conv_1 = self.unpool(conv_2,index_1,output_size=conv_1_copy.shape)
        conv_1 = self.deconv_1_half(conv_1)

        conv_1_copy = conv_1
        out_PRD = self.deconv_1_classifier(conv_1)

        fuse_5 = prio_features_list5
        print('1', fuse_5[0].shape)
        fuse_5[0] = F.interpolate(fuse_5[0], size=(conv_1.shape[2], conv_1.shape[3]), mode='bilinear')
        fuse_5[1] = F.interpolate(fuse_5[1], size=(conv_1.shape[2], conv_1.shape[3]), mode='bilinear')
        fuse_5[2] = F.interpolate(fuse_5[2], size=(conv_1.shape[2], conv_1.shape[3]), mode='bilinear')
        fuse_5[3] = F.interpolate(fuse_5[3], size=(conv_1.shape[2], conv_1.shape[3]), mode='bilinear')
        fuse_5[4] = F.interpolate(fuse_5[4], size=(conv_1.shape[2], conv_1.shape[3]), mode='bilinear')
        print('2', fuse_5[0].shape)
        # print(out_PRD.shape, fuse_5[0].shape)
        out_5_1 = self.fuse_img_prio_conv5_c1(torch.cat([out_PRD[:,0,:,:].unsqueeze(1), fuse_5[0]], 1))
        out_5_2 = self.fuse_img_prio_conv5_c2(torch.cat([out_PRD[:,1,:,:].unsqueeze(1), fuse_5[1]], 1))
        out_5_3 = self.fuse_img_prio_conv5_c3(torch.cat([out_PRD[:,2,:,:].unsqueeze(1), fuse_5[2]], 1))
        out_5_4 = self.fuse_img_prio_conv5_c4(torch.cat([out_PRD[:,3,:,:].unsqueeze(1), fuse_5[3]], 1))
        out_5_5 = self.fuse_img_prio_conv5_c5(torch.cat([out_PRD[:,4,:,:].unsqueeze(1), fuse_5[4]], 1))
        out_5_6 = out_PRD[:,5,:,:].unsqueeze(1)
        out_Guiding = torch.cat([out_5_1,out_5_2,out_5_3,out_5_4,out_5_5,out_5_6],1)
        # out_Guiding = F.interpolate(out_Guiding, size=(x.shape[2], x.shape[3]), mode='bilinear')
        out_CGR = torch.cat([fuse_5[0],fuse_5[1],fuse_5[2],fuse_5[3],fuse_5[4],conv_1[:,5,:,:].unsqueeze(1)],1)
        out_CGR = F.interpolate(out_CGR, size=(x.shape[2], x.shape[3]), mode='bilinear')

        return out_CGR, out_Guiding, out_PRD

    def forward(self, x, prio_features_list1, prio_features_list2, prio_features_list3, prio_features_list4, prio_features_list5, forward_flag):
        if forward_flag == 1:
            return self.forward_first(x)
        if forward_flag == 2:
            return self.forward_second(x, prio_features_list1, prio_features_list2, prio_features_list3, prio_features_list4, prio_features_list5)

class SegNet_prio(nn.Module):
    def __init__(self, input_channels, output_channels, pre_train=True):
        super(SegNet_prio,self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        self.CARB = CARB_Block(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):
                # m.weight.data.normal_(0, 0.2)
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
        if pre_train==True:
            if input_channels == 3:
                print('SegNet param init: input channels == 3, IMAGE, init weight by Pascal model')
                SegNet_load_Pascal_model(self, 'D:\\Code\\LULC\\Hed_Seg\\pretrained\\SegNet\\segnet_pascal_params.npy')
            else:
                print('SegNet param init: Using xavier_normal')
        else:
            print('SegNet param init: Using xavier_normal')
        

    def forward(self, x):
        #encoder
        conv_1 = self.conv_1(x)
        conv_1p, index_1 = self.pool(conv_1)
        #print(conv_1.shape)
        conv_2 = self.conv_2(conv_1p)
        conv_2p, index_2 = self.pool(conv_2)
        #print(conv_2.shape)
        conv_3 = self.conv_3(conv_2p)
        conv_3p, index_3 = self.pool(conv_3)
        #print(conv_3.shape)
        conv_4 = self.conv_4(conv_3p)
        conv_4p, index_4 = self.pool(conv_4)
        #print(conv_4.shape)
        conv_5 = self.conv_5(conv_4p)
        # conv_5p, index_5 = self.pool(conv_5)

        conv_out = self.CARB(conv_5)
        return conv_out

def SegNet_load_Pascal_model(model, model_filename):
    new_params = np.load(model_filename, allow_pickle=True, encoding='bytes')
    model_dict = model.state_dict()
    premodel_dict = new_params[0]
    premodel_list = []
    for key, value in premodel_dict.items():
        temp_dict = {'key':key,'value':value}
        premodel_list.append(temp_dict)
    param_layer = 0
    for key in model_dict:
        if 'deconv_1_classifier' in key or 'CARB.bottlenect.residual_function' in key:
            break
        if 'run' in key or 'num' in key or 'img' in key:
            continue
        else:
            pre_k = premodel_list[param_layer]['key']
            pre_v = premodel_list[param_layer]['value']
            if 'bn' in str(pre_k):
                pre_v = np.reshape(pre_v,[-1])
            pre_v = torch.from_numpy(pre_v)
            assert model_dict[key].shape == pre_v.shape
            model_dict[key] = pre_v
            #print('     set SegNet model %s layer param by premodel %s layer param'%(key, pre_k))
            param_layer += 1

    model.load_state_dict(model_dict)
    return model

def add_conv_channels(model, premodel, conv_num):
    model_dict = model.state_dict()
    premodel_dict = premodel.state_dict()
    # for key, value in premodel_dict.items():
    #     # if b'bn' not in key:
    #     print(key, value.shape)
    for i in range(conv_num[0]):
        conv = torch.FloatTensor(64,1,3,3).cuda()
        nn.init.xavier_normal_(conv)

        orginal1 = premodel_dict['conv_1.0.weight']
        new = torch.cat([orginal1,conv],1)
        premodel_dict['conv_1.0.weight'] = new
    for key, value in model_dict.items():
        if key not in premodel_dict:
            premodel_dict[key] = value
    model.load_state_dict(premodel_dict)
    print('set model with predect model, add channel is ',conv_num)
    return model

def add_pre_model(model, premodel):
    model_dict = model.state_dict()
    premodel_dict = premodel.state_dict()
    
    model_dict_copy = model_dict.copy()
    for key, value in model_dict.items():
        for key_pre, value_pre in premodel_dict.items():
            if key == key_pre:
                model_dict_copy[key] = value_pre
                continue
    model.load_state_dict(model_dict_copy)
    print('set model with pretrained SegNet model')
    return model