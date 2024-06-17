import imp
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from model.Self_Module.CARB import CARB_Block

class SegNet_IMG(nn.Module):
    def __init__(self, input_channels, output_channels, pre_train=True):
        super(SegNet_IMG,self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        #decoder
        self.deconv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.unpool = nn.MaxUnpool2d(2,2)

        #imgdsm_fuse
        self.imgguiding_fuse = nn.Sequential(
            nn.Conv2d(2*output_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
        )
        
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
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):
                # m.weight.data.normal_(0, 0.2)
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
        if pre_train==True:
            if input_channels == 3:
                print('SegNet param init: input channels == 3, IMAGE, init weight by Pascal model')
                SegNet_load_Pascal_model(self, 'D:\\Code\\LULC\\Hed_Seg\\pretrained\\SegNet\\segnet_pascal_params.npy')
            else:
                print('SegNet param init: Using xavier_normal')
        else:
            print('SegNet param init: Using xavier_normal')
        

    def forward(self, x, prio_features_list):
        #encoder
        conv_1 = self.conv_1(x)
        conv_1_copy = conv_1
        conv_1, index_1 = self.pool(conv_1)

        #print(conv_1.shape)
        conv_2 = self.conv_2(conv_1)
        conv_2_copy = conv_2
        conv_2, index_2 = self.pool(conv_2)
        #print(conv_2.shape)
        conv_3 = self.conv_3(conv_2)
        conv_3_copy = conv_3
        conv_3, index_3 = self.pool(conv_3)
        #print(conv_3.shape)
        conv_4 = self.conv_4(conv_3)
        conv_4_copy = conv_4
        conv_4, index_4 = self.pool(conv_4)
        #print(conv_4.shape)
        conv_5 = self.conv_5(conv_4)
        conv_5_copy = conv_5
        conv_5, index_5 = self.pool(conv_5)
        #print(conv_5.shape)
        #print('copy',conv_5_copy.shape)
        #decoder
        conv_5 = self.unpool(conv_5,index_5,output_size=conv_5_copy.shape)
        conv_5 = self.deconv_5(conv_5)
        #print(conv_5.shape)
        conv_4 = self.unpool(conv_5,index_4,output_size=conv_4_copy.shape)
        conv_4 = self.deconv_4(conv_4)
        #print(conv_4.shape)
        conv_3 = self.unpool(conv_4,index_3,output_size=conv_3_copy.shape)
        conv_3 = self.deconv_3(conv_3)
        #print(conv_3.shape)
        conv_2 = self.unpool(conv_3,index_2,output_size=conv_2_copy.shape)
        conv_2 = self.deconv_2(conv_2)
        #print(conv_2.shape)
        conv_1 = self.unpool(conv_2,index_1,output_size=conv_1_copy.shape)
        conv_1 = self.deconv_1(conv_1)
        #print(conv_1.shape)

        fuse_5 = prio_features_list

        fuse_5[0] = F.interpolate(fuse_5[0], size=(conv_1.shape[2], conv_1.shape[3]), mode='bilinear')
        fuse_5[1] = F.interpolate(fuse_5[1], size=(conv_1.shape[2], conv_1.shape[3]), mode='bilinear')
        fuse_5[2] = F.interpolate(fuse_5[2], size=(conv_1.shape[2], conv_1.shape[3]), mode='bilinear')
        fuse_5[3] = F.interpolate(fuse_5[3], size=(conv_1.shape[2], conv_1.shape[3]), mode='bilinear')
        fuse_5[4] = F.interpolate(fuse_5[4], size=(conv_1.shape[2], conv_1.shape[3]), mode='bilinear')
        
        out_5_1 = self.fuse_img_prio_conv5_c1(torch.cat([conv_1[:,0,:,:].unsqueeze(1), fuse_5[0]], 1))
        out_5_2 = self.fuse_img_prio_conv5_c2(torch.cat([conv_1[:,1,:,:].unsqueeze(1), fuse_5[1]], 1))
        out_5_3 = self.fuse_img_prio_conv5_c3(torch.cat([conv_1[:,2,:,:].unsqueeze(1), fuse_5[2]], 1))
        out_5_4 = self.fuse_img_prio_conv5_c4(torch.cat([conv_1[:,3,:,:].unsqueeze(1), fuse_5[3]], 1))
        out_5_5 = self.fuse_img_prio_conv5_c5(torch.cat([conv_1[:,4,:,:].unsqueeze(1), fuse_5[4]], 1))
        out_5_6 = conv_1[:,5,:,:].unsqueeze(1)
        out_Guiding = torch.cat([out_5_1,out_5_2,out_5_3,out_5_4,out_5_5,out_5_6],1)
        return out_Guiding

class SegNet_ADD(nn.Module):
    def __init__(self, input_channels, output_channels, pre_train=True):
        super(SegNet_ADD,self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        #decoder
        self.deconv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.unpool = nn.MaxUnpool2d(2,2)

        self.CARB = CARB_Block(6, 1)
        

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
        conv_1_copy = conv_1
        conv_1, index_1 = self.pool(conv_1)

        #print(conv_1.shape)
        conv_2 = self.conv_2(conv_1)
        conv_2_copy = conv_2
        conv_2, index_2 = self.pool(conv_2)
        #print(conv_2.shape)
        conv_3 = self.conv_3(conv_2)
        conv_3_copy = conv_3
        conv_3, index_3 = self.pool(conv_3)
        #print(conv_3.shape)
        conv_4 = self.conv_4(conv_3)
        conv_4_copy = conv_4
        conv_4, index_4 = self.pool(conv_4)
        #print(conv_4.shape)
        conv_5 = self.conv_5(conv_4)
        conv_5_copy = conv_5
        conv_5, index_5 = self.pool(conv_5)
        #print(conv_5.shape)
        #print('copy',conv_5_copy.shape)
        #decoder
        conv_5 = self.unpool(conv_5,index_5,output_size=conv_5_copy.shape)
        conv_5 = self.deconv_5(conv_5)
        #print(conv_5.shape)
        conv_4 = self.unpool(conv_5,index_4,output_size=conv_4_copy.shape)
        conv_4 = self.deconv_4(conv_4)
        #print(conv_4.shape)
        conv_3 = self.unpool(conv_4,index_3,output_size=conv_3_copy.shape)
        conv_3 = self.deconv_3(conv_3)
        #print(conv_3.shape)
        conv_2 = self.unpool(conv_3,index_2,output_size=conv_2_copy.shape)
        conv_2 = self.deconv_2(conv_2)
        #print(conv_2.shape)
        conv_1 = self.unpool(conv_2,index_1,output_size=conv_1_copy.shape)
        conv_1 = self.deconv_1(conv_1)

        conv_out = self.CARB(conv_1)
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
        if 'deconv_1.1.running_mean' in key:
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