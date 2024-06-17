import re
import torch
from torch import nn
import numpy as np
from model.CARB import CARB_Block
import sys
base_path = '..\\Laplace\\'
sys.path.append(base_path)

class SegNet_AW(nn.Module):
    def __init__(self, input_channels, output_channels, pre_train=True):
        super(SegNet_AW,self).__init__()
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

        #CARB & Multiresult_Fusion
        # self.Class_SELayer = Class_SELayer(64, 6)

        self.CARB0 = CARB_Block(6, 1)
        self.CARB1 = CARB_Block(6, 1)
        self.CARB2 = CARB_Block(6, 1)
        self.CARB3 = CARB_Block(6, 1)
        self.CARB4 = CARB_Block(6, 1)
        self.CARB5 = CARB_Block(6, 1)

        self.auto_weight0 = nn.Parameter(torch.ones([6], requires_grad=True))
        self.auto_weight1 = nn.Parameter(torch.ones([6], requires_grad=True))
        self.auto_weight2 = nn.Parameter(torch.ones([6], requires_grad=True))
        self.auto_weight3 = nn.Parameter(torch.ones([6], requires_grad=True))
        self.auto_weight4 = nn.Parameter(torch.ones([6], requires_grad=True))
        self.auto_weight5 = nn.Parameter(torch.ones([6], requires_grad=True))

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
    
    def encoder(self, x):
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

        upsample_paras = [index_5, index_4, index_3, index_2, index_1, conv_5_copy, conv_4_copy, conv_3_copy, conv_2_copy, conv_1_copy]
        
        return conv_5, upsample_paras

    def decoder(self, x, upsample_paras):
        index_5 = upsample_paras[0]
        index_4 = upsample_paras[1]
        index_3 = upsample_paras[2]
        index_2 = upsample_paras[3]
        index_1 = upsample_paras[4]
        conv_5_copy = upsample_paras[5]
        conv_4_copy = upsample_paras[6]
        conv_3_copy = upsample_paras[7]
        conv_2_copy = upsample_paras[8]
        conv_1_copy = upsample_paras[9]
        
        #decoder
        conv_5 = self.unpool(x,index_5,output_size=conv_5_copy.shape)
        conv_5 = self.deconv_5(conv_5)

        conv_4 = self.unpool(conv_5,index_4,output_size=conv_4_copy.shape)
        conv_4 = self.deconv_4(conv_4)

        conv_3 = self.unpool(conv_4,index_3,output_size=conv_3_copy.shape)
        conv_3 = self.deconv_3(conv_3)

        conv_2 = self.unpool(conv_3,index_2,output_size=conv_2_copy.shape)
        conv_2 = self.deconv_2(conv_2)
    
        conv_1 = self.unpool(conv_2,index_1,output_size=conv_1_copy.shape)
        # if confuse_matrix is not None:
        #     conv_1 = self.Class_SELayer(conv_1, confuse_matrix)
        conv_1 = self.deconv_1(conv_1)
        return conv_1

    def forward(self, x):
        with torch.no_grad():
            conv_5, upsample_paras = self.encoder(x)
            outputs = self.decoder(conv_5, upsample_paras)
        return outputs

        

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
        if 'run' in key or 'num' in key or 'auto' in key:
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

def freeze(model):
    for key, value in model.named_parameters():
        if key.startswith('conv_') or key.startswith('deconv_'):
            value.requires_grad = False
    return model
