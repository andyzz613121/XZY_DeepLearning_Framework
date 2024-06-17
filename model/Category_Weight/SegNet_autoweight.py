from numpy.core.defchararray import decode
import torch
from torch import nn
import numpy as np
from model import SE_Layer
import sys
base_path = 'D:\\Code\\LULC\\Laplace\\'
sys.path.append(base_path)
from Auto_Weights.JM_Distance import auto_weight, JMD_Loss

class AW_SegNet(nn.Module):
    def __init__(self, input_channels, output_channels, pre_train=True):
        super(AW_SegNet,self).__init__()
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
        
        self.AW = Auto_weight_dict()
        
        self.CARB0 = CARB_Block(36, 6)
        # self.CARB1 = CARB_Block(6, 1)
        # self.CARB2 = CARB_Block(6, 1)
        # self.CARB3 = CARB_Block(6, 1)
        # self.CARB4 = CARB_Block(6, 1)

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

        return conv_1, conv_2, conv_3, conv_4, conv_5, upsample_paras

    def decoder(self, conv_1, conv_2, conv_3, conv_4, conv_5, upsample_paras):
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
        conv_5 = self.unpool(conv_5,index_5,output_size=conv_5_copy.shape)
        conv_5 = self.deconv_5(conv_5)

        conv_4 = self.unpool(conv_5,index_4,output_size=conv_4_copy.shape)
        conv_4 = self.deconv_4(conv_4)

        conv_3 = self.unpool(conv_4,index_3,output_size=conv_3_copy.shape)
        conv_3 = self.deconv_3(conv_3)

        conv_2 = self.unpool(conv_3,index_2,output_size=conv_2_copy.shape)
        conv_2 = self.deconv_2(conv_2)

        conv_1 = self.unpool(conv_2,index_1,output_size=conv_1_copy.shape)
        conv_1 = self.deconv_1(conv_1)

        return conv_1

    def forward(self, x):
        conv1_weighted_for_class = []
        conv2_weighted_for_class = []
        conv3_weighted_for_class = []
        conv4_weighted_for_class = []
        conv5_weighted_for_class = []

        #encoder
        conv_1, conv_2, conv_3, conv_4, conv_5, upsample_paras = self.encoder(x)
        
        #feature maps weighting
        conv1_weighted_for_class.append(self.AW.Auto_weights_dict_conv1_class1(conv_1))
        conv1_weighted_for_class.append(self.AW.Auto_weights_dict_conv1_class2(conv_1))
        conv1_weighted_for_class.append(self.AW.Auto_weights_dict_conv1_class3(conv_1))
        conv1_weighted_for_class.append(self.AW.Auto_weights_dict_conv1_class4(conv_1))
        conv1_weighted_for_class.append(self.AW.Auto_weights_dict_conv1_class5(conv_1))

        conv2_weighted_for_class.append(self.AW.Auto_weights_dict_conv2_class1(conv_2))
        conv2_weighted_for_class.append(self.AW.Auto_weights_dict_conv2_class2(conv_2))
        conv2_weighted_for_class.append(self.AW.Auto_weights_dict_conv2_class3(conv_2))
        conv2_weighted_for_class.append(self.AW.Auto_weights_dict_conv2_class4(conv_2))
        conv2_weighted_for_class.append(self.AW.Auto_weights_dict_conv2_class5(conv_2))

        conv3_weighted_for_class.append(self.AW.Auto_weights_dict_conv3_class1(conv_3))
        conv3_weighted_for_class.append(self.AW.Auto_weights_dict_conv3_class2(conv_3))
        conv3_weighted_for_class.append(self.AW.Auto_weights_dict_conv3_class3(conv_3))
        conv3_weighted_for_class.append(self.AW.Auto_weights_dict_conv3_class4(conv_3))
        conv3_weighted_for_class.append(self.AW.Auto_weights_dict_conv3_class5(conv_3))

        conv4_weighted_for_class.append(self.AW.Auto_weights_dict_conv4_class1(conv_4))
        conv4_weighted_for_class.append(self.AW.Auto_weights_dict_conv4_class2(conv_4))
        conv4_weighted_for_class.append(self.AW.Auto_weights_dict_conv4_class3(conv_4))
        conv4_weighted_for_class.append(self.AW.Auto_weights_dict_conv4_class4(conv_4))
        conv4_weighted_for_class.append(self.AW.Auto_weights_dict_conv4_class5(conv_4))

        conv5_weighted_for_class.append(self.AW.Auto_weights_dict_conv5_class1(conv_5))
        conv5_weighted_for_class.append(self.AW.Auto_weights_dict_conv5_class2(conv_5))
        conv5_weighted_for_class.append(self.AW.Auto_weights_dict_conv5_class3(conv_5))
        conv5_weighted_for_class.append(self.AW.Auto_weights_dict_conv5_class4(conv_5))
        conv5_weighted_for_class.append(self.AW.Auto_weights_dict_conv5_class5(conv_5))

        #decoder
        original = self.decoder(conv_1, conv_2, conv_3, conv_4, conv_5, upsample_paras)
        class_1 = self.decoder(conv1_weighted_for_class[0], conv2_weighted_for_class[0], conv3_weighted_for_class[0], conv4_weighted_for_class[0], conv5_weighted_for_class[0], upsample_paras)
        class_2 = self.decoder(conv1_weighted_for_class[1], conv2_weighted_for_class[1], conv3_weighted_for_class[1], conv4_weighted_for_class[1], conv5_weighted_for_class[1], upsample_paras)
        class_3 = self.decoder(conv1_weighted_for_class[2], conv2_weighted_for_class[2], conv3_weighted_for_class[2], conv4_weighted_for_class[2], conv5_weighted_for_class[2], upsample_paras)
        class_4 = self.decoder(conv1_weighted_for_class[3], conv2_weighted_for_class[3], conv3_weighted_for_class[3], conv4_weighted_for_class[3], conv5_weighted_for_class[3], upsample_paras)
        class_5 = self.decoder(conv1_weighted_for_class[4], conv2_weighted_for_class[4], conv3_weighted_for_class[4], conv4_weighted_for_class[4], conv5_weighted_for_class[4], upsample_paras)
        out_Guiding = torch.cat([class_1,class_2,class_3,class_4,class_5,original],1)
        out_Guiding = self.CARB0(out_Guiding)
        return out_Guiding, conv1_weighted_for_class, conv2_weighted_for_class, conv3_weighted_for_class, conv4_weighted_for_class, conv5_weighted_for_class

class Auto_weight_dict(nn.Module):
    def __init__(self):
        super(Auto_weight_dict,self).__init__()
        self.Auto_weights_dict_conv1_class1 = auto_weight(64)
        self.Auto_weights_dict_conv1_class2 = auto_weight(64)
        self.Auto_weights_dict_conv1_class3 = auto_weight(64)
        self.Auto_weights_dict_conv1_class4 = auto_weight(64)
        self.Auto_weights_dict_conv1_class5 = auto_weight(64)

        self.Auto_weights_dict_conv2_class1 = auto_weight(128)
        self.Auto_weights_dict_conv2_class2 = auto_weight(128)
        self.Auto_weights_dict_conv2_class3 = auto_weight(128)
        self.Auto_weights_dict_conv2_class4 = auto_weight(128)
        self.Auto_weights_dict_conv2_class5 = auto_weight(128)

        self.Auto_weights_dict_conv3_class1 = auto_weight(256)
        self.Auto_weights_dict_conv3_class2 = auto_weight(256)
        self.Auto_weights_dict_conv3_class3 = auto_weight(256)
        self.Auto_weights_dict_conv3_class4 = auto_weight(256)
        self.Auto_weights_dict_conv3_class5 = auto_weight(256)

        self.Auto_weights_dict_conv4_class1 = auto_weight(512)
        self.Auto_weights_dict_conv4_class2 = auto_weight(512)
        self.Auto_weights_dict_conv4_class3 = auto_weight(512)
        self.Auto_weights_dict_conv4_class4 = auto_weight(512)
        self.Auto_weights_dict_conv4_class5 = auto_weight(512)

        self.Auto_weights_dict_conv5_class1 = auto_weight(512)
        self.Auto_weights_dict_conv5_class2 = auto_weight(512)
        self.Auto_weights_dict_conv5_class3 = auto_weight(512)
        self.Auto_weights_dict_conv5_class4 = auto_weight(512)
        self.Auto_weights_dict_conv5_class5 = auto_weight(512)

        # self.Auto_weights_dict = {'conv1':[], 'conv2':[], 'conv3':[], 'conv4':[], 'conv5':[]}
        # for classes in range(5):
        #     aw_tmp = auto_weight(64)
        #     self.Auto_weights_dict['conv1'].append(aw_tmp)
        # for classes in range(5):
        #     aw_tmp = auto_weight(128)
        #     self.Auto_weights_dict['conv2'].append(aw_tmp)
        # for classes in range(5):
        #     aw_tmp = auto_weight(256)
        #     self.Auto_weights_dict['conv3'].append(aw_tmp)
        # for classes in range(5):
        #     aw_tmp = auto_weight(512)
        #     self.Auto_weights_dict['conv4'].append(aw_tmp)
        # for classes in range(5):
        #     aw_tmp = auto_weight(512)
        #     self.Auto_weights_dict['conv5'].append(aw_tmp)
        

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