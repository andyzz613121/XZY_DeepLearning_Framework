import re
import torch
from torch import nn
import numpy as np

class SegNet_skeleton(nn.Module):
    def __init__(self, input_channels, output_channels, pre_train=True):
        super(SegNet_skeleton,self).__init__()
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


        self.skeleton_deconv_1 = nn.Sequential(
            nn.Linear(output_channels*output_channels, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64, bias=True),
            nn.ReLU(inplace=True)
        )
        self.skeleton_deconv_2 = nn.Sequential(
            nn.Linear(64, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=True),
            nn.ReLU(inplace=True)
        )
        self.skeleton_deconv_3 = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(inplace=True)
        )
        self.skeleton_deconv_4 = nn.Sequential(
            nn.Linear(256, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
        )
        self.skeleton_deconv_5 = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
        )
        self.skeleton_conv_5 = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
        )
        self.skeleton_conv_4 = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
        )
        self.skeleton_conv_3 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
        )
        self.skeleton_conv_2 = nn.Sequential(
            nn.Linear(128, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),
        )
        self.skeleton_conv_1 = nn.Sequential(
            nn.Linear(64, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 7, bias=True),
            nn.ReLU(inplace=True),
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
                SegNet_load_Pascal_model(self, 'pretrained\\SegNet\\segnet_pascal_params.npy')
            else:
                print('SegNet param init: Using xavier_normal')
        else:
            print('SegNet param init: Using xavier_normal')

    def forward(self, x, accuracy_list):
        #skeleton
        weight_dc1 = self.skeleton_deconv_1(accuracy_list)
        weight_dc2 = self.skeleton_deconv_2(weight_dc1)
        weight_dc3 = self.skeleton_deconv_3(weight_dc2)
        weight_dc4 = self.skeleton_deconv_4(weight_dc3)
        weight_dc5 = self.skeleton_deconv_5(weight_dc4)
        weight_c5 = self.skeleton_conv_5(weight_dc5)
        weight_c4 = self.skeleton_conv_4(weight_c5)
        weight_c3 = self.skeleton_conv_3(weight_c4)
        weight_c2 = self.skeleton_conv_2(weight_c3)
        weight_c1 = self.skeleton_conv_1(weight_c2)

        #encoder
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(x, weight_c1).permute(0, 3, 1, 2).contiguous()
        conv_1 = self.conv_1(x)
        conv_1_copy = conv_1
        conv_1, index_1 = self.pool(conv_1)

        conv_1 = conv_1.permute(0, 2, 3, 1).contiguous()
        conv_1 = torch.mul(conv_1, weight_c2).permute(0, 3, 1, 2).contiguous()
        conv_2 = self.conv_2(conv_1)
        conv_2_copy = conv_2
        conv_2, index_2 = self.pool(conv_2)

        conv_2 = conv_2.permute(0, 2, 3, 1).contiguous()
        conv_2 = torch.mul(conv_2, weight_c3).permute(0, 3, 1, 2).contiguous()
        conv_3 = self.conv_3(conv_2)
        conv_3_copy = conv_3
        conv_3, index_3 = self.pool(conv_3)

        conv_3 = conv_3.permute(0, 2, 3, 1).contiguous()
        conv_3 = torch.mul(conv_3, weight_c4).permute(0, 3, 1, 2).contiguous()
        conv_4 = self.conv_4(conv_3)
        conv_4_copy = conv_4
        conv_4, index_4 = self.pool(conv_4)

        conv_4 = conv_4.permute(0, 2, 3, 1).contiguous()
        conv_4 = torch.mul(conv_4, weight_c5).permute(0, 3, 1, 2).contiguous()
        conv_5 = self.conv_5(conv_4)
        conv_5_copy = conv_5
        conv_5, index_5 = self.pool(conv_5)

        #decoder
        conv_5 = conv_5.permute(0, 2, 3, 1).contiguous()
        conv_5 = torch.mul(conv_5, weight_dc5).permute(0, 3, 1, 2).contiguous()
        conv_5 = self.unpool(conv_5,index_5,output_size=conv_5_copy.shape)
        conv_5 = self.deconv_5(conv_5)
        
        conv_4 = conv_4.permute(0, 2, 3, 1).contiguous()
        conv_4 = torch.mul(conv_4, weight_dc4).permute(0, 3, 1, 2).contiguous()
        conv_4 = self.unpool(conv_5,index_4,output_size=conv_4_copy.shape)
        conv_4 = self.deconv_4(conv_4)
        
        conv_3 = conv_3.permute(0, 2, 3, 1).contiguous()
        conv_3 = torch.mul(conv_3, weight_dc3).permute(0, 3, 1, 2).contiguous()
        conv_3 = self.unpool(conv_4,index_3,output_size=conv_3_copy.shape)
        conv_3 = self.deconv_3(conv_3)
        
        conv_2 = conv_2.permute(0, 2, 3, 1).contiguous()
        conv_2 = torch.mul(conv_2, weight_dc2).permute(0, 3, 1, 2).contiguous()
        conv_2 = self.unpool(conv_3,index_2,output_size=conv_2_copy.shape)
        conv_2 = self.deconv_2(conv_2)
        
        conv_1 = conv_1.permute(0, 2, 3, 1).contiguous()
        conv_1 = torch.mul(conv_1, weight_dc1).permute(0, 3, 1, 2).contiguous()
        conv_1 = self.unpool(conv_2,index_1,output_size=conv_1_copy.shape)
        conv_1 = self.deconv_1(conv_1)
        
        return conv_1


def cal_accuracy_list(confuse_matrix_avg):
    accuracy_list = torch.ones([6]).cuda()
    for classes in range(5):
        pos_num = confuse_matrix_avg[classes][classes]
        total_num = confuse_matrix_avg[classes].sum()
        if total_num != 0:
            accuracy = pos_num/total_num
        else:
            if pos_num != 0:
                accuracy = 0
            else:
                accuracy = 1
        accuracy_list[classes] = accuracy
    return accuracy_list

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

if __name__ == '__main__':
    from osgeo import gdal
    import sys
    base_path = '..\\XZY_DeepLearning_Framework\\'
    sys.path.append(base_path)
    from model.Self_Module.Auto_Weights.Weight_MLP import cal_confuse_matrix
    img_path = 'D:\\Code\\LULC\\Laplace\\result\\new\\Vai_NoCW_decoder\\7_ensemble_0.8809446682413624.tif'
    img_raw = gdal.Open(img_path)
    img_w = img_raw.RasterXSize
    img_h = img_raw.RasterYSize
    label_path = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai\\label_gray\\label7_gray.tif'
    label_raw = gdal.Open(label_path)

    img = np.array(img_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
    label = np.array(label_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('uint8')
    confuse = cal_confuse_matrix(torch.from_numpy(img), torch.from_numpy(label))
    cal_accuracy_list(confuse)