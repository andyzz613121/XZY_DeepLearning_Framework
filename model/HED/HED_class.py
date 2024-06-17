import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

class HED(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(HED,self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv_3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
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
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
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
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
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


        self.deconv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)


        self.sideout_conv1 = nn.Conv2d(64, out_channels, 1)
        self.sideout_conv2 = nn.Conv2d(128, out_channels, 1)
        self.sideout_conv3 = nn.Conv2d(256, out_channels, 1)
        self.sideout_conv4 = nn.Conv2d(512, out_channels, 1)
        self.sideout_conv5 = nn.Conv2d(512, out_channels, 1)
        self.sideout_fuse = nn.Conv2d(5*out_channels, out_channels, 1)

        self.conv_imgdsm1 = nn.Conv2d(2*out_channels, out_channels,kernel_size=1)
        self.conv_imgdsm2 = nn.Conv2d(2*out_channels, out_channels,kernel_size=1)
        self.conv_imgdsm3 = nn.Conv2d(2*out_channels, out_channels,kernel_size=1)
        self.conv_imgdsm4 = nn.Conv2d(2*out_channels, out_channels,kernel_size=1)
        self.conv_imgdsm5 = nn.Conv2d(2*out_channels, out_channels,kernel_size=1)
        self.conv_imgdsmfuse = nn.Conv2d(2*out_channels, out_channels,kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):
                # m.weight.data.normal_(0, 0.2)
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
        
        if input_channels == 3:
            HED_load_premodel(self,'pretrained\\FCN\\fcn32s-heavy-pascal.npy')

    def forward(self, x):

        img_h = x.size(2)
        img_w = x.size(3)

        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        x4 = self.conv_4(x3)
        x5 = self.conv_5(x4)
        # print(x5.shape, x4.shape, x3.shape, x2.shape, x1.shape)
        ###############################################################################################################33
        #ce
        sideout_1 = self.sideout_conv1(x1)
        sideout_2_upsample = F.interpolate(self.sideout_conv2(x2), size=(img_h, img_w), mode='bilinear')
        sideout_3_upsample = F.interpolate(self.sideout_conv3(x3), size=(img_h, img_w), mode='bilinear')
        sideout_4_upsample = F.interpolate(self.sideout_conv4(x4), size=(img_h, img_w), mode='bilinear')
        sideout_5_upsample = F.interpolate(self.sideout_conv5(x5), size=(img_h, img_w), mode='bilinear')
        sideout_concat = self.sideout_fuse(torch.cat((sideout_1,sideout_2_upsample,sideout_3_upsample,sideout_4_upsample,sideout_5_upsample), 1))
        #sideout_plus = sideout_1 + sideout_2_upsample + sideout_3_upsample + sideout_4_upsample + sideout_5_upsample
        #print('sideout_concat')
        ###############################################################################################################33
        
        sideout1 = torch.sigmoid(sideout_1)
        sideout2 = torch.sigmoid(sideout_2_upsample)
        sideout3 = torch.sigmoid(sideout_3_upsample)
        sideout4 = torch.sigmoid(sideout_4_upsample)
        sideout5 = torch.sigmoid(sideout_5_upsample)
        sideoutcat = torch.sigmoid(sideout_concat)

        edge = [sideout1, sideout2, sideout3, sideout4, sideout5, sideoutcat]
        feature = [x1,x2,x3,x4,x5]
        return feature, edge


def HED_load_premodel(model, premodel_filename):
    new_params = np.load(premodel_filename, allow_pickle=True, encoding='bytes')
    model_dict = model.state_dict()
    premodel_dict = new_params[0]
    premodel_list = []
    for key, value in premodel_dict.items():
        temp_dict = {'key':key,'value':value}
        premodel_list.append(temp_dict)
    param_layer = 0

    for key in model_dict:
        try:
            pre_k = premodel_list[param_layer]['key']
            pre_v = premodel_list[param_layer]['value']
            pre_v = torch.from_numpy(pre_v)
            assert model_dict[key].shape == pre_v.shape
            model_dict[key] = pre_v
            #print('     set FCN model %s layer param by Pascal premodel %s layer param'%(key, pre_k))
            param_layer += 1
        except:
            continue
        # if 'deconv' in key:
        #     break
        # if 'conv' in key:
        #     pre_k = premodel_list[param_layer]['key']
        #     pre_v = premodel_list[param_layer]['value']
        #     pre_v = torch.from_numpy(pre_v)
        #     assert model_dict[key].shape == pre_v.shape
        #     model_dict[key] = pre_v
        #     #print('     set FCN model %s layer param by Pascal premodel %s layer param'%(key, pre_k))
        #     param_layer += 1

    model.load_state_dict(model_dict)

    print('HED init weight by %s model'%premodel_filename)
    return model

def add_conv_channels(model, premodel, conv_num):
    model_dict = model.state_dict()

    premodel_dict = premodel.state_dict()
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
    print('set model with pretrained model, add channel is ',conv_num)
    return model








def main():
    model = HED(3)


if __name__ == '__main__':
    main()    


