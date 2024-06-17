from numpy.core.numeric import outer
import torch
from torch import nn
import numpy as np
import model.priori_knowledge as priori_knowledge
import torch.nn.functional as F

class DSM_NDVI_MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(DSM_NDVI_MLP, self).__init__()
            self.fc1 = nn.Sequential(
                nn.Linear(input_size, hidden_size[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            )
            self.fc_hidden = []
            for i in range(len(hidden_size)):
                if (i+1) == len(hidden_size):
                    self.fc_hidden.append(nn.Linear(hidden_size[i], output_size))
                else:
                    self.fc_hidden.append(
                        nn.Sequential(
                            nn.Linear(hidden_size[i], hidden_size[i+1]),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=0.5)
                            ))

        def forward(self, x):
            out = self.fc1(x)
            for fc_hidden in self.fc_hidden:
                print(fc_hidden)
                out = fc_hidden(out)
            return out

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
        super(CARB_Block, self).__init__()
        self.sideout_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels)
        )
        self.bottlenect = BasicBlock(output_channels, output_channels)
        
    def forward(self, x):
        x_1 = self.sideout_conv(x)
        return self.bottlenect(x_1)

        
class Pretraining(nn.Module):
    def __init__(self, input_channels, output_channels, pre_train=True):
        super(Pretraining,self).__init__()
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
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
        )
        self.unpool = nn.MaxUnpool2d(2,2)

        self.prio = priori_knowledge.pri_knowlegde()
        
        self.CARB_block1 = CARB_Block(64*6, 64)
        self.CARB_block2 = CARB_Block(128*6, 128)
        self.CARB_block3 = CARB_Block(256*6, 256)
        self.CARB_block4 = CARB_Block(512*6, 512)
        self.CARB_block5 = CARB_Block(512*6, 512)
        
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
        

    def forward_first(self, x, label):
        #encoder
        self.dem_weight_list5 = []
        self.dis_weight_list5 = []
        self.ndvi_weight_list5 = []

        label = label.unsqueeze(1).float()

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
        out_PRD = self.deconv_1(conv_1)
        out_PRD_argmax = torch.argmax(out_PRD, 1)
        dem_weight, dis_weight, ndvi_weight = self.prio.cal_pri_knowledge_weight(out_PRD_argmax, label)
        
        self.dem_weight_list5.append(dem_weight)
        self.dis_weight_list5.append(dis_weight)
        self.ndvi_weight_list5.append(ndvi_weight)
        return out_PRD

    def forward_second(self, x, prio_features_list1, prio_features_list2, prio_features_list3, prio_features_list4, prio_features_list5):
        fuse_1 = prio_features_list1
        fuse_2 = prio_features_list2
        fuse_3 = prio_features_list3
        fuse_4 = prio_features_list4
        fuse_5 = prio_features_list5
        
        #encoder
        conv_1 = self.conv_1(x)
        conv_1_copy = conv_1
        
        conv_1 = torch.cat([conv_1, fuse_1[0], fuse_2[0], fuse_3[0], fuse_4[0], fuse_5[0]], 1)
        conv_1 = self.CARB_block1(conv_1)
        conv_1, index_1 = self.pool(conv_1)

        conv_2 = self.conv_2(conv_1)
        conv_2_copy = conv_2
        conv_2 = torch.cat([conv_2, fuse_1[1], fuse_2[1], fuse_3[1], fuse_4[1], fuse_5[1]], 1)
        conv_2 = self.CARB_block2(conv_2)
        conv_2, index_2 = self.pool(conv_2)

        conv_3 = self.conv_3(conv_2)
        conv_3_copy = conv_3
        conv_3 = torch.cat([conv_3, fuse_1[2], fuse_2[2], fuse_3[2], fuse_4[2], fuse_5[2]], 1)
        conv_3 = self.CARB_block3(conv_3)
        conv_3, index_3 = self.pool(conv_3)

        conv_4 = self.conv_4(conv_3)
        conv_4_copy = conv_4
        conv_4 = torch.cat([conv_4, fuse_1[3], fuse_2[3], fuse_3[3], fuse_4[3], fuse_5[3]], 1)
        conv_4 = self.CARB_block4(conv_4)
        conv_4, index_4 = self.pool(conv_4)

        conv_5 = self.conv_5(conv_4)
        conv_5_copy = conv_5
        conv_5 = torch.cat([conv_5, fuse_1[4], fuse_2[4], fuse_3[4], fuse_4[4], fuse_5[4]], 1)
        conv_5 = self.CARB_block5(conv_5)
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
        conv_1 = self.deconv_1(conv_1)

        conv_1_copy = conv_1

        return conv_1

    def forward(self, x, label, prio_features_list1, prio_features_list2, prio_features_list3, prio_features_list4, prio_features_list5, forward_flag):
        if forward_flag == 1:
            return self.forward_first(x, label)
        if forward_flag == 2:
            return self.forward_second(x, prio_features_list1, prio_features_list2, prio_features_list3, prio_features_list4, prio_features_list5)

class Guiding(nn.Module):
    def __init__(self, input_channels, output_channels, pre_train=True):
        super(Guiding, self).__init__()
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

        # self.deconv_5 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        # )
        # self.deconv_4 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(512, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=False),
        # )
        # self.deconv_3 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=False),
        # )
        # self.deconv_2 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=False)
        # )
        # self.deconv_1 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
        # )
        # self.unpool = nn.MaxUnpool2d(2,2)

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

        # conv_5 = self.unpool(conv_5,index_5,output_size=conv_5_copy.shape)
        # conv_5 = self.deconv_5(conv_5)

        # conv_4 = self.unpool(conv_5,index_4,output_size=conv_4_copy.shape)
        # conv_4 = self.deconv_4(conv_4)

        # conv_3 = self.unpool(conv_4,index_3,output_size=conv_3_copy.shape)
        # conv_3 = self.deconv_3(conv_3)

        # conv_2 = self.unpool(conv_3,index_2,output_size=conv_2_copy.shape)
        # conv_2 = self.deconv_2(conv_2)

        # conv_1 = self.unpool(conv_2,index_1,output_size=conv_1_copy.shape)
        # out_single_guiding = self.deconv_1(conv_1)

        guiding_one_class_list = [conv_1_copy, conv_2_copy, conv_3_copy, conv_4_copy, conv_5_copy]
        return guiding_one_class_list

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
        if 'deconv_1.1.num_batches_tracked' in key:
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

def SegNet_load_Pascal_model_guiding(model, model_filename):
    new_params = np.load(model_filename, allow_pickle=True, encoding='bytes')
    model_dict = model.state_dict()
    premodel_dict = new_params[0]
    premodel_list = []
    for key, value in premodel_dict.items():
        temp_dict = {'key':key,'value':value}
        premodel_list.append(temp_dict)
    param_layer = 0
    for key in model_dict:
        if 'deconv_3.0.weight' in key:
            break
        if 'run' in key or 'num' in key or 'img' in key or 'conv_4' in key or 'deconv_4' in key or 'conv_5' in key or 'deconv_5' in key:
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


    