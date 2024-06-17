import re
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from model.Self_Module.CARB import CARB_Block

class Mean_Std_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Mean_Std_Attention,self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        #到底是1个MLP输入为2*n维，还是2个MLP输入为1*n
        self.mean_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )
        self.std_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x_std = self.global_std(x)
        x_means = self.avg_pool(x).view(b, c)
        std_w = self.std_fc(x_std)
        mean_w = self.mean_fc(x_means)
        
        #两者如何融合
        w_sum = (std_w + mean_w).view(b, c, 1, 1)

        return x*w_sum

    def global_std(self, x):

        b, c, h, w = x.size()
        x = x.reshape([b,c,h*w])
        std = torch.std(x, 2)
        std = std.view(b, c)
        return std

class CM_Attention(nn.Module):
    def __init__(self, layer_num, node_list):
        super(CM_Attention, self).__init__()
        assert layer_num == len(node_list), "ERROR at Weight_MLP: Layer_num != len(node_list)"
        self.MLP = self.get_mlp(layer_num, node_list)

    def get_mlp(self, layer_num, node_list, drop_rate=0.2):
        layers = []
        for layer in range(layer_num-1):
            layers.append(nn.Linear(node_list[layer], node_list[layer+1]))
            if layer+1 != (layer_num-1):  #Last layer
                layers.append(nn.Dropout(drop_rate))
                layers.append(nn.ReLU())
        mlp = nn.Sequential(*layers)
        for m in mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        return mlp

    def forward(self, x, confuse_matrix):
        b, c, _, _ = x.size()
        confuse_matrix_flatten = torch.reshape(confuse_matrix, (1, -1))
        #混淆矩阵是一个Batch统计全部，还是每个Batch的图像分别统计
        CM_weight = self.MLP(confuse_matrix_flatten)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(x, CM_weight).permute(0, 3, 1, 2).contiguous()
        return x

class SegNet_difference(nn.Module):
    def __init__(self, input_channels, output_channels, pre_train=True):
        super(SegNet_difference,self).__init__()
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

        self.MSA5 = Mean_Std_Attention(512)
        self.MSA4 = Mean_Std_Attention(512)
        self.MSA3 = Mean_Std_Attention(256)
        self.MSA2 = Mean_Std_Attention(128)
        self.MSA1 = Mean_Std_Attention(64)
        self.Sideout5 = CARB_Block(512, 6)
        self.Sideout4 = CARB_Block(512, 6)
        self.Sideout3 = CARB_Block(256, 6)
        self.Sideout2 = CARB_Block(128, 6)
        self.Sideout1 = CARB_Block(64, 6)
        self.CWA5 = CM_Attention(3, [36, 512//2, 512])
        self.CWA4 = CM_Attention(3, [36, 512//2, 256])
        self.CWA3 = CM_Attention(3, [36, 256//2, 128])
        self.CWA2 = CM_Attention(3, [36, 128//2, 64])

        self.MSA_CWA_Fuse4 = CARB_Block(512*2, 512)
        self.MSA_CWA_Fuse3 = CARB_Block(256*2, 256)
        self.MSA_CWA_Fuse2 = CARB_Block(128*2, 128)
        self.MSA_CWA_Fuse1 = CARB_Block(64*2, 64)

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

    def forward(self, x, label):

        # CM2 = torch.tensor([[1.0000e+00, 4.7882e-03, 6.5638e-03, 8.2128e-03, 3.3859e-03, 1.7368e-04],
        # [5.4274e-03, 8.5516e-01, 5.0448e-03, 2.4646e-03, 5.3666e-05, 1.3852e-05],
        # [7.4387e-03, 5.2229e-03, 7.4203e-01, 1.6959e-02, 1.3347e-04, 2.5382e-05],
        # [9.3450e-03, 2.4990e-03, 1.7008e-02, 8.0468e-01, 1.4444e-04, 2.0302e-04],
        # [4.6249e-03, 6.4896e-05, 1.3194e-04, 1.4293e-04, 4.1109e-02, 2.6077e-06],
        # [2.3162e-04, 1.4808e-05, 2.4640e-05, 2.9977e-04, 8.0894e-08, 3.0211e-02]]).cuda()
        # CM3 = torch.tensor([[1.0000e+00, 4.8514e-03, 6.6104e-03, 8.2494e-03, 3.3081e-03, 1.6451e-04],
        # [5.3531e-03, 8.5489e-01, 5.0267e-03, 2.4139e-03, 5.3162e-05, 1.5381e-05],
        # [7.3117e-03, 5.2283e-03, 7.4189e-01, 1.6813e-02, 1.3452e-04, 2.2328e-05],
        # [9.2764e-03, 2.5480e-03, 1.6987e-02, 8.0472e-01, 1.4212e-04, 2.2145e-04],
        # [4.6985e-03, 6.4276e-05, 1.2723e-04, 1.4455e-04, 4.1180e-02, 2.8926e-06],
        # [2.4666e-04, 1.1081e-05, 2.1961e-05, 2.7126e-04, 1.3797e-07, 3.0198e-02]]).cuda()
        # CM4 = torch.tensor([[1.0000e+00, 5.4967e-03, 7.3477e-03, 8.7995e-03, 4.0041e-03, 2.0736e-04],
        # [6.0901e-03, 8.5612e-01, 5.6940e-03, 2.7372e-03, 5.8247e-05, 1.8808e-05],
        # [8.1260e-03, 5.9116e-03, 7.4215e-01, 1.7553e-02, 1.6810e-04, 2.4971e-05],
        # [9.8246e-03, 2.8641e-03, 1.7844e-02, 8.0587e-01, 1.6477e-04, 2.5167e-04],
        # [6.0169e-03, 8.3624e-05, 1.7277e-04, 1.5893e-04, 4.0573e-02, 8.0005e-06],
        # [2.7177e-04, 2.0135e-05, 3.3854e-05, 2.8421e-04, 3.5802e-07, 3.0216e-02]]).cuda()
        # CM5 = torch.tensor([[1.0000e+00, 8.1739e-03, 1.1272e-02, 1.2397e-02, 7.0712e-03, 2.9963e-04],
        # [9.1800e-03, 8.6243e-01, 8.4707e-03, 4.0617e-03, 1.2241e-04, 3.2471e-05],
        # [1.1990e-02, 8.6157e-03, 7.4127e-01, 2.3088e-02, 3.4708e-04, 5.7830e-05],
        # [1.3588e-02, 4.1338e-03, 2.3464e-02, 8.0769e-01, 3.1011e-04, 3.0559e-04],
        # [1.0570e-02, 1.5205e-04, 2.9652e-04, 2.5890e-04, 3.7788e-02, 8.2802e-06],
        # [4.1932e-04, 2.5058e-05, 3.9618e-05, 4.0669e-04, 3.1396e-06, 3.0483e-02]]).cuda()

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

        # #decoder
        # deconv_5 = self.MSA5(conv_5)
        # deconv_5 = self.unpool(deconv_5,index_5,output_size=conv_5_copy.shape)
        # deconv_5 = self.deconv_5(deconv_5)
        # deconv_5_sideout = self.Sideout5(deconv_5)
        # deconv_5_sideout = F.interpolate(deconv_5_sideout, size=(x.shape[2], x.shape[3]), mode='bilinear')
        # pre_5 = torch.argmax(deconv_5_sideout, 1)
        # CM5 = self.cal_confuse_matrix(pre_5, label)
        
        # CMA_4 = self.CWA5(deconv_5, CM5)
        # MSA_4 = self.MSA4(conv_4)
        # deconv_4 = torch.cat([CMA_4, MSA_4], 1)
        # deconv_4 = self.MSA_CWA_Fuse4(deconv_4)
        # deconv_4 = self.unpool(deconv_4,index_4,output_size=conv_4_copy.shape)
        # deconv_4 = self.deconv_4(deconv_4)
        # deconv_4_sideout = F.interpolate(deconv_4, size=(x.shape[2], x.shape[3]), mode='bilinear')
        # pre_4 = torch.argmax(deconv_4_sideout, 1)
        # CM4 = self.cal_confuse_matrix(pre_4, label)

        # CMA_3 = self.CWA4(deconv_4, CM4)
        # MSA_3 = self.MSA3(conv_3)
        # deconv_3 = torch.cat([CMA_3, MSA_3], 1)
        # deconv_3 = self.MSA_CWA_Fuse3(deconv_3)
        # deconv_3 = self.unpool(deconv_3,index_3,output_size=conv_3_copy.shape)
        # deconv_3 = self.deconv_3(deconv_3)
        # deconv_3_sideout = F.interpolate(deconv_3, size=(x.shape[2], x.shape[3]), mode='bilinear')
        # pre_3 = torch.argmax(deconv_3_sideout, 1)
        # CM3 = self.cal_confuse_matrix(pre_3, label)

        # CMA_2 = self.CWA3(deconv_3, CM3)
        # MSA_2 = self.MSA2(conv_2)
        # deconv_2 = torch.cat([CMA_2, MSA_2], 1)
        # deconv_2 = self.MSA_CWA_Fuse2(deconv_2)
        # deconv_2 = self.unpool(deconv_2,index_2,output_size=conv_2_copy.shape)
        # deconv_2 = self.deconv_2(deconv_2)
        # deconv_2_sideout = F.interpolate(deconv_2, size=(x.shape[2], x.shape[3]), mode='bilinear')
        # pre_2 = torch.argmax(deconv_2_sideout, 1)
        # CM2 = self.cal_confuse_matrix(pre_2, label)

        # CMA_1 = self.CWA2(deconv_2, CM2)
        # MSA_1 = self.MSA1(conv_1)
        # deconv_1 = torch.cat([CMA_1, MSA_1], 1)
        # deconv_1 = self.MSA_CWA_Fuse1(deconv_1)
        # deconv_1 = self.unpool(deconv_1,index_1,output_size=conv_1_copy.shape)
        # deconv_1 = self.deconv_1(deconv_1)

        # return [deconv_1, deconv_2_sideout, deconv_3_sideout, deconv_4_sideout, deconv_5_sideout], [CM2, CM3, CM4, CM5]
       
        #decoder
        deconv_5 = self.unpool(conv_5,index_5,output_size=conv_5_copy.shape)
        deconv_5 = self.deconv_5(deconv_5)
        deconv_5_sideout = self.Sideout5(deconv_5)
        deconv_5_sideout = F.interpolate(deconv_5_sideout, size=(x.shape[2], x.shape[3]), mode='bilinear')

        deconv_4 = torch.cat([conv_4, deconv_5], 1)
        deconv_4 = self.MSA_CWA_Fuse4(deconv_4)
        deconv_4 = self.unpool(deconv_4,index_4,output_size=conv_4_copy.shape)
        deconv_4 = self.deconv_4(deconv_4)
        deconv_4_sideout = F.interpolate(deconv_4, size=(x.shape[2], x.shape[3]), mode='bilinear')
        
        deconv_3 = torch.cat([conv_3, deconv_4], 1)
        deconv_3 = self.MSA_CWA_Fuse3(deconv_3)
        deconv_3 = self.unpool(deconv_3,index_3,output_size=conv_3_copy.shape)
        deconv_3 = self.deconv_3(deconv_3)
        deconv_3_sideout = F.interpolate(deconv_3, size=(x.shape[2], x.shape[3]), mode='bilinear')
        
        deconv_2 = torch.cat([conv_2, deconv_3], 1)
        deconv_2 = self.MSA_CWA_Fuse2(deconv_2)
        deconv_2 = self.unpool(deconv_2,index_2,output_size=conv_2_copy.shape)
        deconv_2 = self.deconv_2(deconv_2)
        deconv_2_sideout = F.interpolate(deconv_2, size=(x.shape[2], x.shape[3]), mode='bilinear')
   
        deconv_1 = torch.cat([conv_1, deconv_2], 1)
        deconv_1 = self.MSA_CWA_Fuse1(deconv_1)
        deconv_1 = self.unpool(deconv_1,index_1,output_size=conv_1_copy.shape)
        deconv_1 = self.deconv_1(deconv_1)

        return [deconv_1, deconv_2_sideout, deconv_3_sideout, deconv_4_sideout, deconv_5_sideout], []

    
    def difference(self, x1, x2):
        return x1

    def cal_confuse_matrix(self, predict, label):
        pre_pos_list = [] #predict等于各个类的下标数组
        label_pos_list = [] #label等于各个类的下标数组
        confuse_matrix = torch.zeros([6,6]).float().cuda()
        # label = label[:,0,:,:]
        for pre_class in range(6):
            pos_index = (predict == pre_class)
            pre_pos_list.append(pos_index)
        for label_class in range(6):
            pos_index = (label == label_class)
            label_pos_list.append(pos_index)
        
        for pre_class in range(6):
            for label_class in range(6):
                pos_index = pre_pos_list[pre_class]*label_pos_list[label_class]
                confuse_matrix[pre_class][label_class] = (pos_index.sum())
        return  confuse_matrix

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