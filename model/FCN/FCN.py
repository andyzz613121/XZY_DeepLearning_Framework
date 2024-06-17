import torch
from torch import nn
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image 
import torchvision.models as models
import os
import numpy as np
import torch.nn.functional as F
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG16(nn.Module):
    def __init__(self, input_channels):
        super(VGG16,self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):
                # m.weight.data.normal_(0, 0.2)
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
        if input_channels == 3:
            FCN_load_premodel(self, 'pretrained\\FCN\\VGG_ILSVRC_16_layers.npy')

    def forward(self, x):
        #返回的outputs里面包含了倒数1,2,3尺度的特征图
        #最终输出的图像大小为outputs[1/32,1/16,1/8]
        outputs = []
        x_conv1 = self.conv_1(x)
        x_conv2 = self.conv_2(x_conv1)
        outputs.append(x_conv2)
        x_conv3 = self.conv_3(x_conv2)
        outputs.append(x_conv3)
        x_conv4 = self.conv_4(x_conv3)
        outputs.append(x_conv4)
        x_conv5 = self.conv_5(x_conv4)
        outputs.append(x_conv5)
        return outputs

class FCN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FCN, self).__init__()
        #VGG网络前面去掉全连接的部分
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.relu    = nn.ReLU(inplace=True)
        #倒数1层的反卷积[1/32 --> 1/16]
        self.deconv5 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(512)
        #倒数2层的反卷积[1/16 --> 1/8]
        self.deconv4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(256)
        #倒数3层的反卷积[1/8 --> 1/4]
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        #倒数4层的反卷积[1/4 --> 1/2]
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        #倒数5层的反卷积[1/2 --> 1/1]
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, output_channels, kernel_size=1)

        self.conv_imgedge = nn.Conv2d(2*output_channels, output_channels, kernel_size=1)
        
        
        if input_channels == 3:
            FCN_load_premodel(self, 'pretrained\\FCN\\VGG_ILSVRC_16_layers.npy')
        else:
            print('FCN init by xavier')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal(m.weight.data)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.ConvTranspose2d):
                    # m.weight.data.normal_(0, 0.2)
                    nn.init.xavier_normal(m.weight.data)
                    m.bias.data.fill_(0)
                    
    def forward(self, x):
        # vgg_output = self.VGG_net(x)

        # #最后一层的特征图，大小为原图的1/32
        # x5 = vgg_output[3]
        # #倒数第二层的特征图，大小为原图的1/16
        # x4 = vgg_output[2]
        # #倒数第三层的特征图，大小为原图的1/8
        # x3 = vgg_output[1]
        # #倒数第四层的特征图，大小为原图的1/4
        # x2 = vgg_output[0]

        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        x4 = self.conv_4(x3)
        x5 = self.conv_5(x4)

        #最后一层特征图反卷积，使其大小与倒数第二层特征图一致(原图1/16)
        x5_4 = self.relu(self.deconv5(x5))
        #最后一层特征图与倒数第二层相加
        # x4 = self.bn5(x4 + x5_4)
        # print(x5_4.shape)   
        x5_4 = F.interpolate(x5_4, size=(x4.shape[2], x4.shape[3]), mode='bilinear')
        # print(x5_4.shape)
        x4 = x4 + x5_4
        #倒数第二层特征图反卷积，使其大小与倒数第三层特征图一致(原图1/8)
        x4_3 = self.relu(self.deconv4(x4))
        #倒数第二层特征图与倒数第三层相加 
        #x3 = self.bn4(x3 + x4_3)
        x4_3 = F.interpolate(x4_3, size=(x3.shape[2], x3.shape[3]), mode='bilinear')
        x3 = x3 + x4_3
        #倒数第三层特征图反卷积，使其大小与倒数第四层特征图一致(原图1/4)
        x3_2 = self.relu(self.deconv3(x3))
        #x2 = self.bn3(x2 + x3_2)
        x3_2 = F.interpolate(x3_2, size=(x2.shape[2], x2.shape[3]), mode='bilinear')
        x2 = x2 + x3_2
        
        # x1 = self.bn2(self.relu(self.deconv2(x2)))
        # x0_image = self.bn1(self.relu(self.deconv1(x1)))
        x1 = self.relu(self.deconv2(x2))
        x0_image = self.relu(self.deconv1(x1))

        outputs = self.classifier(x0_image)
        # outputs = torch.cat([outputs, edge], 1)
        # outputs = self.conv_imgedge(outputs)
        return outputs

def FCN_load_premodel(model, premodel_filename):
    new_params = np.load(premodel_filename, allow_pickle=True, encoding='bytes')
    model_dict = model.state_dict()
    premodel_dict = new_params[0]
    premodel_list = []
    for key, value in premodel_dict.items():
        # if b'bn' not in key:
        # print(key, value.shape)
        temp_dict = {'key':key,'value':value}
        premodel_list.append(temp_dict)
    param_layer = 0
    for key in model_dict:
        if 'deconv' in key:
            break
        if 'conv' in key:
            pre_k = premodel_list[param_layer]['key']
            pre_v = premodel_list[param_layer]['value']
            pre_v = torch.from_numpy(pre_v)
            # print(key, model_dict[key].shape, pre_v.shape)
            assert model_dict[key].shape == pre_v.shape
            model_dict[key] = pre_v
            #print('     set FCN model %s layer param by Pascal premodel %s layer param'%(key, pre_k))
            param_layer += 1
        
    model.load_state_dict(model_dict)
    print('FCN init weight by %s model'%premodel_filename)
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

def loss_with_class(input, label, threshold=0.5):#std*sum
    n, c, h, w = input.size()
    log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)
    # print(edge.shape)
    label = label.view(1, -1).long()
    # print(label.shape)
    weights = np.zeros((label.shape[0], label.shape[1])).astype(np.float32)
    # print(weights.shape)
    #calculate the median class&edge class num
    classes_num = []
    for classes in range(3):
        class_index = (label==classes)
        class_num = class_index.sum().cpu().numpy()
        classes_num.append(class_num)

    classes_num = np.array(classes_num)
    sum_num = classes_num.sum()
    
    min_class_num = 100000 #class 3 rate = min class num rate
    for classes in range(3):
        class_index = (label==classes).cpu()
        class_num = class_index.sum().cpu().numpy()
        if class_num < min_class_num:
            min_class_num = class_num
        if class_num != 0:
            weights[class_index] = 1 - class_num/sum_num #give class weight by median/class_num

    # print(weights)    
    weights0 = weights
    weights0 = torch.from_numpy(weights0).cuda()
    
    m = nn.LogSoftmax(dim=1)
    loss_function = nn.NLLLoss()
    labeln_1 = torch.unsqueeze(label[0], 1) #onehot need label shape (n, 1)
    one_hot = torch.zeros(label.shape[1], 3).cuda().scatter_(1, labeln_1, 1)
    one_hot = one_hot.transpose(0, 1)
    log_cross = m(log_cross)
    loss_ce = torch.mul(one_hot, log_cross[0])
    loss_ce = torch.mul(loss_ce, weights0[0])
    loss_ce = -1 * torch.sum(loss_ce)/label.shape[1]
 

    return loss_ce
    
def main():
    v=VGG16(1)
    model = FCN(v,4)
    FCN_load_premodel(model, 'D:\\Code\\Hed_Seg\\pretrained\\FCN_Pascal\\VGG_ILSVRC_16_layers.npy')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

if __name__ == "__main__":
    main()