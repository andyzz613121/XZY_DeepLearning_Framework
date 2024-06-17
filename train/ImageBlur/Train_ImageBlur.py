import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from model.SRNet.SRNet_Base import SRNet
from model.SegNet.SegNet import SegNet, add_conv_channels
from model.FCN.FCN import FCN
from model.DeepLabV3.deeplabv3plus_xzy import deeplabv3plus
from dataset.XZY_dataset_20221227 import XZY_train_dataset
from loss.loss_functions import Classification_Loss

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.Self_Module.Attention_SRNet import AttachAttentionModule
from model.Self_Module.SRM_filter import spam11, minmax41, srmfilter
class Srnet(nn.Module):
    def __init__(self):
        super(Srnet, self).__init__()
        print("Using SRNet in 'SRNet//SRNet_Base//'")
        # Layer 1
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.attention1 = AttachAttentionModule(64)

        # Layer 2
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.attention2 = AttachAttentionModule(16)

        # Layer 3
        self.layer31 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(16)
        self.layer32 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(16)
        self.attention3 = AttachAttentionModule(16)

        # Layer 4
        self.layer41 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(16)
        self.layer42 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(16)
        self.attention4 = AttachAttentionModule(16)

        # Layer 5
        self.layer51 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(16)
        self.layer52 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(16)
        self.attention5 = AttachAttentionModule(16)
        
        # Layer 6
        self.layer61 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn61 = nn.BatchNorm2d(16)
        self.layer62 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(16)
        self.attention6 = AttachAttentionModule(16)
        
        # Layer 7
        self.layer71 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(16)
        self.layer72 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(16)
        self.attention7 = AttachAttentionModule(16)
        
        # Layer 8
        self.layer81 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=1, stride=2, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(16)
        self.layer82 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn82 = nn.BatchNorm2d(16)
        self.layer83 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 9
        self.layer91 = nn.Conv2d(in_channels=16, out_channels=64,
            kernel_size=1, stride=2, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(64)
        self.layer92 = nn.Conv2d(in_channels=16, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(64)
        self.layer93 = nn.Conv2d(in_channels=64, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn93 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 10
        self.layer101 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=1, stride=2, padding=0, bias=False)
        self.bn101 = nn.BatchNorm2d(128)
        self.layer102 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn102 = nn.BatchNorm2d(128)
        self.layer103 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn103 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 11
        self.layer111 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=1, stride=2, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(256)
        self.layer112 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.layer113 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 12
        self.layer121 = nn.Conv2d(in_channels=256, out_channels=512,
            kernel_size=3, stride=2, padding=1, bias=False)
        self.bn121 = nn.BatchNorm2d(512)
        self.layer122 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122 = nn.BatchNorm2d(512)
        
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.debn2   = nn.BatchNorm2d(512)
        #倒数2层的反卷积[1/16 --> 1/8],
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.debn1   = nn.BatchNorm2d(256)
        #倒数3层的反卷积[1/8 --> 1/4],
        self.deconv0_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.debn0_1   = nn.BatchNorm2d(128)
        #倒数4层的反卷积[1/4 --> 1/2],
        self.deconv0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.debn0_2   = nn.BatchNorm2d(64)
        #倒数5层的反卷积[1/2 --> 1/1],
        self.deconv0_3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.debn0_3   = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, 2, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)    
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)

    def forward(self, inputs):
        # Layer 1
        conv = self.layer1(inputs)
        actv = F.relu(self.bn1(conv))
        actv = self.attention1(actv)
        
        # Layer 2
        conv = self.layer2(actv)
        actv = F.relu(self.bn2(conv))
        actv = self.attention2(actv)
        
        # Layer 3
        conv1 = self.layer31(actv)
        actv1 = F.relu(self.bn31(conv1))
        conv2 = self.layer32(actv1)
        bn = self.bn32(conv2)
        res = torch.add(actv, bn)
        res = self.attention3(res)
        
        # Layer 4
        conv1 = self.layer41(res)
        actv1 = F.relu(self.bn41(conv1))
        conv2 = self.layer42(actv1)
        bn = self.bn42(conv2)
        res = torch.add(res, bn)
        res = self.attention4(res)
        
        # Layer 5
        conv1 = self.layer51(res)
        actv1 = F.relu(self.bn51(conv1))
        conv2 = self.layer52(actv1)
        bn = self.bn52(conv2)
        res = torch.add(res, bn)
        res = self.attention5(res)
        
        # Layer 6
        conv1 = self.layer61(res)
        actv1 = F.relu(self.bn61(conv1))
        conv2 = self.layer62(actv1)
        bn = self.bn62(conv2)
        res = torch.add(res, bn)
        res = self.attention6(res)
        
        # Layer 7
        conv1 = self.layer71(res)
        actv1 = F.relu(self.bn71(conv1))
        conv2 = self.layer72(actv1)
        bn = self.bn72(conv2)
        res = torch.add(res, bn)
        res = self.attention7(res)
        
        # Layer 8
        convs = self.layer81(res)
        convs = self.bn81(convs)
        conv1 = self.layer82(res)
        actv1 = F.relu(self.bn82(conv1))
        conv2 = self.layer83(actv1)
        bn = self.bn83(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        
        # Layer 9
        convs = self.layer91(res)
        convs = self.bn91(convs)
        conv1 = self.layer92(res)
        actv1 = F.relu(self.bn92(conv1))
        conv2 = self.layer93(actv1)
        bn = self.bn93(conv2)
        pool = self.pool2(bn)
        res = torch.add(convs, pool)
        
        # Layer 10
        convs = self.layer101(res)
        convs = self.bn101(convs)
        conv1 = self.layer102(res)
        actv1 = F.relu(self.bn102(conv1))
        conv2 = self.layer103(actv1)
        bn = self.bn103(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        
        # Layer 11
        convs = self.layer111(res)
        convs = self.bn111(convs)
        conv1 = self.layer112(res)
        actv1 = F.relu(self.bn112(conv1))
        conv2 = self.layer113(actv1)
        bn = self.bn113(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        
        # Layer 12
        conv1 = self.layer121(res)
        actv1 = F.relu(self.bn121(conv1))
        conv2 = self.layer122(actv1)
        bn = self.bn122(conv2)
        
        #最后一层特征图反卷积，使其大小与倒数第二层特征图一致(原图1/16)
        x2_1 = F.relu(self.deconv2(bn))
        #倒数第二层特征图反卷积，使其大小与倒数第三层特征图一致(原图1/8)
        x1_0 = F.relu(self.deconv1(x2_1))
        
        #倒数第三层特征图连续3次反卷积，使其大小与原始图像一致(由原图1/8到1/1)
        x0_image = self.debn0_1(F.relu(self.deconv0_1(x1_0)))
        x0_image = self.debn0_2(F.relu(self.deconv0_2(x0_image)))
        x0_image = self.debn0_3(F.relu(self.deconv0_3(x0_image)))
        outputs = self.classifier(x0_image)
        return outputs

def main():
    for times in [1, 2, 3, 4, 5]:
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        batch_size = 32

        # csv_file = 'E:\\dataset\\ImageBlur\\Data\\train\\无模糊图像（复杂度+模糊半径）\\selected\\r_' + str(times) + '\\train\\trainblur_r' + str(times) + '.csv'
        csv_file = 'E:\\dataset\\ImageBlur\\Data\\train\\训练图像\\train\\new_trainblur_rgb.csv'
        # csv_file = 'E:\\dataset\\ImageBlur\\Data\\train\\训练图像\\test\\new_testblur_rgb.csv'
        train_dst = XZY_train_dataset(csv_file, norm=True)
        train_loader = data.DataLoader(
            train_dst, batch_size = batch_size, shuffle = True)

        # model_img = SRNet(in_channels=3).cuda()
        # model_img = SegNet(3,2,pre_train=False).cuda()
        model_img = deeplabv3plus(input_channels=3, out_channels=2, pretrained=False).cuda()
        # model_img = FCN(input_channels=3, out_channels=2).cuda()
        # model_img_pre = SegNet(3,2).cuda()
        # model_img = SegNet(4,2).cuda()
        # model_img = add_conv_channels(model_img, model_img_pre, [1])
        # model_img = torch.load('result\\SRNetResult\\Response\\SRM+IMG\\1\\Model\\SRNet_model200.pkl').cuda()
        optimizer = torch.optim.SGD([{'params': model_img.parameters()}], lr=0.001, momentum=0.9, weight_decay=0.00015)
        weights = [1-(255553985/270729216), 1-(15175231/270729216)]
        # Restore
        total_epoch = 202
        cur_itrs = 0
        cur_epochs = 0
        save_folder = 'result\\SRNetResult\\' + str(times) + '\\'
        if os.path.exists(save_folder) == False:
            os.makedirs(save_folder)
        loss_record = open(save_folder + '\\loss_record.txt', 'w')
        #==========   Train Loop   ==========#
        while True: #cur_itrs < opts.total_itrs:
            cur_epochs += 1
            interval_loss = 0
            model_img.train()

            for i, sample in enumerate(train_loader, 0):
                optimizer.zero_grad()
                cur_itrs += 1

                # raw_imgs=sample['raw_image']
                img=sample['img_0']
                # srm=sample['img_1']
                label=sample['label']

                # input = torch.cat([img, srm], 1)
                input = img
                out = model_img([input], 'base')
                # out = model_img(input)

                # loss = 0
                # for l in range(img.shape[0]):
                #     tempout0 = out[l][0]
                #     tempout1 = out[l][1]

                #     tempout0 = tempout0.reshape(-1,1)
                #     tempout1 = tempout1.reshape(-1,1)
                #     tempoutput = torch.cat([tempout0, tempout1], 1)

                #     templabel = label[l]
                #     templabel = templabel.reshape(-1,1)
                #     #使用属于训练集1的损失函数计算方法\n",
                #     loss = loss + loss_function(tempoutput, templabel.long())
                # loss = loss / img.shape[0]
                loss = Classification_Loss.loss_with_class_norm(out, label, zhiding_weight=weights)
                loss.backward()
                interval_loss += loss.item()
                optimizer.step()

            print('--------epoch %d done --------'%cur_epochs)
            print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
            print('loss: %f' %(interval_loss/len(train_dst)))
            print('lr: %f', optimizer.param_groups[0]['lr'])
            loss_record.write("Epoch %d, loss %f\n" %(cur_epochs, interval_loss/len(train_dst)))
                
            if cur_epochs in [1, 10, 20, 30, 50, 100, 150, 200, 300]:
            # save_folder = 'result\\SRNetResult\\' + str(times) + '\\' + str(cur_epochs)
            # if os.path.exists(save_folder) == False:
            #     os.makedirs(save_folder)

                image_model_name = save_folder + '\\SRNet_model' + str(cur_epochs) + '.pkl'
                torch.save(model_img, image_model_name)
                    
                # predect = torch.argmax(out[0], 0).cpu().detach().numpy().astype(np.uint8)
                # predect = transforms.ToPILImage()(predect)
                # # predect = predect.convert('RGB')
                # predect_fn = 'result\\SRNetResult\\' + str(times) + '\\' + str(cur_epochs)+'_pre.tif'
                # predect.save(predect_fn)

                # label = label[0].cpu().numpy().astype(np.uint8)
                # label = transforms.ToPILImage()(label)
                # # label = label.convert('RGB')
                # label_fn = 'result\\SRNetResult\\' + str(times) + '\\' + str(cur_epochs) + 'lab.tif'
                # label.save(label_fn)
                
                # img = transforms.ToPILImage()(raw_imgs[0].cpu())
                # img_fn = 'result\\SRNetResult\\' + str(times) + '\\' + str(cur_epochs) + 'img.tif'
                # img.save(img_fn)

                # test_image_name = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\' + str(cur_epochs)
                # model_img.eval()
                # test_IMG(model_img, 2, norm, PCA)

            if cur_epochs >= total_epoch:
                break

        
if __name__ == '__main__':
    main()
