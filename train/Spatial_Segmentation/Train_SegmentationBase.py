import os
import sys
import time
import numpy as np
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

from model.DeepLabV3 import deeplabv3plus_resnet101
from model.DeepLabV3.deeplabv3plus_xzy import deeplabv3plus
from model.SegNet.SegNet import *
# from model.FCN.FCN import *

import torch
from torch.utils import data
import torchvision.transforms as transforms

from dataset.XZY_dataset_new import XZY_train_dataset, XZY_test_dataset
from loss.loss_functions import Classification_Loss

def main(date):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

    if date == '6.29':
        output_channels = 8
    elif date == '7.14':
        output_channels = 10
    elif date == '7.24':
        output_channels = 11
    elif date == '8.04':
        output_channels = 11
    elif date == '9.12':
        output_channels = 10
    elif date == '4.10':
        output_channels = 7
    elif date == '5.20':
        output_channels = 7
    elif date == '10.17':
        output_channels = 6


    batch_size = 32
    lr = 0.001
    trainfile = 'E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\Train\\'+date+'\\train.csv'
    train_dst = XZY_train_dataset(trainfile, norm_list=[True, False])
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, drop_last=True) 

    # model_img = deeplabv3plus(input_channels=4, out_channels=6).cuda()
    # optimizer = torch.optim.SGD(params=[
    #     {'params': model_img.backbone.parameters(), 'lr': 0.1*lr},
    #     {'params': model_img.classifier.parameters(), 'lr': lr}
    # ], lr=lr, momentum=0.9, weight_decay=1e-4)

    # model_img = FCN(4, output_channels).cuda()
    # model_img_pre = FCN(3, output_channels).cuda()
    model_img = SegNet(12, output_channels).cuda()
    model_img_pre = SegNet(3, output_channels).cuda()
    model_img = add_conv_channels(model_img, model_img_pre, [9])
    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': lr}
    ], lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Restore
    total_epoch = 200
    cur_epochs = 0
    
    model_img.train()
    #==========   Train Loop   ==========#
    while cur_epochs <= total_epoch: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        cur_epochs += 1
        interval_loss = 0
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()

            raw_image=sample['raw_image']
            img=sample['img_0']
            label=sample['lab_0']
            
            img4c = torch.cat([img[:,1:4,:,:], img[:,7,:,:].unsqueeze(1)], 1)
            # raw_image = torch.cat([raw_image[:,1:4,:,:], raw_image[:,7,:,:].unsqueeze(1)], 1)
            
            # inputs = [img4c]
            # output = model_img(inputs, 'base')
            output = model_img(img)

            loss = Classification_Loss.loss_with_class_nonorm(output, label.long())
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            interval_loss += loss

        scheduler.step()
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), 'epoch is %d, loss is %f'%(cur_epochs, interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs == 200:
            out_folder = 'result\\Segment\\' + date + '\\allchannel_train\\SegNet\\\\'
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            image_model_name = out_folder + 'SegNet_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
        
if __name__ == '__main__':
    for date in ['10.17']:
        main(date)
