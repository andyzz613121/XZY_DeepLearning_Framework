import os
import sys
import time
import numpy as np
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

from model.DeepLabV3 import deeplabv3plus_resnet101
from model.DeepLabV3.deeplabv3plus_xzy import deeplabv3plus
from model.SegNet.SegNet import *
from model.FCN.FCN import *

import torch
from torch.utils import data
import torchvision.transforms as transforms

from dataset.XZY_dataset_new import XZY_train_dataset, XZY_test_dataset
from loss.loss_functions import Classification_Loss

def main(netmodel):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

    batch_size = 32
    lr = 0.001
    trainfile = 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Pixs_500\\Spatial_pred\\train.csv'
    train_dst = XZY_train_dataset(trainfile, norm_list=[True, False], type=['img', 'img'])
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, drop_last=True)
    criterion = torch.nn.CrossEntropyLoss()
    if netmodel == 'DeepLab':
        model_img = deeplabv3plus(input_channels=6, out_channels=13).cuda()
        optimizer = torch.optim.SGD(params=[
            {'params': model_img.backbone.parameters(), 'lr': 0.1*lr},
            {'params': model_img.classifier.parameters(), 'lr': lr}
        ], lr=lr, momentum=0.9, weight_decay=1e-4)
    elif netmodel == 'FCN':
        model_img = FCN(6, 13).cuda()
        model_img_pre = FCN(3, 13).cuda()
        model_img = add_conv_channels(model_img, model_img_pre, [3])
        optimizer = torch.optim.SGD(params=[
            {'params': model_img.parameters(), 'lr': lr}
        ], lr=lr, momentum=0.9, weight_decay=1e-4)
    elif netmodel == 'SegNet':
        model_img = SegNet(6, 13).cuda()
        model_img_pre = SegNet(3, 13).cuda()
        model_img = add_conv_channels(model_img, model_img_pre, [3])
        optimizer = torch.optim.SGD(params=[
            {'params': model_img.parameters(), 'lr': lr}
        ], lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        print('Unknown Net')
        return
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Restore
    total_epoch = 10
    cur_epochs = 0
    
    model_img.train()
    #==========   Train Loop   ==========#
    while cur_epochs <= total_epoch: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        cur_epochs += 1
        interval_loss = 0
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            img1=sample['rawimg_0'][:,:,:256,:256]
            # img2=sample['rawimg_1'][:,:,:256,:256]
            # img1=sample['img_0'][:,:,:256,:256]
            label=sample['lab_0'][:,:256,:256]
            # img=torch.cat([img1, img2], 1)
            #img=img1
            if netmodel == 'DeepLab':
                output = model_img([img1], 'base')
            else:
                output = model_img(img1)

            # loss = Classification_Loss.loss_with_class_nonorm(output, label.long())
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            interval_loss += loss

        scheduler.step()
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), 'epoch is %d, loss is %f'%(cur_epochs, interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs % 1 == 0:
            out_folder = 'result\\Time_500\\Spatial\\' + netmodel + '\\'
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            image_model_name = out_folder + netmodel + '_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
        
if __name__ == '__main__':
    for netmodel in ['DeepLab']:
        main(netmodel)
