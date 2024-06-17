import imp
import os
import sys
import time
import numpy as np
from PIL import Image

from torch.utils import data
import torchvision.transforms as transforms
import torch
import torch.nn as nn

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from dataset import XZY_dataset
from dataset.XZY_dataset import ISPRS_dataset, RS_dataset

from loss.loss_functions import Classification_Loss
from model.ResNet.ResU_Net_BoundarySimilar import ResU_Net_BS, gdal_write_tif, add_ResUNet
from model.SegNet.SegNet_BoundarySimilar import SegNet_BS
from model.model_operation import SegNet_add_conv_channels
def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 16
    init_lr = 0.001
    num_classes = 6
    input_channels = 5
    dataset = 'Vaihingen'
    # dataset = 'Potsdam'

    train_dst = ISPRS_dataset('train\\csv_files\\train_Vai.csv', dataset)
    # train_dst = ISPRS_dataset('train\\csv_files\\train_Pos.csv', dataset)
    # train_dst = RS_dataset('train\\csv_files\\train_GID.csv')
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    # model_img_pre = SegNet_BS(3, num_classes).cuda()
    # model_img = SegNet_BS(input_channels, num_classes).cuda()
    # model_img = SegNet_add_conv_channels(model_img, model_img_pre, [input_channels-3])
    model_img = torch.load('result\\Boundary_Similar\\SegNet_Vai_Base\\image_model50.pkl').cuda()

    for name, para in model_img.named_parameters():
        if name == 'T1':
            para_T1 = para
        if name == 'a':
            para_a = para
        if name == 'w1':
            para_w1 = para
        if name == 'w2':
            para_w2 = para

    optimizer = torch.optim.SGD(params=[
        {'params': para_T1, 'lr': 0.1*init_lr},
        {'params': para_a, 'lr': 0.1*init_lr},
        {'params': para_w1, 'lr': 0.1*init_lr},
        {'params': para_w2, 'lr': 0.1*init_lr}
    ], lr=init_lr, momentum=0.9, weight_decay=1e-4)

    # optimizer = torch.optim.SGD(params=[
    #     {'params': model_img.parameters(), 'lr': 1*init_lr}
    # ], lr=init_lr, momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # Restore
    total_epoch = 200
    cur_epochs = 0
    # =====  Train  =====
    model_img.train()
    #==========   Train Loop   ==========#
    while True: 
        cur_epochs += 1
        interval_loss = 0
        # model_img.spectral_list = torch.zeros([num_classes, input_channels]).cuda()
        # model_img.class_list = torch.zeros([num_classes]).cuda()
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            img=sample['img']
            label=sample['label']
            dsm = sample['dsm']
            ndvi = sample['ndvi']
            # for name, para in model_img.named_parameters():
            #     if name == 'T1':
            #         print(para)
            #     if name == 'a':
            #         print(para)
            #     if name != 'T1' and name != 'a':
            #         print(para[0][0][0])
            #         break
            # raw_img=sample['raw_image']
            # img_out = raw_img.cpu().detach().numpy()[0]
            # gdal_write_tif('C:\\Users\\25321\\Desktop\\img.tif', img_out, 256, 256, 4)

            # lab_out = label.cpu().detach().numpy()[0]
            # gdal_write_tif('C:\\Users\\25321\\Desktop\\label.tif', lab_out, 256, 256, 1)
            inputs = torch.cat([img, dsm, ndvi], 1)
            output = model_img(inputs)
            loss = Classification_Loss.loss_with_class(output, label)
            
            loss.backward()
            interval_loss += loss.item()
            optimizer.step()

        scheduler.step()
        print('--------epoch %d done --------', cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs >= 0:
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
        
        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()
