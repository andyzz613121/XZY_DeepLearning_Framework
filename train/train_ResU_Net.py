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
from dataset.XZY_dataset import ISPRS_dataset
from loss.loss_functions import Classification_Loss
from model.ResNet.ResU_Net import ResU_Net

def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 8
    init_lr = 0.001
    # dataset = 'Vaihingen'
    dataset = 'Potsdam'

    # train_dst = ISPRS_dataset('train\\csv_files\\train_Vai.csv', dataset)
    train_dst = ISPRS_dataset('train\\csv_files\\train_Pos.csv', dataset)
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    
    model_img = ResU_Net('ResNet34', 6, 6).cuda()
    # model_add = SegNet(3, 6).cuda()

    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': 1*init_lr}
    ], lr=init_lr, momentum=0.9, weight_decay=1e-4)
    

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # Restore
    total_epoch = 200
    cur_itrs = 0
    cur_epochs = 0
    # =====  Train  =====
    model_img.train()
    #==========   Train Loop   ==========#
    while True: #cur_itrs < opts.total_itrs:
        cur_epochs += 1
        interval_loss = 0
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            cur_itrs += 1

            img=sample['img']
            label=sample['label']
            dsm = sample['dsm']
            ndvi = sample['ndvi']

            inputs = torch.cat([img, dsm, ndvi], 1)
            out, d1_map, d2_map, d3_map, d4_map = model_img(inputs)

            loss1 = Classification_Loss.loss_with_class(out, label)
            # loss2 = Classification_Loss.loss_with_class(d1_map, label)
            # loss3 = Classification_Loss.loss_with_class(d2_map, label)
            # loss4 = Classification_Loss.loss_with_class(d3_map, label)
            # loss5 = Classification_Loss.loss_with_class(d4_map, label)
            # loss = loss1 + loss2 + loss3 + loss4 + loss5
            loss = loss1
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
