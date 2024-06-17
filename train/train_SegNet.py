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

from model.SegNet.SegNet_dis import SegNet as SegNet_dis
from model.SegNet.SegNet import SegNet
def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 8
    init_lr = 0.001
    dataset = 'Vaihingen'
    # dataset = 'Potsdam'

    train_dst = ISPRS_dataset('train\\csv_files\\train_Vai.csv', dataset)
    # train_dst = ISPRS_dataset('train\\csv_files\\train_Pos.csv', dataset)
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    
    model_img = SegNet(6, 6).cuda()
    # model_add = SegNet(3, 6).cuda()

    model_dis = SegNet_dis(3, 1).cuda()
    model_dis.load_state_dict(torch.load('pretrained\\Dis\\Dis_Vai.pth'))
    # model_dis.load_state_dict(torch.load('pretrained\\Dis\\Dis_Pos.pth'))

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
    # model_add.train()
    model_dis.eval()
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
            dis = sample['dis']
            ndvi = sample['ndvi']

            dis = model_dis(img)
            if torch.max(dis)==torch.min(dis):
                if torch.max(dis)!=0:
                    dis = dis/torch.max(dis)
            else:
                dis = (dis-torch.min(dis))/(torch.max(dis)-torch.min(dis))
            dis = (dis - 0.5)/0.5

            adds = torch.cat([img, dsm, dis, ndvi], 1)

            imgout = model_img(adds)
            # addout = model_add(adds)
            # out = model_img.imgguiding_fuse(torch.cat([imgout, addout], 1))

            loss = Classification_Loss.loss_with_class(imgout, label)
            
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
            # add_model_name = 'result\\add_model' + str(cur_epochs) + '.pkl'
            # torch.save(model_add, add_model_name)
        
        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()
