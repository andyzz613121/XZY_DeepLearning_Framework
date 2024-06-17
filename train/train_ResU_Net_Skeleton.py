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
from model.ResNet.ResU_Net_Skeleton import ResU_Net_Skeleton, add_pre_model
from model.ResNet.ResU_Net_Skeleton_softmax import ResU_Net_Skeleton_softmax, add_pre_model
from model.Self_Module.Auto_Weights.Weight_MLP import cal_confuse_matrix

def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 16
    init_lr = 0.001
    dataset = 'Vaihingen'
    # dataset = 'Potsdam'

    # train_dst = ISPRS_dataset('train\\csv_files\\train_Vai.csv', dataset)
    # train_dst = ISPRS_dataset('train\\csv_files\\train_Pos.csv', dataset)
    train_dst = RS_dataset('train\\csv_files\\train_GID.csv')
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    # model_img_pre = SegNet_skeleton.SegNet_skeleton(3, 6).cuda()
    # model_img_pre = torch.load('result\\Skeleton\\ResU_Net_Pos_Base\\image_model50.pkl')
    # model_img_pre = torch.load('result\\Skeleton\\ResU_Net_Vai_Base\\image_model50.pkl')
    model_img = ResU_Net_Skeleton('ResNet34', 4, 6).cuda()
    # model_img = add_pre_model(model_img, model_img_pre)

    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': 1*init_lr}
    ], lr=init_lr, momentum=0.9, weight_decay=1e-4)
    

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # Restore
    total_epoch = 200
    cur_epochs = 0
    confuse_matrix_init = torch.ones([6, 6]).cuda()
    confuse_matrix_init[0][0], confuse_matrix_init[1][1], confuse_matrix_init[2][2] = 0, 0, 0
    confuse_matrix_init[3][3], confuse_matrix_init[4][4], confuse_matrix_init[5][5] = 0, 0, 0

    confuse_matrix_sum = 0
    confuse_matrix_avg = 0
    first_epoch = True
    # =====  Train  =====
    model_img.train()
    #==========   Train Loop   ==========#
    while True: 
        cur_epochs += 1
        interval_loss = 0
        
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()

            img=sample['img']
            label=sample['label']
            # dsm = sample['dsm']
            # ndvi = sample['ndvi']

            # inputs = torch.cat([img, dsm, ndvi], 1)

            # Run First Time
            if first_epoch == True:
                confuse_matrix_input = confuse_matrix_init
            else:
                confuse_matrix_input = confuse_matrix_avg

            # output = model_img(inputs, accuracy_list)
            confuse_matrix_flatten = torch.reshape(confuse_matrix_input, (1, -1))
            
            # output, d1, d2, d3, d4 = model_img(inputs, confuse_matrix_flatten)
            output, d1, d2, d3, d4 = model_img(img, confuse_matrix_flatten)
            pre = torch.argmax(output, 1)

            confuse_matrix = cal_confuse_matrix(pre, label, 6)
            confuse_matrix_sum += confuse_matrix
            
            # BackWard
            loss1 = Classification_Loss.loss_with_class(output, label)
            loss2 = Classification_Loss.loss_with_class(d1, label)
            loss3 = Classification_Loss.loss_with_class(d2, label)
            loss4 = Classification_Loss.loss_with_class(d3, label)
            loss5 = Classification_Loss.loss_with_class(d4, label)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            loss.backward()
            interval_loss += loss.item()
            optimizer.step()

        first_epoch = False
        confuse_matrix_avg = confuse_matrix_sum/(i+1)
        confuse_matrix_avg = confuse_matrix_avg/torch.max(confuse_matrix_avg)
        scheduler.step()
        print('--------epoch %d done --------', cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        print('confuse_matrix_avg: %f', confuse_matrix_avg)
        print('confuse_matrix: %f', confuse_matrix)
        print('confuse_matrix_input: %f', confuse_matrix_input)
        
        if cur_epochs >= 0:
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
        
        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()
