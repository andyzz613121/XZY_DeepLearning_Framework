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
from dataset.XZY_dataset import ISPRS_dataset, RS_dataset

from loss.loss_functions import Classification_Loss
from model.ResNet.ResU_Net_Difference_CM_ClassmaskDG import ResU_Net_Diff_Global_Attention
from model.Self_Module.Auto_Weights.Weight_MLP import cal_confuse_matrix
def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 16
    init_lr = 0.001
    dataset = 'Vaihingen'
    # dataset = 'Potsdam'
    out_class = 6
    train_dst = RS_dataset('train\\csv_files\\train_GID.csv')
    # train_dst = ISPRS_dataset('train\\csv_files\\train_Pos.csv', dataset)
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    model_img = ResU_Net_Diff_Global_Attention(4, out_class).cuda()

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
    Global_CM = 0
    #==========   Train Loop   ==========#
    while True: #cur_itrs < opts.total_itrs:
        cur_epochs += 1
        interval_loss = 0
        confuse_matrix_sum = 0
        confuse_matrix_avg = 0
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            cur_itrs += 1

            img=sample['img']
            label=sample['label']
            # dsm = sample['dsm']
            # dis = sample['dis']
            # ndvi = sample['ndvi']

            # inputs = torch.cat([img, dsm, ndvi], 1)
            inputs = img

            # Run First Time
            if cur_epochs <= 1:
                confuse_matrix_input = torch.ones([out_class, out_class]).cuda()
            else:
                confuse_matrix_input = Global_CM

            output, d1_map, d2_map, d3_map, d4_map = model_img(inputs, confuse_matrix_input)
            pre = torch.argmax(output, 1)

            confuse_matrix = cal_confuse_matrix(pre, label, out_class)
            confuse_matrix_sum += confuse_matrix
            confuse_matrix_avg = confuse_matrix_sum/(i+1)
            Global_CM = confuse_matrix_avg/torch.max(confuse_matrix_avg)

            # BackWard
            loss1 = Classification_Loss.loss_with_class(output, label)
            loss2 = Classification_Loss.loss_with_class(d1_map, label)
            loss3 = Classification_Loss.loss_with_class(d2_map, label)
            loss4 = Classification_Loss.loss_with_class(d3_map, label)
            loss5 = Classification_Loss.loss_with_class(d4_map, label)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            loss.backward()
            interval_loss += loss.item()
            optimizer.step()

        scheduler.step()
        print('--------epoch %d done --------', cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        print('Global_CM: %f', Global_CM)
        print('confuse_matrix: %f', confuse_matrix)
        print('confuse_matrix_sum: %f', confuse_matrix_sum)
        
        if cur_epochs >= 0:
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
        
        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()
