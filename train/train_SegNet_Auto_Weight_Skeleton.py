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
# from model.SegNet import SegNet_skeleton_0128 as SegNet_skeleton
from model.Self_Module.Auto_Weights.Weight_MLP import cal_confuse_matrix
from model.SegNet.SegNet_dis import SegNet as SegNet_dis
from model.SegNet.SegNet import SegNet
def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 16
    init_lr = 0.001
    dataset = 'Vaihingen'
    # dataset = 'Potsdam'

    train_dst = ISPRS_dataset('train\\csv_files\\train_Vai.csv', dataset)
    # train_dst = ISPRS_dataset('train\\csv_files\\train_Pos.csv', dataset)
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    # model_img_pre = SegNet_skeleton.SegNet_skeleton(3, 6).cuda()
    model_img_pre = torch.load('result\\SegNet_Vai_Base\\image_model50.pkl')
    model_img = SegNet_skeleton.SegNet_skeleton(6, 6).cuda()
    model_img = SegNet_skeleton.add_pre_model(model_img, model_img_pre)

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
    model_dis.eval()
    #==========   Train Loop   ==========#
    while True: #cur_itrs < opts.total_itrs:
        cur_epochs += 1
        interval_loss = 0
        confuse_matrix = torch.ones([6, 6]).cuda()
        confuse_matrix_sum = 0
        confuse_matrix_avg = 0
        first_flag = True
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            cur_itrs += 1

            raw_image=sample['raw_image']
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

            inputs = torch.cat([img, dsm, dis, ndvi], 1)

            # Run First Time
            if first_flag == True:
                first_flag = False
                # accuracy_list = torch.ones([6]).cuda()
                confuse_matrix_input = confuse_matrix
            else:
                # accuracy_list = SegNet_skeleton.cal_accuracy_list(confuse_matrix_avg)
                confuse_matrix_input = confuse_matrix_avg

            # output = model_img(inputs, accuracy_list)
            confuse_matrix_flatten = torch.reshape(confuse_matrix_input, (1, -1))
            conv_1, conv_2, conv_3, conv_4, conv_5 = model_img(inputs, confuse_matrix_flatten)
            pre = torch.argmax(conv_1, 1)

            confuse_matrix = cal_confuse_matrix(pre, label)
            confuse_matrix_sum += confuse_matrix
            confuse_matrix_avg = confuse_matrix_sum/(i+1)
            confuse_matrix_avg = confuse_matrix_avg/torch.max(confuse_matrix_avg)

            # BackWard
            loss1 = Classification_Loss.loss_with_class(conv_1, label)
            loss2 = Classification_Loss.loss_with_class(conv_2, label)
            loss3 = Classification_Loss.loss_with_class(conv_3, label)
            loss4 = Classification_Loss.loss_with_class(conv_4, label)
            loss5 = Classification_Loss.loss_with_class(conv_5, label)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            loss.backward()
            interval_loss += loss.item()
            optimizer.step()
            
        scheduler.step()
        print('--------epoch %d done --------', cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        print('confuse_matrix_avg: %f', confuse_matrix_avg)
        print('confuse_matrix: %f', confuse_matrix)
        print('confuse_matrix_sum: %f', confuse_matrix_sum)
        
        if cur_epochs >= 0:
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
        
        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()
