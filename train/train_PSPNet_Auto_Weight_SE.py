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
from model.PSPNet.PSPNet_Basic import PSPNet_AW
from model.Self_Module.Auto_Weights.Weight_MLP import cal_confuse_matrix
from model.SegNet.SegNet_dis import SegNet

def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 4
    init_lr = 0.001
    dataset = 'Vaihingen'

    train_dst = ISPRS_dataset('train\\csv_files\\train_Vai.csv', dataset)
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    model_dis = SegNet(3, 1).cuda()
    model_dis.load_state_dict(torch.load('pretrained\\Dis\\Dis_Vai.pth'))
    model_img = PSPNet_AW(6, 6).cuda()

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
        confuse_matrix = None
        confuse_matrix_sum = 0
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

            # Run First Time
            inputs = torch.cat([img, dsm, dis, ndvi], 1).permute(0, 2, 3, 1).contiguous()
            inputs_0 = torch.mul(inputs, model_img.auto_weight0).permute(0, 3, 1, 2).contiguous()
            inputs_1 = torch.mul(inputs, model_img.auto_weight1).permute(0, 3, 1, 2).contiguous()
            inputs_2 = torch.mul(inputs, model_img.auto_weight2).permute(0, 3, 1, 2).contiguous()
            inputs_3 = torch.mul(inputs, model_img.auto_weight3).permute(0, 3, 1, 2).contiguous()
            inputs_4 = torch.mul(inputs, model_img.auto_weight4).permute(0, 3, 1, 2).contiguous()
            inputs_5 = torch.mul(inputs, model_img.auto_weight5).permute(0, 3, 1, 2).contiguous()

            output_0 = model_img.CARB0(model_img(inputs_0))
            output_1 = model_img.CARB1(model_img(inputs_1))
            output_2 = model_img.CARB2(model_img(inputs_2))
            output_3 = model_img.CARB3(model_img(inputs_3))
            output_4 = model_img.CARB4(model_img(inputs_4))
            output_5 = model_img.CARB5(model_img(inputs_5))
            output = torch.cat([output_0, output_1, output_2, output_3, output_4, output_5], 1)
            # inputs = torch.cat([img, dsm, dis, ndvi], 1)
            # output = model_img(inputs)

            pre = torch.argmax(output, 1)
            
            # BackWard
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
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pth'
            torch.save(model_img.state_dict(), image_model_name)
        
        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()
