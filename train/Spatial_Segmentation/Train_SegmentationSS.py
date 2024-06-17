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
from model.SegNet.SegNet import *

import torch
from torch.utils import data
import torchvision.transforms as transforms

from dataset.XZY_dataset_new import XZY_train_dataset, XZY_test_dataset
from loss.loss_functions import Classification_Loss

def main(date):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

    if date == '5.20':
        output_channels = 7
        start = 0
    elif date == '6.29':
        output_channels = 8
        start = 1
    elif date == '7.14':
        output_channels = 10
        start = 2
    elif date == '8.04':
        output_channels = 11
        start = 3
    elif date == '9.12':
        output_channels = 10
        start = 4
    elif date == '10.17':
        output_channels = 6
        start = 5

    batch_size = 32
    lr = 0.001
    trainfile = 'E:\\dataset\\毕设数据\\new\\2. MS\\SS_month\\train.csv'
    train_dst = XZY_train_dataset(trainfile, norm_list=[True, False])
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, drop_last=True) 

    model_img = SegNet(3, output_channels).cuda()
    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': lr}
    ], lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Restore
    total_epoch = 100
    cur_epochs = 0
    
    model_img.train()
    #==========   Train Loop   ==========#
    while cur_epochs <= total_epoch: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        cur_epochs += 1
        interval_loss = 0
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()

            img=sample['img_0']
            label=sample['lab_0']

            img = torch.cat([img[:,start,:,:].unsqueeze(1), img[:,start,:,:].unsqueeze(1), img[:,start+6,:,:].unsqueeze(1)], 1)
            lab = label[:, start,:,:]

            output = model_img(img)
            loss = Classification_Loss.loss_with_class_nonorm(output, lab.long())
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            interval_loss += loss

        scheduler.step()
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), 'epoch is %d, loss is %f'%(cur_epochs, interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs % 20 == 0:
            out_folder = 'result\\SS\\' + date + '\\'
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            image_model_name = out_folder + 'SegNet_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
        
if __name__ == '__main__':
    for date in ['9.12', '6.29', '7.14', '8.04', '10.17']:
        main(date)