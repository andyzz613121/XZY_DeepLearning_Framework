import os
import sys
import time
import numpy as np
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

from model.DeepLabV3 import deeplabv3plus_resnet101
from model.DeepLabV3.deeplabv3plus_xzy import deeplabv3plus

import torch
from torch.utils import data
import torchvision.transforms as transforms

from dataset.XZY_dataset_new import XZY_train_dataset, XZY_test_dataset
from loss.loss_functions import Classification_Loss
from model.SegNet.SegNet import *

def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

    batch_size = 32
    lr = 0.001
    train_dst = XZY_train_dataset('E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\Train\\10.17\\train.csv', norm_list=[True, False])
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, drop_last=True) 

    # model_img = SegNetGB(13, 10).cuda()
    # model_img_pre = SegNetGB(3, 10).cuda()
    # model_img = add_conv_channels(model_img, model_img_pre, [10])
    model_img = deeplabv3plus(input_channels=5, out_channels=6).cuda()
    model_hed = torch.load('result\\Edge\\10.17\\GB\\Hed_image_model200.pkl').cuda()

    # optimizer = torch.optim.SGD(params=[
    #     {'params': model_img.backbone.parameters(), 'lr': 0.1*lr},
    #     {'params': model_img.classifier.parameters(), 'lr': lr}
    # ], lr=lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': 1*lr}
    ], lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Restore
    total_epoch = 200
    cur_epochs = 0
    
    model_img.train()
    model_hed.train()
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
            
            edge = model_hed(img4c)[1][5]
            
            input1 = torch.cat([img4c, edge], 1)
            inputs = [input1, edge]
            output = model_img(inputs, 'GB')

            # input1 = torch.cat([img, edge], 1)
            # output = model_img(input1, edge)

            loss = Classification_Loss.loss_with_class_nonorm(output, label.long())
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            interval_loss += loss

        scheduler.step()
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), 'epoch is %d, loss is %f'%(cur_epochs, interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs % 50 == 0:
            out_folder = 'result\\Segment\\'
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            image_model_name = out_folder + 'deeplabv3_plus_image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)


        
if __name__ == '__main__':
    main()
