import sys
import os
import random
import time
import argparse
import numpy as np
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

from model.DeepLabV3 import deeplabv3plus_resnet101

import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms

from dataset.XZY_dataset import ISPRS_dataset, RS_dataset

from loss.loss_functions import Classification_Loss

def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

    batch_size = 8
    lr = 0.001
    train_dst = RS_dataset('E:\\dataset\\连云港GF2数据\\1_RPC+全色融合\\GF2_PMS1_E119.1_N34.2_20210730_L1A0005787958-pansharp1\\train\\GF2_LYG.csv')
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, drop_last=True) 

    model_img = deeplabv3plus_resnet101('img',input_channels=4,num_classes=2, output_stride=8)

    optimizer = torch.optim.SGD(params=[
        {'params': model_img.backbone.parameters(), 'lr': 1*lr},
        {'params': model_img.classifier.parameters(), 'lr': lr}
    ], lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Restore
    total_epoch = 200
    cur_itrs = 0
    cur_epochs = 0
    model_img = model_img.cuda()

    #==========   Train Loop   ==========#
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model_img.train()
        # model_HED.train()
        cur_epochs += 1
        interval_loss = 0
        for i, sample in enumerate(train_loader, 0):
            cur_itrs += 1
            optimizer.zero_grad()

            raw_image=sample['raw_image']
            img=sample['img']
            label=sample['label']

            output = model_img(img)

            loss = Classification_Loss.loss_with_class(output, label.long())
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            interval_loss += loss
            

        scheduler.step()
        # print("cur_itrs: ", cur_itrs)
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), 'epoch is %d, loss is %f'%(cur_epochs, interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs % 1 == 0:
            image_model_name = 'result\\deeplabv3_plus_image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)

            predect = torch.argmax(output[0], 0).cpu().detach().numpy().astype(np.uint8)
            predect = transforms.ToPILImage()(predect)
            # predect = predect.convert('RGB')
            predect_fn = 'result\\'+str(cur_epochs)+'_pre.tif'
            predect.save(predect_fn)

            label = label[0].cpu().numpy().astype(np.uint8)
            label = transforms.ToPILImage()(label)
            # label = label.convert('RGB')
            label_fn = 'result\\' + str(cur_epochs) + 'lab.tif'
            label.save(label_fn)
            
            img = transforms.ToPILImage()(raw_image[0].cpu())
            img_fn = 'result\\' + str(cur_epochs) + 'img.tif'
            img.save(img_fn)

        
if __name__ == '__main__':
    main()
