import os
import sys
import cv2
import time
import numpy as np
from PIL import Image
from osgeo import gdal

import torch
from torch import nn
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
import model.HED.HED_class as HED
from dataset.HED_dataset import HED_dataset as HED_dataset
from dataset.XZY_dataset_new import XZY_train_dataset, XZY_test_dataset

from loss.loss_functions import HED_Loss

#train
#####################################################################################
#HED_IMG用VGG模型初始化，DEM使用xavier_normal初始化
Hed_IMG_pre = HED.HED(input_channels=3, out_channels=1).cuda()
Hed_IMG = HED.HED(input_channels=4, out_channels=1).cuda()
Hed_IMG = HED.add_conv_channels(Hed_IMG, Hed_IMG_pre, [1])

train_dataset = XZY_train_dataset('E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\Train\\10.17\\train_edge_dilate.csv', norm_list=[True, False, False])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

learn_rate = 0.0001
optimizer = torch.optim.SGD([{'params': Hed_IMG.parameters()}], lr=learn_rate, momentum=0.9, weight_decay=0.00015)

for epoch in range(201):
    epoch_loss_list = [0,0,0,0,0,0,0]
    for i, sample in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        
        raw_images, images, labels, labels_class = sample['raw_image'], sample['img_0'], sample['lab_0'], sample['lab_1']

        images = torch.cat([images[:,1:4,:,:], images[:,7,:,:].unsqueeze(1)], 1)
        raw_images = torch.cat([raw_images[:,1:4,:,:], raw_images[:,7,:,:].unsqueeze(1)], 1)
        img_outputs = Hed_IMG(images)[1]
        img_sideout1 = img_outputs[0]
        img_sideout2 = img_outputs[1]
        img_sideout3 = img_outputs[2]
        img_sideout4 = img_outputs[3]
        img_sideout5 = img_outputs[4]
        img_sideoutfuse = img_outputs[5]
        
        ##################################################################
        loss_side1 = HED_Loss.HED_LOSS(img_sideout1, labels.float())
        if loss_side1 == False:
            continue
        loss_side2 = HED_Loss.HED_LOSS(img_sideout2, labels.float())
        loss_side3 = HED_Loss.HED_LOSS(img_sideout3, labels.float())
        loss_side4 = HED_Loss.HED_LOSS(img_sideout4, labels.float())
        loss_side5 = HED_Loss.HED_LOSS(img_sideout5, labels.float())
        loss_fuse = HED_Loss.HED_LOSS(img_sideoutfuse, labels.float())

        loss = 0.2*loss_side1 + 0.2*loss_side2 + 0.2*loss_side3 + 0.2*loss_side4 + 0.2*loss_side5 + loss_fuse

        epoch_loss_list[0] += loss_side1.item()
        epoch_loss_list[1] += loss_side2.item()
        epoch_loss_list[2] += loss_side3.item()
        epoch_loss_list[3] += loss_side4.item()
        epoch_loss_list[4] += loss_side5.item()
        epoch_loss_list[5] += loss_fuse.item()
        epoch_loss_list[6] = epoch_loss_list[0]+epoch_loss_list[1]+epoch_loss_list[2]+epoch_loss_list[3]+epoch_loss_list[4]+epoch_loss_list[5]

        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        image_model_name = 'result\\Edge\\Hed_image_model' + str(epoch) + '.pkl'
        torch.save(Hed_IMG, image_model_name)

        layer = 0#输出的是第几层的
        for i in [img_sideoutfuse]:
            predects = i.cpu().detach().numpy() * 255
            classes = 0
            for j in range(predects[0].shape[0]):
                predect = predects[0][j].reshape(predects[0].shape[1],predects[0].shape[2])
                predect = transforms.ToPILImage()(predect)
                predect = predect.convert('RGB')
                predect_fn = 'result\\Edge\\'+str(epoch)+'_'+str(layer)+'_'+str(classes)+'pre.tif'
                predect.save(predect_fn)
                classes+=1
            layer += 1
        labels = labels_class.cpu().numpy() * 255
        label = labels[0].reshape(predects[0].shape[1],predects[0].shape[2]).astype(np.uint8)
        label = transforms.ToPILImage()(label)
        label = label.convert('RGB')
        label_fn = 'result\\Edge\\' + str(epoch) + 'lab.tif'
        label.save(label_fn)
        
        img = transforms.ToPILImage()(raw_images[0].cpu())
        img_fn = 'result\\Edge\\' + str(epoch) + 'img.tif'
        img.save(img_fn)

    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    print('epoch is %d : loss1 is %f , loss2 is %f , loss3 is %f , loss4 is %f , loss5 is %f , lossfuse is %f , lossall is %f'
        %(epoch,epoch_loss_list[0],epoch_loss_list[1],epoch_loss_list[2],epoch_loss_list[3],epoch_loss_list[4],epoch_loss_list[5],epoch_loss_list[6]))

