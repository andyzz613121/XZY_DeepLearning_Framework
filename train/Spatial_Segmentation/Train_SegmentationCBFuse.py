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
import model.HED.HED_class as HED
from dataset.HED_dataset import HED_dataset as HED_dataset
from dataset.XZY_dataset_new import XZY_train_dataset, XZY_test_dataset

from loss.loss_functions import HED_Loss

def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

    batch_size = 8
    lr = 0.001
    train_dst = XZY_train_dataset('E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\Train\\7.14\\train_edge_dilate.csv', norm_list=[True, False, False])
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, drop_last=True) 

    model_img = deeplabv3plus(input_channels=14, out_channels=10).cuda()
    Hed_IMG_pre = HED.HED(input_channels=3, out_channels=10).cuda()
    model_hed = HED.HED(input_channels=4, out_channels=10).cuda()
    model_hed = HED.add_conv_channels(model_hed, Hed_IMG_pre, [1])

    optimizer = torch.optim.SGD(params=[
        {'params': model_img.backbone.parameters(), 'lr': 0.1*lr},
        {'params': model_img.classifier.parameters(), 'lr': lr},
        {'params': model_hed.parameters(), 'lr': lr}
    ], lr=lr, momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Restore
    total_epoch = 100
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

            raw_images, images, labels, labels_class = sample['raw_image'], sample['img_0'], sample['lab_0'], sample['lab_1']

            images = torch.cat([images[:,1:4,:,:], images[:,7,:,:].unsqueeze(1)], 1)
            raw_images = torch.cat([raw_images[:,1:4,:,:], raw_images[:,7,:,:].unsqueeze(1)], 1)
            img_outputs = model_hed(images)[1]
            img_sideout1 = img_outputs[0]
            img_sideout2 = img_outputs[1]
            img_sideout3 = img_outputs[2]
            img_sideout4 = img_outputs[3]
            img_sideout5 = img_outputs[4]
            img_sideoutfuse = img_outputs[5]
            
            edge = img_sideoutfuse
            input1 = torch.cat([images, edge], 1)
            inputs = [input1, edge]
            output = model_img(inputs, 'CB')

            # loss_seg = Classification_Loss.loss_with_CB(output, edge, labels_class.long())
            loss = Classification_Loss.loss_with_class_norm(output, labels_class.long())
            # rate_seg = int(loss_seg/loss_edge)
            # rate_edge = int(loss_edge/loss_seg)
            # if loss_seg>loss_edge:
            #     loss_edge = rate_seg*loss_edge
            # else:
            #     loss_seg = rate_edge*loss_seg

            # loss = loss_seg + 20*loss_edge
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
            HED_model_name = out_folder + 'HED_model' + str(cur_epochs) + '.pkl'
            torch.save(model_hed, HED_model_name)
        
if __name__ == '__main__':
    main()
