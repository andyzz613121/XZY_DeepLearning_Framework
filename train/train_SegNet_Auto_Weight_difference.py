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
from model.SegNet import SegNet_difference
from model.SegNet.SegNet_dis import SegNet
def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 8
    init_lr = 0.001
    dataset = 'Vaihingen'

    train_dst = ISPRS_dataset('train\\csv_files\\train_Vai.csv', dataset)
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    model_img_pre = SegNet_difference.SegNet_difference(3, 6).cuda()
    model_img = SegNet_difference.SegNet_difference(6, 6).cuda()
    model_img = SegNet_difference.add_conv_channels(model_img, model_img_pre, [3])

    model_dis = SegNet(3, 1).cuda()
    model_dis.load_state_dict(torch.load('pretrained\\Dis\\Dis_Vai.pth'))

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
        confuse_matrix_sum = [0,0,0,0]
        confuse_matrix_avg = [0,0,0,0]
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
            outimg_list, outCM_list = model_img(inputs, label)
            loss_1 = Classification_Loss.loss_with_class(outimg_list[0], label)
            loss_2 = Classification_Loss.loss_with_class(outimg_list[1], label)
            loss_3 = Classification_Loss.loss_with_class(outimg_list[2], label)
            loss_4 = Classification_Loss.loss_with_class(outimg_list[3], label)
            loss_5 = Classification_Loss.loss_with_class(outimg_list[4], label)

            # for layer_num in range(4):
            #     confuse_matrix_sum[layer_num] += outCM_list[layer_num]
            #     confuse_matrix_avg[layer_num] = confuse_matrix_sum[layer_num]/(i+1)
            #     # confuse_matrix_avg[layer_num] = confuse_matrix_avg[layer_num]/torch.max(confuse_matrix_avg[layer_num])

            # BackWard
            loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5
            loss.backward()
            interval_loss += loss.item()
            optimizer.step()
            
            

        scheduler.step()
        print('--------epoch %d done --------', cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        # print('confuse_matrix_avg: ', confuse_matrix_avg)
        # print('confuse_matrix_sum: ', confuse_matrix_sum)

        if cur_epochs >= 0:
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)

        if cur_epochs >= 0:
            img_fn = 'result\\' + str(cur_itrs) + 'img.png'
            lab_fn = 'result\\' + str(cur_itrs) + 'lab.png'
            out_img = raw_image.cpu()
            out_img = transforms.ToPILImage()(out_img[0])
            out_img.save(img_fn)

            out_lab = label.cpu()
            out_lab = transforms.ToPILImage()(out_lab[0])

            out_lab.save(lab_fn)

            flag = 0
            for item in outimg_list:
                fn = 'result\\' + str(cur_itrs) + str(flag) + '.png'
                predects = item
                _, predects = torch.max(item[0], 0)
                
                predects = predects.int()
                predects = predects.cpu()
                predects = transforms.ToPILImage()(predects)
                predects.save(fn)
                flag += 1

        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()
