import os
import sys
import time
import numpy as np
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

from model.MLP.mlp import mlp

import torch
from torch import nn
from torch.utils import data

from dataset.XZY_dataset_new import XZY_train_dataset, XZY_test_dataset
from loss.loss_functions import Classification_Loss

def main(date):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

    if date == '5_20':
        output_channels = 7
        start = 0
        datet = '5.20'
    elif date == '6_29':
        output_channels = 8
        start = 1
        datet = '6.19'
    elif date == '7_14':
        output_channels = 10
        start = 2
        datet = '7.14'
    elif date == '8_04':
        output_channels = 11
        start = 3
        datet = '8.04'
    elif date == '9_12':
        output_channels = 10
        start = 4
        datet = '9.12'
    elif date == '10_17':
        output_channels = 6
        start = 5
        datet = '10.17'

    batch_size = 8
    lr = 0.001
    trainfile = 'E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\Train\\'+datet+'\\train.csv'
    train_dst = XZY_train_dataset(trainfile, norm_list=[True, False])
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, drop_last=True) 

    model_img = mlp(2*output_channels, output_channels).cuda()
    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': lr}
    ], lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Restore
    total_epoch = 10
    cur_epochs = 0
    
    model_img.train()
    hsmodel_path = 'result\\HS500\\Sentinel'+date+'\\LGSF\\image_model_xzy500.pkl'
    cbmodel_path = 'result\\Spatial_paper\\Edge\\'+datet+'\\CB\\Hed_image_model200.pkl'
    sgmodel_path = 'result\\Spatial_paper\\Segment\\'+datet+'\\allchannel_train\\CB\\deeplabv3_plus_image_model300.pkl'
    Pre_Folder = 'result\\Spatial_Spectral\\'+date+'\\'
    if not os.path.exists(Pre_Folder):
        os.makedirs(Pre_Folder)

    hsmodel = torch.load(hsmodel_path).cuda().eval()
    cbmodel = torch.load(cbmodel_path).cuda().eval()
    sgmodel = torch.load(sgmodel_path).cuda().eval() # # 11111111111

    loss_function = nn.CrossEntropyLoss()
    #==========   Train Loop   ==========#
    while cur_epochs <= total_epoch: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        cur_epochs += 1
        interval_loss = 0
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()

            img=sample['img_0']
            label=sample['lab_0']
            with torch.no_grad():
                hs_outputs = hsmodel(img, 'test')

                img_patch4c = torch.cat([img[:,1:4,:,:], img[:,7,:,:].unsqueeze(1)], 1)
                cb_outputs = cbmodel(img_patch4c)[1][5]
                img_cb = torch.cat([img, cb_outputs], 1)
                cb_inputs = [img_cb, cb_outputs]
                cb_outputs = sgmodel(cb_inputs, 'CB')
            inputs = torch.cat([hs_outputs, cb_outputs], 1).view(batch_size, 2*output_channels, -1).transpose(1, 2).contiguous().view(-1, 2*output_channels)
            outputs = model_img(inputs)
            lab = label.view(-1)

            # loss = Classification_Loss.loss_with_class_nonorm(output, lab.long())
            loss = loss_function(outputs, lab.long())
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            interval_loss += loss

        scheduler.step()
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), 'epoch is %d, loss is %f'%(cur_epochs, interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs % 1 == 0:
            out_folder = 'result\\SS\\mlp\\' + date + '\\'
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            image_model_name = out_folder + 'mlp_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
        
if __name__ == '__main__':
    for date in ['9_12', '6_29', '7_14', '8_04', '10_17', '5_20']:
        main(date)