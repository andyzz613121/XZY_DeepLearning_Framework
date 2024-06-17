import os
import sys
import time
import numpy as np
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

import torch
import torch.nn as nn
from torch.utils import data
from model.HyperSpectral.Auto_Multiply.AutoContrast_ASPP import AutoContrast_ASPP
from model.HyperSpectral.Baseline import SpecAttenNet, HybridSN, CNN_2D, CNN_3D, SSRN, FDSSC, DBDA, Baselines
from dataset.XZY_dataset_new import XZY_train_dataset
from testing.HS.test_HS import testHS
from loss.loss_functions import Contrastive_Loss

def main(times, name, date):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 32
    init_lr = 0.001

    input_channels = 12
    if date == '6_29':
        output_channels = 8
    elif date == '7_14':
        output_channels = 10
    elif date == '7_24':
        output_channels = 11
    elif date == '8_04':
        output_channels = 11
    elif date == '9_12':
        output_channels = 10
    elif date == '4_10':
        output_channels = 7
    elif date == '5_20':
        output_channels = 7
    elif date == '10_17':
        output_channels = 6
    elif date == 'year':
        output_channels = 13
        input_channels = 6
    dataname = 'Sentinel' + date
    # test_imgpath = 'E:\\dataset\\毕设数据\\new\\2. MS\\Imgs\\'+date+'.tif'
    test_imgpath = 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Imgs\\Spatial_pred\\img_test.tif'
    # train_dst = XZY_train_dataset('E:\\dataset\\毕设数据\\new\\2. MS\\HS\\'+date+'\\train.csv', norm_list=[True, False], type=['img', 'value'])
    train_dst = XZY_train_dataset('E:\\dataset\\毕设数据\\new\\2. MS\\Time_Pixs\\Spatial_pred\\train.csv', norm_list=[True, False], type=['img', 'value'])
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, drop_last=True) 
    
    if name == '1D':
        model_img = Baselines.HuEtAl(input_channels, output_channels).cuda()
    elif name == '2D':
        model_img = SpecAttenNet.SpecAttenNet(input_channels, output_channels).cuda()
    elif name == '3D':
        model_img = CNN_3D.CNN3D(output_channels).cuda()
    # elif name == 'MCM':
    #     model_img = MCM.MCM(output_channels).cuda()
    # elif name == 'R2D':
    #     model_img = R2D.R2D(output_channels).cuda()

    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': 1*init_lr}
    ], lr=init_lr, momentum=0.9, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    # Restore
    total_epoch = 500
    cur_itrs = 0
    cur_epochs = 0
    #==========   Train Loop   ==========#
    while True: #cur_itrs < opts.total_itrs:
        cur_epochs += 1
        interval_loss = 0
        
        model_img.train()
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            cur_itrs += 1

            img=sample['rawimg_0']
            label=sample['lab_0']
            out_list = model_img(img)

            cfy_loss = 0
            for index in range(len(out_list)):
                outs = out_list[index]
                cfy_loss += loss_function(outs, label.long())

            loss = cfy_loss

            loss.backward()
            interval_loss += loss.item()
            optimizer.step()

        scheduler.step()

        print('--------epoch %d done --------'%cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f' %(optimizer.param_groups[0]['lr']))
        
        if cur_epochs in [1, 100, 200, 300]:
            folder = 'result\\Time_HS\\' + dataname + '\\' + name + '\\' + str(times) + '\\'
            if os.path.exists(folder) == False:
                os.makedirs(folder)

            image_model_name = folder + '\\image_model_xzy' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)

            from testing.XZY_testImage_Base import XZY_testIMG_HS_Base
            XZY_testIMG_HS_Base([model_img], output_channels, [test_imgpath], Pre_Folder=folder, pre_imgtag = 'pred_seg', patchsize=256, stride=128, randomtime=0)
            
        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    for times in range(4):
        for name in ['2D', '3D']:
            for date in ['year']:
                main(times, name, date)
