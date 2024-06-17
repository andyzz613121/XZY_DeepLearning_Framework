import os
import sys
import time
import numpy as np
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

from model.Transformer.transformer_decompose import Transformer

import torch
from torch.utils import data
from torch import nn

from dataset.XZY_dataset_new import XZY_train_dataset 
from loss.loss_functions import Classification_Loss
from testing.XZY_testImage_Base import *

def main(netmodel):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

    seq_len = 6
    batch_size = 4
    lr = 0.001
    total_epoch = 10
    
    trainfile = 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Pixs_500\\Spectral_pred\\train.csv'
    train_dst = XZY_train_dataset(trainfile, norm_list=[True, False], type=['img', 'img'])
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, drop_last=True) 

    model_img = Transformer().cuda()
    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': lr}
    ], lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    criterion = nn.CrossEntropyLoss()

    # Restore
    cur_epochs = 0
    model_img.train()
    #==========   Train Loop   ==========#
    while cur_epochs < total_epoch: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        cur_epochs += 1
        interval_loss = 0
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            rawimg=sample['rawimg_0'][:,:,125:175,125:175].contiguous()
            lab=sample['lab_0'][:,125:175,125:175].contiguous()

            # B, C, H, W -> B*H*W, C, 1
            rawimg = rawimg.view(rawimg.shape[0], rawimg.shape[1], -1).long()      # B, C, H*W
            rawimg = rawimg.transpose(1, 2).contiguous().view(-1, seq_len)          # B*H*W, C, 1
            lab = lab.view(-1, 1).long()
            # rawimg_cy = []
            # lab_cy = []
            # for i in range(13):
            #     pos_idx = (lab == i)
            #     maxs = pos_idx.sum()
            #     if maxs == 0:
            #         continue
            #     maxs = 2000 if maxs > 2000 else maxs
            #     labs_tmp = lab[pos_idx][:maxs].view(-1, 1)
            #     raw_tmp = rawimg[pos_idx][:maxs]
            #     lab_cy.append(labs_tmp)
            #     rawimg_cy.append(raw_tmp)
            # rawimg = torch.cat([x for x in rawimg_cy], 0)
            # lab = torch.cat([x for x in lab_cy], 0)
            enc_input = rawimg
            dec_word = lab
            dec_sin, dec_ein = torch.zeros_like(dec_word)+13, torch.zeros_like(dec_word)+14
            dec_input = torch.cat([dec_sin, dec_word], 1)
            label = torch.cat([dec_word, dec_ein], 1)
            output = model_img(enc_input, dec_input)
            label = label.view(-1, 1, 1)
            output = output.view(-1, 15)
            loss = criterion(output, label.view(-1))
            # output = output.view(-1, 15, 1, 1)
            # loss = Classification_Loss.loss_with_class_nonorm(output, label.long())
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            interval_loss += loss

        scheduler.step()
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), 'epoch is %d, loss is %f'%(cur_epochs, interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs % 1 ==0:
            out_folder = 'result\\Time_500\\Spectral\\' + netmodel + '\\'
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            image_model_name = out_folder + netmodel + '_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
            # img_path2 = 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Imgs\\Spectral_pred\\img_test.tif'
            # test = XZY_testIMG_classification_Base([image_model_name], 13, [img_path2, img_path2], norm_list=[True, True], Pre_Folder = out_folder+netmodel+'\\', pre_imgtag = 'pred_'+netmodel, patchsize=128, randomtime=3000)

if __name__ == '__main__':
    # linspace_list = [[1, 2],[1,2,3,4],[1,2,3,4,5]]
    for i in range(10):
        main('Transformer_decompms_(1_2_3)'+str(i))
