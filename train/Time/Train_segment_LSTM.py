import os
import sys
import time
import numpy as np
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

from model.LSTM.CONVLSTM import CLSTM
from model.LSTM.LSTM import lstm_with_classification

import torch
from torch.utils import data

from dataset.XZY_dataset_new import XZY_train_dataset
from loss.loss_functions import Classification_Loss

def main(netmodel):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

    seq_len = 12
    feat_len = 1
    hidden_size = 64
    num_layers = 3
    num_classes = 13
    batch_size = 32
    lr = 0.01
    total_epoch = 10
    
    trainfile = 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Pixs_500\\train_SS.csv'
    train_dst = XZY_train_dataset(trainfile, norm_list=[True, True, False], type=['img', 'img', 'img'])
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, drop_last=True) 

    model_img = lstm_with_classification(feat_len, hidden_size, num_layers, num_classes).cuda()
    # optimizer = torch.optim.SGD(params=[
    #     {'params': model_img.parameters(), 'lr': lr}
    # ], lr=lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model_img.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    # Restore
    cur_epochs = 0
    model_img.train()
    #==========   Train Loop   ==========#
    while cur_epochs <= total_epoch: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        cur_epochs += 1
        interval_loss = 0
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()

            img1=sample['img_0'][:,:,125:175,125:175].contiguous()
            img2=sample['img_1'][:,:,125:175,125:175].contiguous()
            img = torch.cat([img1, img2], 1)
            label=sample['lab_0'][:,125:175,125:175].contiguous()
            # B, C, H, W -> B*H*W, C, 1
            # img = img.view(img.shape[0], img.shape[1], -1)      # B, C, H*W
            # img = img.transpose(1, 2).contiguous().view(-1, seq_len, 1) # B*H*W, C, 1
            
            img = img.transpose(0, 1).contiguous().view(seq_len, -1, 1)
            output = model_img(img) #.transpose(0, 1).contiguous().view(batch_size, num_classes, 1, 1)
            # loss = criterion(output, label.view(-1).long())
            output = output.view(-1, 13, label.shape[0], label.shape[1])
            loss = Classification_Loss.loss_with_class_nonorm(output, label.long())
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            interval_loss += loss
            
        scheduler.step()
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), 'epoch is %d, loss is %f'%(cur_epochs, interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs % 1 == 0:
            out_folder = 'result\\Time_500\\SS\\' + netmodel + '\\'
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            image_model_name = out_folder + netmodel + '_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
        
if __name__ == '__main__':
    for i in range(100):
        print(i)
        main('LSTM'+str(i))
