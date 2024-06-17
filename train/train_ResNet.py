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
from dataset.deeplab_v3plus_dataset import classification_dataset
from loss.loss_functions import Classification_Loss
from model.ResNet.ResNet import resnet34
from test_ResNet import test
def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 32
    init_lr = 0.0001

    train_dst = classification_dataset('E:\\dataset\\GF2\\train\\GF2.csv')
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    model_img = resnet34(4).cuda()

    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': 1*init_lr}
    ], lr=init_lr, momentum=0.9, weight_decay=1e-4)
    
    loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.1,1.0])).float().cuda())

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # Restore
    total_epoch = 200
    cur_itrs = 0
    cur_epochs = 0
    # =====  Train  =====
    model_img.train()
    #==========   Train Loop   ==========#
    while True: #cur_itrs < opts.total_itrs:
        cur_epochs += 1
        interval_loss = 0
    
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            cur_itrs += 1

            img=sample['img']
            label=sample['label']
            
            out = model_img(img)
            # index_1 = (label != 1)
            index_23 = (label > 1)
            index_1 = (label == 1)
            label[index_1] = 0
            label[index_23] = 1
            label = torch.reshape(label, [label.shape[0], -1])
            label, _ = torch.max(label, 1)

            loss = loss_function(out, label.long())
            # print(label, out, loss)
            loss.backward()
            interval_loss += loss.item()
            optimizer.step()
            
        scheduler.step()
        print('--------epoch %d done --------'%cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs >= 0:
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
        
        if cur_epochs % 5 == 0:
            test(model_img)

        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()
