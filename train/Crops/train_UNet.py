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

from dataset.XZY_dataset import RS_dataset
from loss.loss_functions import Classification_Loss
from model.UNet.UNet import UNet
def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 8
    init_lr = 0.001

    train_dst = RS_dataset('E:\\dataset\\连云港GF2数据\\1_RPC+全色融合\\GF2_PMS1_E119.1_N34.2_20210730_L1A0005787958-pansharp1\\train_all_label\\train_random.csv')
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    model_img = UNet(4, 2, is_batchnorm=False).cuda()
    # print(model_img)
    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': 1*init_lr}
    ], lr=init_lr, momentum=0.9, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # Restore
    total_epoch = 50
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

            raw_image=sample['raw_image']
            img=sample['img']
            label=sample['label']
            
            imgout = model_img(img)
            loss = Classification_Loss.loss_with_class(imgout, label)
            
            loss.backward()
            interval_loss += loss.item()
            optimizer.step()
        
        scheduler.step()
        print('--------epoch %d done --------' %cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs >= 0:
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)

            predect = torch.argmax(imgout[0], 0).cpu().detach().numpy().astype(np.uint8)
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
        
        if cur_epochs >= total_epoch:
            break

if __name__ == '__main__':
    main()
