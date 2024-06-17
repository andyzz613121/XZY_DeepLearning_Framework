
import sys
import time
import numpy as np

from torch.utils import data
import torchvision.transforms as transforms
import torch
import torch.nn as nn

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from dataset.HS_dataset_new import HS_dataset
from model.HyperSpectral.Auto_Multiply.AutoContrast_ASPP import AutoContrast_ASPP, AutoContrast_ASPP_Branchs
from model.HyperSpectral.Auto_Multiply.AutoConstrast_2D import AutoContrast_2D
from model.HyperSpectral.Auto_Multiply.AutoContrast_4D import AutoContrast_4D
from model.HyperSpectral.Auto_Multiply.Backbone import SP_3D

from testing.HS.test_HS import testHS
from loss.loss_functions import Contrastive_Loss

def main(times, dataset_num):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 32
    init_lr = 0.001
    PCA = False
    norm = True

    if dataset_num == 2:
        dataset = 'Pavia'
    elif dataset_num == 1:
        dataset = 'Salinas'
    elif dataset_num == 3:
        dataset = 'Houston18'
    elif dataset_num == 4:        
        dataset = 'Houston13'
    else:
        return
    
    if dataset == 'Salinas':
        input_channels = 204
        output_channels = 16

    elif dataset == 'Houston13':
        input_channels = 144
        output_channels = 15

    elif dataset == 'Pavia': 
        input_channels = 103
        output_channels = 9

    elif dataset == 'Houston18':
        input_channels = 48
        output_channels = 20


    if PCA == True:
        input_channels = 30

    train_dst = HS_dataset(dataset, PCA, norm)

    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    model_img = AutoContrast_ASPP(1, output_channels).cuda() 

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
        # model_AutoFE.train()
        # transprob_step = torch.zeros([10, output_channels, output_channels]).cuda()

        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            cur_itrs += 1

            img=sample['img']
            label=sample['label']
            out_list, Feat_list = model_img(img)

            cfy_loss = 0
            for index in range(len(out_list)):
                outs = out_list[index]
                cfy_loss += loss_function(outs, label.long())

            cst_loss = 0
            for index in range(len(Feat_list)):
                Feats = Feat_list[index]
                logits, conloss_labels = Contrastive_Loss.info_nce_loss_withclass(Feats, label, other_class_num = 20)
                if logits == None:
                    continue
                cst_loss += loss_function(logits, conloss_labels)
            
            loss = cst_loss + cfy_loss
            # loss = cfy_loss

            loss.backward()
            interval_loss += loss.item()
            optimizer.step()

        scheduler.step()

        print('--------epoch %d done --------'%cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs in [300, 500]:
            folder = 'result\\' + dataset + '\\' + str(times) + '\\'
            if os.path.exists(folder) == False:
                os.makedirs(folder)

            image_model_name = folder + '\\image_model_xzy' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
            # image_model_name = folder + '\\AutoFE_model' + str(cur_epochs) + '.pkl'
            # torch.save(model_AutoFE, image_model_name)

            path = folder + '\\' + str(cur_epochs)
            testHS(dataset, model_img, path, norm, PCA)

        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    import os
    for times in range(10):
        for dataset_num in range(2, 3):
            main(times, dataset_num)

    # from visual import Draw
    # for b in range(out_list[0].shape[0]):
    #     de = out_list[0][b].cpu().detach().numpy()
    #     imgs = out_list[1][b].cpu().detach().numpy()
    #     lab = label[b].cpu().detach().numpy()
    #     path1 = 'result\\lab' + str(lab) + '_' + str(iii) + '.tif'
    #     path2 = 'result\\lab' + str(lab) + '_' + str(iii) + '.png'
    #     print(path1)
    #     Draw.draw_curve([x for x in range(len(de))], de, path1)
    #     Draw.draw_curve([x for x in range(len(de)+1)], imgs, path2)
    #     iii += 1
