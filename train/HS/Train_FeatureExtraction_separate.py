
import imp
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
from model.HyperSpectral.Baseline.Local_BaseLine import SP_1D, SP_MLP, SP_2D, SP_2D_AutoContrast
from model.HyperSpectral.FeatureExtract.AutoFeature import AutoFeature
from testing.HS.test_HS import testHS
from loss.loss_functions import Contrastive_Loss

def main(times, dataset_num):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 32
    init_lr = 0.001
    PCA = False
    norm = True

    if dataset_num == 1:
        dataset = 'Pavia'
    elif dataset_num == 2:
        dataset = 'Salinas'
    elif dataset_num == 3:
        dataset = 'Houston13'
    elif dataset_num == 4:
        dataset = 'Houston18'
    else:
        return

    if dataset == 'Pavia': 
        input_channels = 103
        output_channels = 9
    elif dataset == 'Houston13':
        input_channels = 144
        output_channels = 15 
    elif dataset == 'Houston18':
        input_channels = 48
        output_channels = 20
    elif dataset == 'Salinas':
        input_channels = 204
        output_channels = 16

    if PCA == True:
        input_channels = 30

    train_dst = HS_dataset(dataset, PCA, norm)

    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    # model_img = SP_2D_AutoContrast(input_channels, output_channels).cuda()
    model_img = AutoFeature(input_channels).cuda()

    
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
        # transprob_step = torch.zeros([10, output_channels, output_channels]).cuda()

        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            cur_itrs += 1

            img=sample['img']
            label=sample['label']

            Codes, _ = model_img(img)

            cst_loss = 0
            logits, conloss_labels = Contrastive_Loss.info_nce_loss_withclass(Codes, label)
            if logits == None:
                continue
            cst_loss += loss_function(logits, conloss_labels)

            cst_loss.backward()
            interval_loss += cst_loss.item()
            optimizer.step()
        
        # if cur_epochs > 200:
        #     model_img.transprob_glob = transprob_step/(transprob_step.sum(1).unsqueeze(2))

        scheduler.step()

        print('--------epoch %d done --------'%cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs in [200, 250, 300, 350, 400, 450, 500]:
            folder = 'result\\' + dataset + '\\' + str(times) + '\\'
            if os.path.exists(folder) == False:
                os.makedirs(folder)

            image_model_name = folder + '\\AutoFeatExct_' + dataset + '_' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)

        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    import os
    for times in range(5):
        for dataset_num in range(1, 6):
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
