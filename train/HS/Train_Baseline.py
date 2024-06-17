
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from dataset.HS_dataset_new import HS_dataset
from model.HyperSpectral.Baseline.Baselines import HuEtAl
from model.HyperSpectral.Baseline import SpecAttenNet, HybridSN, CNN_2D, CNN_3D, SSRN, FDSSC, DBDA
from testing.HS.test_HS import testHS

def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 32
    init_lr = 0.001
    PCA = False
    norm = True

    dataset = 'Pavia'

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

    train_dst = HS_dataset(dataset, PCA, norm)

    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    # model_img = SpecAttenNet.SpecAttenNet(input_channels, output_channels).cuda()
    # model_img = CNN_2D.CNN2D(input_channels, output_channels).cuda()
    # model_img = HybridSN.HybridSN(output_channels).cuda()
    # model_img = FDSSC.FDSSC(input_channels, output_channels).cuda()
    # model_img = CNN_3D.CNN3D(input_channels, output_channels).cuda()
    # model_img = DBDA.DBDA_network(input_channels, output_channels).cuda()
    model_img = HuEtAl(input_channels, output_channels).cuda()
    
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

            img=sample['img']
            # img = sample['pca']
            label=sample['label']

            out = model_img(img)[0]

            loss = loss_function(out, label.long())
            loss.backward()
            interval_loss += loss.item()
            optimizer.step()
            
        scheduler.step()
        print('--------epoch %d done --------'%cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])

            
        if cur_epochs in [500]:
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
            model_img.eval()
            path = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\' + str(cur_epochs)
            testHS(dataset, model_img, path, norm, PCA)

        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()
