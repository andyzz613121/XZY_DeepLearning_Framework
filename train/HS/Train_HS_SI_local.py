
import sys
import time
import numpy as np

from torch.utils import data
import torchvision.transforms as transforms
import torch
import torch.nn as nn

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from dataset.HS_dataset import HS_dataset
from model.HyperSpectral.SpectralImage_local import HS_SI_LOCAL
from model.HyperSpectral.SpectralImage_local_ASPP import HS_SI_LOCAL_ASPP
from model.HyperSpectral.SpectralImage_ensemble import HS_SI_ENSEMBLE
from testing.HS.test_HS import testHS

def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 64
    init_lr = 0.001

    # dataset = 'Pavia'
    # input_channels = 103
    # output_channels = 9
    # train_dst = HS_dataset('E:\\dataset\\高光谱数据集\\Pavia\\Train\\data\\label.csv')

    PCA_flag = False
    dataset = 'Houston'
    
    output_channels = 15
    if PCA_flag == True:
        train_dst = HS_dataset('E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\data_pca\\label.csv')
        input_channels = 30
    else:
        train_dst = HS_dataset('E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\data\\label.csv')
        input_channels = 144

    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    model_img = HS_SI_LOCAL_ASPP(input_channels, output_channels).cuda()

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
            label=sample['label']
            out_list = model_img(img)
            loss = 0
            for outs in out_list:
                loss += loss_function(outs, label.long())
            loss.backward()
            interval_loss += loss.item()
            optimizer.step()
            
        scheduler.step()
        print('--------epoch %d done --------'%cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f' %(optimizer.param_groups[0]['lr']))
            
        if cur_epochs in [1, 200, 250, 300, 350, 400, 450, 500]:
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
            model_img.eval()
            path = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\' + str(cur_epochs)
            testHS(dataset, model_img, path, PCA_flag)

        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()
