
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
from model.HyperSpectral.Cascade.Cascade import Cascade_3D2D
from model.HyperSpectral.SPSA_Att.SPSA_Att import spsa_att
from model.HyperSpectral.Grid.SpectralImage_GridaAtten import HS_SI_3D_Grid_ATTEN
from model.HyperSpectral.CMAttention.SpectralImage_CM import HS_SI_CM
from testing.HS.test_HS import testHS

def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 32
    init_lr = 0.001
    PCA = False
    norm = True

    dataset = 'Houston13'

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

    model_img = HS_SI_CM(input_channels, output_channels).cuda()
    # model_img = torch.load('result\\image_model500.pkl').cuda()

    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': 1*init_lr}
    ], lr=init_lr, momentum=0.9, weight_decay=1e-4)

    loss_function = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    # Restore
    total_epoch = 500
    cur_itrs = 0
    cur_epochs = 0
    iii = 0
    #==========   Train Loop   ==========#
    while True: #cur_itrs < opts.total_itrs:
        cur_epochs += 1
        interval_loss = 0
        total_num = 0
        
        model_img.train()
        model_img.CM_iter = torch.zeros([output_channels, output_channels]).cuda()
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            cur_itrs += 1
            
            img=sample['img']
            label=sample['label']

            out_list = model_img(img)

            loss = 0
            for outs in out_list:
                loss += loss_function(outs, label.long())
                pre = torch.argmax(outs, 1)
                for b in range(pre.shape[0]):
                    if pre[b]!=label[b]:
                        model_img.CM_iter[pre[b]][label[b]] += 1
                        total_num+=1
            
            loss.backward()
            interval_loss += loss.item()
            optimizer.step()

        scheduler.step()

        model_img.CM_epoch = model_img.CM_iter/torch.max(model_img.CM_iter)
        # print(model_img.CM_epoch, model_img.CM_iter)
        print('--------epoch %d done --------'%cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs in [200, 250, 300, 350, 400, 450, 500]:
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
            model_img.eval()
            path = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\' + str(cur_epochs)
            testHS(dataset, model_img, path, norm, PCA)

        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()

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


# from visual import Draw
            # from data_processing.Raster import gdal_write_tif
            # for b in range(out_list[0].shape[0]):
            #     de = out_list[0][b].cpu().detach().numpy()*10
            #     lab = label[b].cpu().detach().numpy()
            #     path1 = 'result\\lab' + str(lab) + '_' + str(iii) + '.tif'
            #     print(path1)
            #     gdal_write_tif(path1, de, 144, 144,datatype=2)
            #     iii += 1