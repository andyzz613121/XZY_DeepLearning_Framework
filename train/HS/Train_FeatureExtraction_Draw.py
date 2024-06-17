
import sys
import time
import numpy as np

from torch.utils import data
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.nn import functional as F


base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from dataset.HS_dataset_new import HS_dataset
from data_processing.Raster import gdal_write_tif
from visual.Draw import draw_heatmap
from model.HyperSpectral.Auto_Multiply.AutoContrast_ASPP import AutoContrast_ASPP
def main(times, dataset_num):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 16
    init_lr = 0.001
    PCA = False
    norm = True


    dataset = 'Pavia'

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

    train_dst = HS_dataset(dataset, PCA, norm)
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = False)
    filename = 'result\\HyperSpectral\\Mlp_Auto\\bandwise\\多像素合成\\Bandwise+ReLU\\ratio4，scale1_2_4, 2Dx2D，5X5窗口，特征图求均值，3个波段，bandwise_mlp2层\\Single Scale\\Base_GSF和LSF都2层\\' + dataset +'\\0\\image_model_xzy500.pkl'
    model_img = torch.load(filename).cuda()
    # model_img = AutoContrast_ASPP(24, output_channels).cuda()
    # model_img = AutoContrast_ASPP(3, output_channels).cuda() 

    # Restore
    iii = 0
    out_flag = np.zeros(output_channels)
        
    model_img.eval()
    for i, sample in enumerate(train_loader, 0):
        
        img=sample['img']
        label=sample['label']

        out_list = model_img(img)[2][0]
        # out_list = F.interpolate(out_list,scale_factor=(2,2),mode='bilinear')
        img = model_img(img)[3][0]
        from visual import Draw
        for b in range(16):
            de = out_list[b][0].cpu().detach().numpy()
            imgs = img[b].cpu().detach().numpy()
            lab = label[b].cpu().detach().numpy()

            if out_flag[lab] == 0:
                path1 = 'result\\Pic\\lab' + str(lab) + '_' + str(iii) + '_norm.tif'
                path2 = 'result\\Pic\\lab' + str(lab) + '_' + str(iii) + '.png'
                path3 = 'result\\Pic\\lab' + str(lab) + '_' + str(iii) + '.jpg'
                path4 = 'result\\Pic\\lab' + str(lab) + '_' + str(iii) + '.tif'
                de1 = (de-np.min(de))/(np.max(de)-np.min(de))
                gdal_write_tif(path1, de1, de.shape[1], de.shape[0], datatype=2)
                gdal_write_tif(path4, de, de.shape[1], de.shape[0], datatype=2)
                draw_heatmap(de, path3, norm=False)
                Draw.draw_curve([x for x in range(len(imgs))], imgs, path2)
                
                # img_ct = imgs
                # img_ct = np.interp(np.arange(0, len(img_ct), 0.09), np.arange(0, len(img_ct)), img_ct)

                # img_ct = np.reshape(img_ct, [40, 40])
                # # gdal_write_tif(path1, img_ct, 10, 144, datatype=2)
                # draw_heatmap(img_ct, path3)
                iii += 1
                out_flag[lab] = 1


if __name__ == '__main__':
    import os
    main(1, 1)

    
