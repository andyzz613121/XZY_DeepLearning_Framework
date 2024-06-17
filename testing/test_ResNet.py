import os
import sys
import time
import numpy as np
import xlwt
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

def test(model_img):
    batch_size = 32

    train_dst = classification_dataset('E:\\dataset\\GF2\\train\\GF2.csv')
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = False)

    # model_img = torch.load('D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\image_model105.pkl').cuda()
    # =====  eval  =====
    model_img.eval()
    total_num = 0
    true_num = 0
    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet('M')
    excel_file = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\1.xls'
    line_num = 0
    #==========   Train Loop   ==========#
    for i, sample in enumerate(train_loader, 0):
        img=sample['img']
        label=sample['label']
        name = sample['name']
        
        out = nn.Softmax()(model_img(img))
        prob_0 = out[:,0].cpu().detach().numpy()
        prob_1 = out[:,1].cpu().detach().numpy()
        # prob_2 = out[:,2].cpu().detach().numpy()
        # prob_3 = out[:,3].cpu().detach().numpy()
        
        # index_1 = (label != 1)
        index_23 = (label > 1)
        index_1 = (label == 1)
        label[index_1] = 0
        label[index_23] = 1
        label = torch.reshape(label, [label.shape[0], -1])
        label, _ = torch.max(label, 1)
        
        pre = torch.argmax(out, 1).cpu().detach().numpy()

        for item in range(pre.shape[0]):
            pre_item = pre[item]
            out_item = out[item][1].cpu().detach().numpy()
            label_item = label[item]
            prob_0_item = prob_0[item]
            prob_1_item = prob_1[item]
            # prob_2_item = prob_2[item]
            # prob_3_item = prob_3[item]
            worksheet.write(line_num, 0, label = name[item])
            worksheet.write(line_num, 1, label = int(pre_item))
            worksheet.write(line_num, 2, label = int(label_item))
            worksheet.write(line_num, 3, label = float(prob_0_item))
            worksheet.write(line_num, 4, label = float(prob_1_item))
            line_num += 1
            # if pre_item != label_item:

            #     # worksheet.write(line_num, 0, label = name[item])
            #     # worksheet.write(line_num, 1, label = int(pre_item))
            #     # worksheet.write(line_num, 2, label = int(label_item))
            #     # worksheet.write(line_num, 3, label = str(out_item))

            #     # worksheet.write(line_num, 3, label = float(prob_0_item))
            #     # worksheet.write(line_num, 4, label = float(prob_1_item))
            #     # worksheet.write(line_num, 5, label = float(prob_2_item))
            #     # worksheet.write(line_num, 6, label = float(prob_3_item))

            #     line_num += 1
            workbook.save(excel_file)
            print(name[item], float(prob_0_item), float(prob_1_item))
    print(line_num)
    #     for item in range(pre.shape[0]):
    #         pre_item = pre[item]
    #         label_item = label[item]
    #         if label_item != 0:
    #             total_num += 1
    #             if pre_item == label_item:
    #                 true_num += 1

    # print(true_num, total_num, true_num/total_num)
        
if __name__ == '__main__':
    for i in range(165, 0, -1):
        print('model', i)
        name = 'D:\\Code\\ImageRetrieval\\pkl\\Special_Model_1\\image_model163.pkl'
        test(torch.load(name).cuda())
        break
    # name = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\1\\image_model175.pkl'
    # test(torch.load(name).cuda())