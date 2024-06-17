import torch
from torch import nn
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image 
import random
import os
import cv2
import torch.nn.functional as F
import numpy as np
from osgeo import gdal
from data_processing.layer_data_augmentator import DataAugmentation

class HS_dataset(Dataset.Dataset):
    def __init__(self, dataset=None, pca_flag = False, norm_flag = True, train_flag=True):
        self.dataset = dataset
        self.pca = pca_flag          
        self.train_flag = train_flag
        self.norm_flag = norm_flag

        self.names_list = []
        self.MAX_list = []
        self.names_list_pca = []
        self.MAX_list_pca = []
        self.size = 0
        self.sample = {'raw_image':[], 'img': [], 'label': []}
        
        # --------------------------------------------------------------
        # 设置训练文件&最大值文件路径
        if dataset == 'Houston13':
            self.max_file = 'E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\data\\max_Houston.txt'
            self.csv_dir  = 'E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\data\\label.csv'
            if pca_flag==True:
                print('Houston13: With PCA')
                self.max_file_pca = 'E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\data_pca\\max_Houston13_pca.txt'
                self.csv_dir_pca  = 'E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\data_pca\\label.csv'
            else:
                print('Houston13: No PCA')

        elif dataset == 'Houston18':
            self.max_file = 'E:\\dataset\\高光谱数据集\\2018IEEE_Contest\\Train\\data\\max_Houston2018.txt'
            self.csv_dir  = 'E:\\dataset\\高光谱数据集\\2018IEEE_Contest\\Train\\data\\label.csv'
            if pca_flag==True:
                print('Houston18: With PCA')
                self.max_file_pca = 'E:\\dataset\\高光谱数据集\\2018IEEE_Contest\\Train\\data_pca\\max_Houston2018_pca.txt'
                self.csv_dir_pca  = 'E:\\dataset\\高光谱数据集\\2018IEEE_Contest\\Train\\data_pca\\label.csv'
            else:
                print('Houston18: No PCA')

        elif dataset == 'Pavia':
            self.max_file = 'E:\\dataset\\高光谱数据集\\Pavia\\Train\\data\\max_Pavia.txt'
            self.csv_dir = 'E:\\dataset\\高光谱数据集\\Pavia\\Train\\data\\label.csv'
            if pca_flag==True:
                print('Pavia: With PCA')
                self.max_file_pca = 'E:\\dataset\\高光谱数据集\\Pavia\\Train\\data_pca\\max_Pavia_pca.txt'
                self.csv_dir_pca = 'E:\\dataset\\高光谱数据集\\Pavia\\Train\\data_pca\\label.csv'
            else:
                print('Pavia: No PCA')
        
        elif dataset == 'Salinas':
            self.max_file = 'E:\\dataset\\高光谱数据集\\Salinas\\Train\\data\\max_Salinas.txt'
            self.csv_dir = 'E:\\dataset\\高光谱数据集\\Salinas\\Train\\data\\label.csv'
            if pca_flag==True:
                print('Salinas: With PCA')
                self.max_file_pca = 'E:\\dataset\\高光谱数据集\\Salinas\\Train\\data_pca\\max_Salinas_pca.txt'
                self.csv_dir_pca = 'E:\\dataset\\高光谱数据集\\Salinas\\Train\\data_pca\\label.csv'
            else:
                print('Salinas: No PCA')
        else:
            print('ERROR: UnKnown dataset')
        # --------------------------------------------------------------
        # 读训练文件
        if self.train_flag == True:
            # 读非PCA的训练文件
            if not os.path.isfile(self.csv_dir):
                print(self.csv_dir + ':txt file does not exist!')
            file = open(self.csv_dir)
            for f in file:
                self.names_list.append(f)
                self.size += 1
            
            if self.pca == True:
                if not os.path.isfile(self.csv_dir_pca):
                    print(self.csv_dir_pca + ':PCA txt file does not exist!')
                file = open(self.csv_dir_pca)
                for f in file:
                    self.names_list_pca.append(f)
        # --------------------------------------------------------------
        # 读最大值文件
        # 读非PCA的最大值文件
        if not os.path.isfile(self.max_file):
            print(self.max_file + ':MAX FILE does not exist!')
        MAX_file = open(self.max_file)
        for max_f in MAX_file:
            self.MAX_list.append(max_f)

        # 读PCA的最大值文件
        if self.pca == True:
            if not os.path.isfile(self.max_file_pca):
                print(self.max_file_pca + ':PCA MAX FILE does not exist!')
            MAX_file = open(self.max_file_pca)
            for max_f in MAX_file:
                self.MAX_list_pca.append(max_f)
        # --------------------------------------------------------------

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.names_list[idx].split(',')[0]
        if self.pca == True:
            pca_path = self.names_list_pca[idx].split(',')[0]
        else:
            pca_path = None
        label = np.array(int(self.names_list[idx].split(',')[1].strip('\n')))
        return self.open_and_procress_data(img_path, pca_path, label)
    
    def normalize(self, img):
        for b in range(len(self.MAX_list)):
            min, max = self.MAX_list[b].split(',')[0], self.MAX_list[b].split(',')[1].strip('\n')
            min, max = np.array(float(min)), np.array(float(max))
            img[b] = (img[b] - min)/(max- min)
        return img

    def open_and_procress_data(self, img_path, pca_path, label):
        sample = {}
        # 对普通高光谱影像的处理
        img = self.open_img(img_path)
        if self.norm_flag == True:
            img = torch.from_numpy(self.normalize(img)).cuda()
        else:
            img = torch.from_numpy(img).cuda()
        raw_image = img
        if self.train_flag == False:
            img = img.unsqueeze(0)

        # 对PCA影像的处理
        if self.pca == True:
            pca_img = self.open_img(pca_path)
            pca_img = torch.from_numpy(pca_img).cuda()
            if self.train_flag == False:
                pca_img = pca_img.unsqueeze(0)
        else:
            pca_img = None

        sample['raw_image']=raw_image
        sample['img']=img
        sample['pca']=pca_img
        if label!=None:
            sample['label'] = torch.from_numpy(label).cuda()

        return sample

    def open_img(self, img_path):
        img_raw = gdal.Open(img_path)
        img_w = img_raw.RasterXSize
        img_h = img_raw.RasterYSize
        img = np.array(img_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        return img
#先初始化前面
def add_conv_channels(model, premodel, conv_num):
    model_dict = model.state_dict()
    premodel_dict = premodel.state_dict()

    for i in range(conv_num[0]):
        conv = torch.FloatTensor(64,1,3,3).cuda()
        nn.init.xavier_normal_(conv)

        orginal1 = premodel_dict['conv_1.0.weight']
        new = torch.cat([orginal1,conv],1)
        premodel_dict['conv_1.0.weight'] = new

    model.load_state_dict(premodel_dict)
    print('set model with predect model, add channel is ',conv_num)
    return model
