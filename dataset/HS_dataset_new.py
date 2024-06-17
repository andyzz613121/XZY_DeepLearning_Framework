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
import configparser

class HS_dataset(Dataset.Dataset):
    def __init__(self, dataset=None, pca_flag = False, norm_flag = True, train_flag=True):
        self.dataset = dataset
        self.pca = pca_flag          
        self.train_flag = train_flag
        self.norm_flag = norm_flag

        self.names_list = []
        self.MAX_list = []
        self.size = 0
        self.sample = {'raw_image':[], 'img': [], 'label': []}
        
        # Read Configs
        HS_config = configparser.ConfigParser()
        HS_config.read('dataset\\Configs\\HS_Config.ini',encoding='UTF-8')
        HS_key_list = HS_config.sections()
        HS_value_list = []
        for item in HS_key_list:
            HS_value_list.append(HS_config.items(item))
        HS_config_dict = dict(zip(HS_key_list, HS_value_list))

        # --------------------------------------------------------------
        # 设置训练文件&最大值文件路径
        if dataset == 'Houston13':
            if pca_flag==False:
                print('Houston13: No PCA')
                self.max_file = HS_config_dict['Houston13'][2][1]
                self.csv_dir  = HS_config_dict['Houston13'][0][1]
            if pca_flag==True:
                print('Houston13: With PCA')
                self.max_file = HS_config_dict['Houston13'][3][1]
                self.csv_dir  = HS_config_dict['Houston13'][1][1]

        elif dataset == 'Houston18':
            if pca_flag==False:
                print('Houston18: No PCA')
                self.max_file = HS_config_dict['Houston18'][2][1]
                self.csv_dir  = HS_config_dict['Houston18'][0][1]
            if pca_flag==True:
                print('Houston18: With PCA')
                self.max_file = HS_config_dict['Houston18'][3][1]
                self.csv_dir  = HS_config_dict['Houston18'][1][1]

        elif dataset == 'Pavia':
            if pca_flag==False:
                print('Pavia: No PCA')
                self.max_file = HS_config_dict['Pavia'][2][1]
                self.csv_dir = HS_config_dict['Pavia'][0][1]
            else:
                print('Pavia: With PCA')
                self.max_file = HS_config_dict['Pavia'][3][1]
                self.csv_dir = HS_config_dict['Pavia'][1][1]
        
        elif dataset == 'Salinas':
            if pca_flag==False:
                print('Salinas: No PCA')
                self.max_file = HS_config_dict['Salinas'][2][1]
                self.csv_dir = HS_config_dict['Salinas'][0][1]
            else:
                print('Salinas: With PCA')
                self.max_file = HS_config_dict['Salinas'][3][1]
                self.csv_dir = HS_config_dict['Salinas'][1][1]

        else:
            print('ERROR: UnKnown dataset')
        # --------------------------------------------------------------
        # 读训练文件
        if self.train_flag == True:
            print("---Train %s Dataset---" %self.dataset)
            if not os.path.isfile(self.csv_dir):
                print(self.csv_dir + ':txt file does not exist!')
            file = open(self.csv_dir)
            for f in file:
                self.names_list.append(f)
                self.size += 1
        # --------------------------------------------------------------
        # 读最大值文件
        if not os.path.isfile(self.max_file):
            print(self.max_file + ':MAX FILE does not exist!')
        MAX_file = open(self.max_file)
        for max_f in MAX_file:
            self.MAX_list.append(max_f)
        # --------------------------------------------------------------

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.names_list[idx].split(',')[0]
        label = np.array(int(self.names_list[idx].split(',')[1].strip('\n')))
        return self.open_and_procress_data(img_path, label)
    
    def normalize(self, img):
        for b in range(len(self.MAX_list)):
            min, max = self.MAX_list[b].split(',')[0], self.MAX_list[b].split(',')[1].strip('\n')
            min, max = np.array(float(min)), np.array(float(max))
            img[b] = (img[b] - min)/(max - min)
        return img

    def open_and_procress_data(self, img_path, label):
        sample = {}
        img_raw = gdal.Open(img_path)
        img_w = img_raw.RasterXSize
        img_h = img_raw.RasterYSize
        img = np.array(img_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        
        if self.norm_flag == True:
            img = torch.from_numpy(self.normalize(img)).cuda()
        else:
            img = torch.from_numpy(img).cuda()

        raw_image = img

        if self.train_flag == False:
            img = img.unsqueeze(0)

        sample['raw_image']=raw_image
        sample['img']=img
        if label!=None:
            sample['label'] = torch.from_numpy(label).cuda()

        return sample

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
