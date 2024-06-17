import torch
from torch import nn
import torch.utils.data.dataset as Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from osgeo import gdal
from data_processing.layer_data_augmentator import DataAugmentation

class HED_dataset(Dataset.Dataset):
    def __init__(self, csv_dir, gpu=True):
        self.csv_dir = csv_dir          
        self.names_list = []
        self.size = 0
        self.gpu = gpu
        if not os.path.isfile(self.csv_dir):
            print(self.csv_dir + ':txt file does not exist!')

        file = open(self.csv_dir)
        
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.names_list[idx].split(',')[0]
        img = gdal.Open(img_path)
        img_w = img.RasterXSize
        img_h = img.RasterYSize
        # dsm_path = self.names_list[idx].split(',')[1]
        # dsm = gdal.Open(dsm_path)
        # ndsm_path = self.names_list[idx].split(',')[2]
        # ndsm = gdal.Open(ndsm_path)
        label_path = self.names_list[idx].split(',')[1]
        label = gdal.Open(label_path)
        label_class_path = self.names_list[idx].split(',')[2].strip('\n')
        label_class = gdal.Open(label_class_path)

        img = np.array(img.ReadAsArray(0,0,img_w,img_h)).astype('float32')
        # dsm = np.array(dsm.ReadAsArray(0,0,img_w,img_h)).astype('float32')
        # ndsm = np.array(ndsm.ReadAsArray(0,0,img_w,img_h)).astype('float32')
        label = np.array(label.ReadAsArray(0,0,img_w,img_h))
        label_class = np.array(label_class.ReadAsArray(0,0,img_w,img_h))

        #数据增广
        Data_Aug = DataAugmentation()
        rotate_flag = np.random.randint(4) #0,90,180,270
        flip_flag = np.random.randint(4)   #no,horizontal,vertical,all
        img = Data_Aug.apply_augmentation(img, rotate_flag, flip_flag)
        # dsm = Data_Aug.apply_augmentation(dsm, rotate_flag, flip_flag)
        # ndsm = Data_Aug.apply_augmentation(ndsm, rotate_flag, flip_flag)
        label = Data_Aug.apply_augmentation(label, rotate_flag, flip_flag)
        label_class = Data_Aug.apply_augmentation(label_class, rotate_flag, flip_flag)
        #数据归一化
        # img = (img-np.min(img))/(np.max(img)-np.min(img))
        # if np.max(dsm)==np.min(dsm):
        #     if np.max(dsm)!=0:
        #         dsm = dsm/np.max(dsm)
        # else:
        #     dsm = (dsm-np.min(dsm))/(np.max(dsm)-np.min(dsm))

        # if np.max(ndsm)==np.min(ndsm):
        #     if np.max(ndsm)!=0:
        #         ndsm = ndsm/np.max(ndsm)
        # else:
        #     ndsm = (ndsm-np.min(ndsm))/(np.max(ndsm)-np.min(ndsm))

        # dsm_ndsm = np.array([dsm, ndsm])

        img = torch.from_numpy(img)
        raw_image = img
        # dsm_ndsm = torch.from_numpy(dsm_ndsm)

        label = torch.from_numpy(label)
        label = label.contiguous().view(1,label.size()[0],label.size()[1])
        label_class = torch.from_numpy(label_class.astype(np.uint8)).long()
        #label_class = label_class.contiguous().view(1,label_class.size()[0],label_class.size()[1])
        #数据标准化Normalize
        norm_list = [0.5] * img.shape[0]
        img = transforms.Normalize(mean=norm_list,std=norm_list)(img)
        # dsm_ndsm = transforms.Normalize(mean=[0.5,0.5],std=[0.5,0.5])(dsm_ndsm)

        if self.gpu == True:
            img = img.cuda()
            label = label.cuda()
            label_class = label_class.cuda()

        # sample = {'raw_image':raw_image, 'img': img, 'dsm_ndsm': dsm_ndsm, 'label': label, 'label_class': label_class}
        sample = {'raw_image':raw_image, 'img': img, 'label': label, 'label_class': label_class}
        return sample