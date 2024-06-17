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
from dataset.layer_data_augmentator import DataAugmentation
from model.priori_knowledge import DIS_MAP

class ISPRS_dataset(Dataset.Dataset):
    def __init__(self, csv_dir, dataset_name, compress=False, gramma=False, gpu=True, train_flag=True):
        self.csv_dir = csv_dir          
        self.names_list = []
        self.size = 0
        self.gpu = gpu
        self.img_num = 0
        self.dataset_name = dataset_name
        self.compress = compress
        self.gramma = gramma
        self.train_flag = train_flag
        assert self.dataset_name == 'Vaihingen' or self.dataset_name == 'Potsdam', "ERROR:: ISPRS_dataset_Laplace//dataset_name"
        
        if self.train_flag == True:
            print("---Train Dataset---")
            print('---ISPRS Dataset(%s), compress(%s), gramma(%s)---'%(dataset_name, compress, gramma))
            if not os.path.isfile(self.csv_dir):
                print(self.csv_dir + ':txt file does not exist!')
            file = open(self.csv_dir)
            for f in file:
                self.names_list.append(f)
                self.size += 1

        elif self.train_flag == False:
            print('---Test ISPRS Dataset(%s)---'%(dataset_name))


    def __len__(self):
        return self.size

    def open_and_procress_data(self, dataset_name, img_path, dsm_path, dis_path, label_path):
        img_raw = gdal.Open(img_path)
        img_w = img_raw.RasterXSize
        img_h = img_raw.RasterYSize
        dsm_raw = gdal.Open(dsm_path)
        ndsm_raw = gdal.Open(dis_path)
        label_raw = gdal.Open(label_path)

        img = np.array(img_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        dsm = np.array(dsm_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        ndsm = np.array(ndsm_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        label = np.array(label_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('uint8')
        sample = {'raw_image':[], 'img': [], 'dsm_ndsm': [], 'label': [], 'SRM': []}

        if self.train_flag == True:
            if (self.compress==True):
                img = compress_graylevel(img, 256, 64)
            if (self.gramma==True):
                gamma_param = np.random.uniform(0.5, 1.5)
                img = gamma_transform(img, gamma_param)

            #数据增广
            Data_Aug = DataAugmentation()
            Trans = Data_Aug.get_random_transform_params(img)
            img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
            img = cv2.warpPerspective(img, Trans, dsize=(img.shape[0], img.shape[1]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
            dsm = cv2.warpPerspective(dsm, Trans, dsize=(img.shape[1], img.shape[2]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            ndsm = cv2.warpPerspective(ndsm, Trans, dsize=(img.shape[1], img.shape[2]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpPerspective(label, Trans, dsize=(img.shape[1], img.shape[2]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        raw_image = img
        
        #DIS
        DIS = DIS_MAP()
        dis = DIS.Gen_Dis_Map(label)

        #NDVI
        if dataset_name == 'Vaihingen':
            i = 1
            #NDVI Vaihingen
            zero_0_index = (img[0,:,:]==0)
            zero_1_index = (img[1,:,:]==0)
            zero_2_index = (img[2,:,:]==0)
            img[0,zero_0_index] = 1
            img[1,zero_1_index] = 1
            img[2,zero_2_index] = 1
            ndvi = (img[0,:,:] - img[1,:,:])/(img[0,:,:] + img[1,:,:])
            ndwi = (img[2,:,:] - img[0,:,:])/(img[2,:,:] + img[0,:,:])
        elif dataset_name == 'Potsdam':
            #NDVI Potsdam
            zero_0_index = (img[0,:,:]==0)
            zero_1_index = (img[1,:,:]==0)
            zero_2_index = (img[2,:,:]==0)
            zero_3_index = (img[3,:,:]==0)
            img[0,zero_0_index] = 1
            img[1,zero_1_index] = 1
            img[2,zero_2_index] = 1
            img[3,zero_3_index] = 1
            ndvi = (img[3,:,:] - img[0,:,:])/(img[3,:,:] + img[0,:,:])
            ndwi = (img[1,:,:] - img[3,:,:])/(img[1,:,:] + img[3,:,:])

        #数据归一化
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        
        ndvi = (ndvi-np.min(ndvi))/(np.max(ndvi)-np.min(ndvi))
        ndwi = (ndwi-np.min(ndwi))/(np.max(ndwi)-np.min(ndwi))

        
        if np.max(dsm)==np.min(dsm):
            if np.max(dsm)!=0:
                dsm = dsm/np.max(dsm)
        else:
            dsm = (dsm-np.min(dsm))/(np.max(dsm)-np.min(dsm))
        if np.max(ndsm)==np.min(ndsm):
            if np.max(ndsm)!=0:
                ndsm = ndsm/np.max(ndsm)
        else:
            ndsm = (ndsm-np.min(ndsm))/(np.max(ndsm)-np.min(ndsm))
        # dsm_ndsm = np.array([dsm, ndsm])

        # if np.max(dis)==np.min(dis):
        #     if np.max(dis)!=0:
        #         dis = dis/np.max(dis)
        # else:
        #     dis = (dis-np.min(dis))/(np.max(dis)-np.min(dis))


        img = torch.from_numpy(img)
        raw_image = img
        dsm = torch.from_numpy(dsm).unsqueeze(0)
        label = torch.from_numpy(label)
        label = label.contiguous().view(label.size()[0],label.size()[1])
        # dis = torch.from_numpy(dis).unsqueeze(0)
        dis = torch.from_numpy(dis)
        ndvi = torch.from_numpy(ndvi).unsqueeze(0)
        ndwi = torch.from_numpy(ndwi).unsqueeze(0)

        #数据标准化Normalize
        if dataset_name == 'Vaihingen':
            img = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(img)
            dsm = transforms.Normalize(mean=[0.5],std=[0.5])(dsm)
            # dis = transforms.Normalize(mean=[0.5],std=[0.5])(dis)
            ndvi = transforms.Normalize(mean=[0.5],std=[0.5])(ndvi)
            ndwi = transforms.Normalize(mean=[0.5],std=[0.5])(ndwi)
        elif dataset_name == 'Potsdam':
            img = transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5, 0.5])(img)
            # dis = transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5])(dis)
           
        if self.train_flag == False:
            img = img.unsqueeze(0)
            dsm = dsm.unsqueeze(0)
            dis = dis.unsqueeze(0)
            ndvi = ndvi.unsqueeze(0)
            ndwi = ndwi.unsqueeze(0)
            label = label.unsqueeze(0)
        
        if self.gpu == True:
            img = img.cuda()
            dsm = dsm.cuda()
            dis = dis.cuda()
            ndvi = ndvi.cuda()
            ndwi = ndwi.cuda()
            label = label.cuda()

        sample['raw_image']=raw_image
        sample['img']=img
        sample['dsm']=dsm
        sample['label']=label
        sample['dis'] = dis
        sample['ndvi'] = ndvi
        sample['ndwi'] = ndwi
        sample['image_name'] = img_path
        sample['label_name'] = label_path
        return sample
    def __getitem__(self, idx):
        basename = ''
        img_path = basename + self.names_list[idx].split(',')[0]
        dsm_path = basename + self.names_list[idx].split(',')[1]
        dis_path = basename + self.names_list[idx].split(',')[2]
        label_path = basename + self.names_list[idx].split(',')[3].strip('\n')
        
        return self.open_and_procress_data(self.dataset_name, img_path,dsm_path,dis_path,label_path)  

class DeepGlobe_dataset(Dataset.Dataset):
    def __init__(self, csv_dir, compress=False, gramma=False, gpu=True):
        self.csv_dir = csv_dir          
        self.names_list = []
        self.size = 0
        self.gpu = gpu
        self.img_num = 0
        self.compress = compress
        self.gramma = gramma
        
        print('---DeepGlobe dataset, compress(%s), gramma(%s)---'%(compress, gramma))
        if not os.path.isfile(self.csv_dir):
            print(self.csv_dir + ':txt file does not exist!')

        file = open(self.csv_dir)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        basename = ''
        img_path = basename + self.names_list[idx].split(',')[0]
        img_raw = gdal.Open(img_path)
        img_w = img_raw.RasterXSize
        img_h = img_raw.RasterYSize
        label_path = basename + self.names_list[idx].split(',')[1].strip('\n')
        label_raw = gdal.Open(label_path)

        sample = {'raw_image':[], 'img': [], 'label': []}
        
        img = np.array(img_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        label = np.array(label_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('uint8')

        if (self.compress==True):
            img = compress_graylevel(img, 256, 64)
        if (self.gramma==True):
            gamma_param = np.random.uniform(0.5, 1.5)
            img = gamma_transform(img, gamma_param)

        #数据增广
        Data_Aug = DataAugmentation()
        Trans = Data_Aug.get_random_transform_params(img)
        img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
        img = cv2.warpPerspective(img, Trans, dsize=(img.shape[0], img.shape[1]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        label = cv2.warpPerspective(label, Trans, dsize=(img.shape[1], img.shape[2]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        raw_image = img
        
        #Laplace
        img_t = img.transpose(1, 2, 0)
        img_Laplace = cv2.Laplacian(img_t, cv2.CV_32F) #(256, 256, 3)
        img_Laplace = img_Laplace.transpose(2, 0, 1)

        # #Sobel
        # img_Sobel_x = cv2.Sobel(img_t, cv2.CV_32F, 1, 0)
        # img_Sobel_x = img_Sobel_x.transpose(2, 0, 1)
        # img_Sobel_y = cv2.Sobel(img_t, cv2.CV_32F, 0, 1)
        # img_Sobel_y = img_Sobel_y.transpose(2, 0, 1)


        #数据归一化
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        img_Laplace = (img_Laplace-np.min(img_Laplace))/(np.max(img_Laplace)-np.min(img_Laplace))
        # img_Sobel_x = (img_Sobel_x-np.min(img_Sobel_x))/(np.max(img_Sobel_x)-np.min(img_Sobel_x))
        # img_Sobel_y = (img_Sobel_y-np.min(img_Sobel_y))/(np.max(img_Sobel_y)-np.min(img_Sobel_y))
        # img_Sobel = np.concatenate([img_Sobel_x, img_Sobel_y], 0)


        img = torch.from_numpy(img)
        raw_image = img
        img_Laplace = torch.from_numpy(img_Laplace)
        # img_Sobel = torch.from_numpy(img_Sobel)
        label = torch.from_numpy(label)
        label = label.contiguous().view(label.size()[0],label.size()[1])

        #数据标准化Normalize
        img = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(img)
        img_Laplace = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(img_Laplace)
        # img_Sobel = transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5])(img_Sobel)

        if self.gpu == True:
            img = img.cuda()
            img_Laplace = img_Laplace.cuda()
            # img_Sobel = img_Sobel.cuda()
            label = label.cuda()

        sample['raw_image']=raw_image
        sample['img']=img
        sample['label']=label
        sample['img_Laplace'] = img_Laplace
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

def gamma_transform(img, gamma=0.8):
    img /= 255
    img = np.power(img, gamma)
    img *= 255
    return img

def compress_graylevel(img, input_graylevel, output_graylevel):
    print("---doing compress_graylevel---")
    rate = input_graylevel/output_graylevel
    img = img//rate
    img = img*rate
    return img

def main():
    print('1')
    return 1
if __name__ == '__main__':
    main()
