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

        self.imgH_test = 0
        self.imgW_test = 0
        
        if self.train_flag == True:
            print("---Train Dataset---")
            print('---Dataset(%s), compress(%s), gramma(%s)---'%(dataset_name, compress, gramma))
            if not os.path.isfile(self.csv_dir):
                print(self.csv_dir + ':txt file does not exist!')
            file = open(self.csv_dir)
            for f in file:
                self.names_list.append(f)
                self.size += 1

        elif self.train_flag == False:
            print('---Test Dataset(%s)---'%(dataset_name))


    def __len__(self):
        return self.size

    def open_and_procress_data(self, dataset_name, img_path, dsm_path, dis_path, label_path, edge_path):
        img_raw = gdal.Open(img_path)
        img_w = img_raw.RasterXSize
        img_h = img_raw.RasterYSize
        self.imgH_test = img_h
        self.imgW_test = img_w
        dsm_raw = gdal.Open(dsm_path)
        dis_raw = gdal.Open(dis_path)
        label_raw = gdal.Open(label_path)
        edge_raw = gdal.Open(edge_path)

        img = np.array(img_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        dsm = np.array(dsm_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        dis = np.array(dis_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        label = np.array(label_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('uint8')
        edge = np.array(edge_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')

        sample = {'raw_image':[], 'img': [], 'dsm': [], 'dis': [], 'label': [], 'edge': []}


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
            dis = cv2.warpPerspective(dis, Trans, dsize=(img.shape[1], img.shape[2]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            edge = cv2.warpPerspective(edge, Trans, dsize=(img.shape[1], img.shape[2]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpPerspective(label, Trans, dsize=(img.shape[1], img.shape[2]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)


        raw_image = img
        if dataset_name == 'Vaihingen':
            #NDVI Vaihingen
            zero_0_index = (img[0,:,:]==0)
            zero_1_index = (img[1,:,:]==0)
            zero_2_index = (img[2,:,:]==0)
            img[0,zero_0_index] = 1
            img[1,zero_1_index] = 1
            img[2,zero_2_index] = 1
            ndvi = (img[0,:,:] - img[1,:,:])/(img[0,:,:] + img[1,:,:])

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
            
    
        #数据归一化
        img = (img-np.min(img))/(np.max(img)-np.min(img))

        if np.max(dsm)==np.min(dsm):
            if np.max(dsm)!=0:
                dsm = dsm/np.max(dsm)
        else:
            dsm = (dsm-np.min(dsm))/(np.max(dsm)-np.min(dsm))

        if np.max(dis)==np.min(dis):
            if np.max(dis)!=0:
                dis = dis/np.max(dis)
        else:
            dis = (dis-np.min(dis))/(np.max(dis)-np.min(dis))


        img = torch.from_numpy(img)
        dsm = torch.from_numpy(dsm).unsqueeze(0)
        label = torch.from_numpy(label)
        label = label.contiguous().view(label.size()[0],label.size()[1])
        ndvi = torch.from_numpy(ndvi).unsqueeze(0)

        
        #数据标准化Normalize
        if dataset_name == 'Vaihingen':
            img = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(img)
            
        elif dataset_name == 'Potsdam':
            img = transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5, 0.5])(img)
            
        dsm = transforms.Normalize(mean=[0.5],std=[0.5])(dsm)
        ndvi = transforms.Normalize(mean=[0.5],std=[0.5])(ndvi)


        if self.train_flag == False:
            img = img.unsqueeze(0)
            dsm = dsm.unsqueeze(0)
            ndvi = ndvi.unsqueeze(0)
            label = label.unsqueeze(0)
        
        if self.gpu == True:
            img = img.cuda()
            dsm = dsm.cuda()
            ndvi = ndvi.cuda()
            label = label.cuda()


        sample['raw_image']=raw_image
        sample['img']=img
        sample['dsm']=dsm
        sample['label']=label
        sample['ndvi'] = ndvi

        return sample
    def __getitem__(self, idx):
        basename = ''
        img_path = basename + self.names_list[idx].split(',')[0]
        dsm_path = basename + self.names_list[idx].split(',')[1]
        dis_path = basename + self.names_list[idx].split(',')[2]
        label_path = basename + self.names_list[idx].split(',')[3]
        edge_path = basename + self.names_list[idx].split(',')[4].strip('\n')
        
        return self.open_and_procress_data(self.dataset_name, img_path,dsm_path,dis_path,label_path,edge_path)  

class RS_train_dataset(Dataset.Dataset):
    def __init__(self, csv_dir=None, gpu=True, train_flag=True, DataAug=False, norm=True):
        self.csv_dir = csv_dir          
        self.names_list = []
        self.size = 0
        self.gpu = gpu
        self.train_flag = train_flag
        self.DataAug = DataAug
        self.norm = norm

        if self.train_flag == True:
            print("---Train RS Dataset---")
        elif self.train_flag == False:
            print("---Test RS Dataset---")

        if not os.path.isfile(self.csv_dir):
            print(self.csv_dir + ':txt file does not exist!')
        file = open(self.csv_dir)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.train_flag == True:
            img_path = self.names_list[idx].split(',')[0]
            label_path = self.names_list[idx].split(',')[1].strip('\n')
            return self.combine_train_samples(img_path, label_path, self.norm, self.DataAug, self.gpu)
        elif self.train_flag == False:
            print("Please use 'combine_test_samples' to get test samples")
            return False
           
    def open_img(self, img_path, datatype='float32'):
        '''
            datatype of img = 'float32' 
            datatype of lab = 'uint8' 
        '''
        img_raw = gdal.Open(img_path)
        self.img_w = img_raw.RasterXSize
        self.img_h = img_raw.RasterYSize
        img = np.array(img_raw.ReadAsArray(0,0,self.img_w,self.img_h,buf_xsize=self.img_w,buf_ysize=self.img_h)).astype(datatype)
        return img

    def process_img(self, img_path, norm=True, DataAug_Trans=None, gpu=True, train=True):
        '''
            将图像数据转换为可以训练或预测的数据
        '''
        img = self.open_img(img_path, 'float32')
        img = torch.from_numpy(img)
        raw_image = img

        #数据归一化
        # if norm == True:
        #     img = (img-torch.min(img))/(torch.max(img)-torch.min(img))   #还需要abs，否则负数就相反了
        norm_list = [0.5] * img.shape[0]
        img = transforms.Normalize(mean=norm_list,std=norm_list)(img)
        
        if gpu == True:
            img = img.cuda()
            raw_image = raw_image.cuda()
        
        if train == False:
            img = img.unsqueeze(0)
            raw_image = raw_image.unsqueeze(0)

        return img, raw_image

    def process_lab(self, lab_path, DataAug_Trans=None, gpu=True):
        '''
            将标签数据转换为可以训练的训练标签
        '''
        lab = self.open_img(lab_path, 'uint8')
        lab = torch.from_numpy(lab)
        lab = lab.contiguous().view(lab.size()[0],lab.size()[1])
        
        if gpu == True:
            lab = lab.cuda()
            
        return lab

    def combine_train_samples(self, img_path, lab_path, norm=True, DataAug=False, gpu=True):
        sample = {'raw_image':[], 'img': [], 'label': []}

        img, raw_img = self.process_img(img_path, norm=norm, gpu=gpu, train=True)
        lab = self.process_lab(lab_path, gpu=gpu)

        sample['raw_image'] = raw_img
        sample['img'] = img
        sample['label']=lab
        return sample

    def get_test_samples(self, img_path, norm=True, gpu=True):
        sample = {'raw_image':[], 'img': []}

        img, raw_img = self.process_img(img_path, norm=norm, gpu=gpu, train=False)

        sample['raw_image'] = raw_img
        sample['img'] = img
        return sample


class SRNet_dataset(Dataset.Dataset):
    def __init__(self,csv_dir):
        self.csv_dir = csv_dir

        self.names_list = []
        self.size = 0
        self.transform = transforms.ToTensor()
        #把csv文件中的路径读进来
        if not os.path.isfile(self.csv_dir):
            print(self.csv_dir + ':text file does not exist!')
        file = open(self.csv_dir)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self,idx):
        #读取图像路径并打开图像

        image_path = self.names_list[idx].split(',')[0]
        img_raw = gdal.Open(image_path)
        self.img_w = img_raw.RasterXSize
        self.img_h = img_raw.RasterYSize
        img = np.array(img_raw.ReadAsArray(0,0,self.img_w,self.img_h,buf_xsize=self.img_w,buf_ysize=self.img_h)).astype('float32')
    
        #读取标签路径并打开标签图像
        label_path = self.names_list[idx].split(',')[1].strip('\n')
        label = Image.open(label_path)
        # pic_path = self.names_list[idx].split(',')[2].strip('\\n')
        # pic = Image.open(pic_path)
        #函数返回一个字典类型的数据，里面包括了图像和标签，并将它们转为tensor形式
        sample = {'image':img,'label':label}
        sample['image'] = self.transform(sample['image']).cuda()
        sample['label'] = torch.from_numpy(np.array(sample['label'])).cuda()
        # sample['pic'] = self.transform(sample['pic'])
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
