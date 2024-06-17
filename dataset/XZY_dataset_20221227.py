import os
import numpy as np
from osgeo import gdal

import torch
import torch.utils.data.dataset as Dataset
import torchvision.transforms as transforms

class basic_dataset_operation():
    def open_img(self, img_path, datatype='float32'):
        '''
            datatype of img = 'float32' 
            datatype of lab = 'uint8' 
        '''
        img_raw = gdal.Open(img_path)
        img_w = img_raw.RasterXSize
        img_h = img_raw.RasterYSize
        img = np.array(img_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(datatype)

        return img

    def process_img(self, img_path, norm=True, DataAug_Trans=None, gpu=True, train=True):
        '''
            将图像数据转换为可以训练或预测的数据
        '''
        img = self.open_img(img_path, 'float32')
        if len(img.shape) == 2:
            img = img[np.newaxis, :]

        img = torch.from_numpy(img)
        raw_image = img

        #数据归一化
        if norm == True:
            img = (img-torch.min(img))/(torch.max(img)-torch.min(img))   #还需要abs，否则负数就相反了

            assert len(img.shape) == 3, 'Unknow Image shape: Image shape used by (C, H, W)'
            
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

class XZY_train_dataset(Dataset.Dataset):
    def __init__(self, csv_dir=None, gpu=True, DataAug=False, norm=True):
        
        print("---XZY_train_dataset---")
        self.csv_dir = csv_dir          
        self.names_list = []
        self.size = 0
        self.gpu = gpu
        self.DataAug = DataAug
        self.norm = norm
        self.dst_opr = basic_dataset_operation()

        if not os.path.isfile(self.csv_dir):
            print(self.csv_dir + ':txt file does not exist!')
        file = open(self.csv_dir)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_list = self.names_list[idx].split(',')

        imgpath_list = input_list[0:-1]
        label_path = input_list[-1].strip('\n')
        return self.combine_train_samples(imgpath_list, label_path, self.norm, self.DataAug, self.gpu)

    def combine_train_samples(self, imgpath_list, lab_path, norm=True, DataAug=False, gpu=True):
        '''
            Input: 
                    imgpath_list(输入图像的路径列表，可包含多种输入图像，例如RGB，DEM等)，需注意，第一个为最主要的影像
                    lab_path(标签图像路径)
        '''
        sample = {}
        for num in range(len(imgpath_list)):
            img, raw_img = self.dst_opr.process_img(imgpath_list[num], norm=norm, gpu=gpu, train=True)
            if num == 0:
                sample['raw_image'] = raw_img
            img_name = 'img_' + str(num)
            sample[img_name] = img

        lab = self.dst_opr.process_lab(lab_path, gpu=gpu)
        sample['label']=lab

        return sample

class XZY_test_dataset(Dataset.Dataset):
    def __init__(self, gpu=True, norm=True):
        
        print("---XZY_test_dataset---")
        self.gpu = gpu
        self.norm = norm
        self.dst_opr = basic_dataset_operation()

    def get_test_samples(self, imgpath_list, norm=True, gpu=True):
        '''
            Input: 
                    imgpath_list(输入图像的路径列表，可包含多种输入图像，例如RGB，DEM等)，需注意，第一个为最主要的影像
        '''
        sample = {}
        for num in range(len(imgpath_list)):
            img, raw_img = self.dst_opr.process_img(imgpath_list[num], norm=norm, gpu=gpu, train=False)
            if num == 0:
                sample['raw_image'] = raw_img
            img_name = 'img_' + str(num)
            sample[img_name] = img

        return sample

if __name__ == '__main__':
    aa = 1