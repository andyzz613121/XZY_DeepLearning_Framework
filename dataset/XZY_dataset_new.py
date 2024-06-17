import os
import sys
import numpy as np
from osgeo import gdal

import torch
import torch.utils.data.dataset as Dataset
import torchvision.transforms as transforms

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from data_processing.Raster import *

class basic_dataset_operation():
    def read_imglab(self, imglab_dir):
        '''
            Return list of img and label names, size of dataset
        '''
        names_list = []
        size = 0
        if not os.path.isfile(imglab_dir):
            print(imglab_dir + ':txt file does not exist!')
        imglab_file = open(imglab_dir)
        for f in imglab_file:
            names_list.append(f)
            size += 1
        return names_list, size
    
    def read_maxmin(self, maxmin_dir):
        '''
            Return MAXMIN_array: [[min_b1, max_b1], ..., [min_bn, max_bn]]
        '''
        MAXMIN_list = []
        if not os.path.isfile(maxmin_dir):
            print(maxmin_dir + ':MAX FILE does not exist!')
        MAX_file = open(maxmin_dir)
        tmp_maxmin = []
        for max_f in MAX_file:
            if 'Img' in max_f:          # 如果遇到分隔符ImgX
                if len(tmp_maxmin) > 0:
                    MAXMIN_list.append(tmp_maxmin)
                tmp_maxmin = []
                continue
            
            min, max = max_f.split(',')[0], max_f.split(',')[1].strip('\n')
            min, max = np.array(float(min)), np.array(float(max))
            tmp_maxmin.append([min, max])
        MAXMIN_list.append(tmp_maxmin)
        return np.array(MAXMIN_list)
    
    def process_img(self, img_path, norm=True, DataAug_Trans=None, gpu=True, train=True):
        '''
            将图像数据转换为可以训练或预测的数据
        '''
        img, para = gdal_read_tif(img_path)
        if len(img.shape) == 2:
            img = img[np.newaxis, :]

        img = torch.from_numpy(img.astype(np.float32))
        raw_image = img
        
        #数据归一化
        if norm == True:
            img_min, img_max = get_metadata(para[5], 'min'), get_metadata(para[5], 'max')
            if img_min != None and img_max != None:
                img_min, img_max = np.array(list(map(int,img_min.strip().split()))), np.array(list(map(int,img_max.strip().split())))
                img_min = np.reshape(img_min, [img_min.shape[0], 1, 1])
                img_max = np.reshape(img_max, [img_max.shape[0], 1, 1])
            else:
                img_min = torch.tensor([torch.min(img[i]) for i in range(img.shape[0])]).view(-1, 1, 1)
                img_max = torch.tensor([torch.max(img[i]) for i in range(img.shape[0])]).view(-1, 1, 1)

            img = (img-img_min)/(img_max-img_min)   #还需要abs，否则负数就相反了
            norm_list = [0.5] * img.shape[0]
            img = transforms.Normalize(mean=norm_list,std=norm_list)(img).float()
        
        if gpu == True:
            img = img.cuda()
            raw_image = raw_image.cuda()
        
        if train == False:
            img = img.unsqueeze(0)
            raw_image = raw_image.unsqueeze(0)
        return img, raw_image

    def process_lab(self, lab_path, type, DataAug_Trans=None, gpu=True):
        '''
            将标签数据转换为可以训练的训练标签
        '''
        if type == 'img':
            lab, _ = gdal_read_tif(lab_path)
            lab = torch.from_numpy(lab.astype(np.uint8))
            if len(lab.size()) == 2:
                lab = lab.contiguous().view(lab.size()[0],lab.size()[1])
            else:
                lab = lab.contiguous().view(lab.size()[0],lab.size()[1],lab.size()[2])
            if gpu == True:
                lab = lab.cuda()
                
            return lab
        elif type == 'value':
            lab = np.array(int(lab_path))
            return torch.from_numpy(lab).cuda()


class XZY_train_dataset(Dataset.Dataset):
    def __init__(self, csv_file, norm_list=[True, False], type=['img', 'img'],  DataAug=False, gpu=True):
        print("---XZY_train_dataset---")
        self.gpu = gpu
        self.DataAug = DataAug
        self.names_list = []
        self.size = 0
        self.dst_opr = basic_dataset_operation()
        # 如果有多个输入数据，哪些输入图像要归一化
        self.norm_list = norm_list        
        # 表示输入的是什么类型的数据：img图像，value值
        self.type = type
        if not os.path.isfile(csv_file):
            print(csv_file + ':txt file does not exist!')
        file = open(csv_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        imgpath_list = self.names_list[idx].strip().split(',')
        return self.combine_train_samples(imgpath_list, self.norm_list, self.type, self.DataAug, self.gpu)

    def combine_train_samples(self, imgpath_list, norm_list, type, DataAug=False, gpu=True):
        '''
            Input: 
                    imgpath_list(输入图像的路径列表，可包含多种输入图像，例如RGB，DEM等)，需注意，第一个为最主要的影像
                    lab_path(标签图像路径)
        '''
        sample = {}
        assert len(norm_list) == len(imgpath_list), 'len(norm_list) != len(imgpath_list)'
        num_img, num_lab = 0, 0
        for num in range(len(imgpath_list)):
            if norm_list[num] == True:      # 必定是图像数据
                img, raw_img = self.dst_opr.process_img(imgpath_list[num], norm=norm_list[num], gpu=gpu, train=True)
                raw_name = 'rawimg_' + str(num_img)
                sample[raw_name] = raw_img
                img_name = 'img_' + str(num_img)
                sample[img_name] = img
                num_img += 1
            else:
                lab = self.dst_opr.process_lab(imgpath_list[num], type[num], gpu=gpu)
                img_name = 'lab_' + str(num_lab)
                sample[img_name]=lab
                num_lab += 1

        return sample

class XZY_test_dataset(Dataset.Dataset):
    def __init__(self, gpu=True, norm_list=[True]):
        
        print("---XZY_test_dataset---")
        self.gpu = gpu
        self.norm_list = norm_list
        self.dst_opr = basic_dataset_operation()

    def get_test_samples(self, imgpath_list, norm_list, gpu=True):
        '''
            Input: 
                    imgpath_list(输入图像的路径列表，可包含多种输入图像，例如RGB，DEM等)，需注意，第一个为最主要的影像
        '''
        sample = {}
        for num in range(len(imgpath_list)):
            img, raw_img = self.dst_opr.process_img(imgpath_list[num], norm=norm_list[num], gpu=gpu, train=False)
            raw_name = 'rawimg_' + str(num)
            sample[raw_name] = raw_img
            img_name = 'img_' + str(num)
            sample[img_name] = img

        return sample

if __name__ == '__main__':
    aa = 1