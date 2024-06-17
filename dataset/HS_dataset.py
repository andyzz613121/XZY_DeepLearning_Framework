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
    def __init__(self, csv_dir=None, gpu=True, train_flag=True):
        self.csv_dir = csv_dir          
        self.names_list = []
        self.size = 0
        self.gpu = gpu
        self.img_num = 0
        self.train_flag = train_flag
        self.sample = {'raw_image':[], 'img': [], 'label': []}

        if self.train_flag == True:
            print("---Train HS Dataset---")
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
        label = np.array(int(self.names_list[idx].split(',')[1].strip('\n')))
        return self.open_and_procress_data(img_path, label)
    
    def open_and_procress_data(self, img_path, label):
        sample = {}
        img_raw = gdal.Open(img_path)
        img_w = img_raw.RasterXSize
        img_h = img_raw.RasterYSize
        img = np.array(img_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        
        # if self.train_flag == True:
        #     #数据增广
        #     Data_Aug = DataAugmentation()
        #     Trans = Data_Aug.get_random_transform_params(img)
        #     img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
        #     img = cv2.warpPerspective(img, Trans, dsize=(img.shape[0], img.shape[1]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        #     img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        # raw_image = img

        #数据归一化
        # img = (img-np.min(img))/(np.max(img)-np.min(img))   #还需要abs，否则负数就相反了

        img = torch.from_numpy(img)/65536
        raw_image = img
        

        # #数据标准化Normalize
        # if img.shape[0] == 4:
        #     img = transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5, 0.5])(img)
        # elif img.shape[0] == 3:
        #     img = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(img)
        # else:
        #     para_list = list(np.ones(img.shape[0]) * 0.5)
        #     img = transforms.Normalize(mean=para_list,std=para_list)(img)


        if self.train_flag == False:
            img = img.unsqueeze(0)

        if self.gpu == True:
            img = img.cuda()

        sample['raw_image']=raw_image
        sample['img']=img
        if label!=None:
            sample['label'] = torch.from_numpy(label).cuda()

        return sample

def PCA(X, k=30):
    c, h, w = X.size()
    X = X.view(c, -1)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))  # U*Diag(S)*V_T
    return torch.mm(X, U[:,:k])

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
