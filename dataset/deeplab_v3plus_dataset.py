import torch
from torch import nn
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image 
import random
import os
import cv2
import numpy as np
from osgeo import gdal
from data_processing.layer_data_augmentator import DataAugmentation
class classification_dataset(Dataset.Dataset):
    def __init__(self, csv_dir, gpu=True):
        self.csv_dir = csv_dir          
        self.names_list = []
        self.size = 0
        self.gpu = gpu
        self.img_num = 0
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


        #数据增广
        Data_Aug = DataAugmentation()
        Trans = Data_Aug.get_random_transform_params(img)
        img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
        img = cv2.warpPerspective(img, Trans, dsize=(img.shape[0], img.shape[1]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        label = cv2.warpPerspective(label, Trans, dsize=(img.shape[1], img.shape[2]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        
        raw_image = img
        
        #数据归一化
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        img = torch.from_numpy(img)
        raw_image = img

        label = torch.from_numpy(label)
        label = label.contiguous().view(label.size()[0],label.size()[1])
        #数据标准化Normalize
        if img.shape[0]==3:
            img = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(img)
        elif img.shape[0]==4:
            img = transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5, 0.5])(img)

        if self.gpu == True:
            img = img.cuda()
            label = label.cuda()

        sample['raw_image']=raw_image
        sample['img']=img
        sample['label']=label
        sample['name']= self.names_list[idx].split(',')[0]
        return sample


class Dataset_myself():
    def __init__(self, csv_dir, batch, shuffle=True, gpu=True):
        self.csv_dir = csv_dir   
        self.batch = batch 
        self.shuffle = shuffle      
        self.gpu = gpu
        self.img_num = 0
        if not os.path.isfile(self.csv_dir):
            print(self.csv_dir + ':txt file does not exist!')

        self.names_list = list(open(self.csv_dir))
        if shuffle:
            random.shuffle(self.names_list)
            print('random shuffle dataset')
        self.size = len(self.names_list)
        self.cur_item = 0
        print("Dataset is : %s, Train set size is : %d, Batch is : %d, Shuffle is : %s, GPU is : %s" % (self.csv_dir, self.size, self.batch, self.shuffle, self.gpu))
    def __len__(self):
        return len(self.names_list)

    def __iter__(self):
        return self

    def __next__(self):
        return_list = []
        if self.cur_item + self.batch > self.size:
            if self.shuffle:
                random.shuffle(self.names_list)
                self.cur_item = 0
                print('an epoch is done, random shuffle dataset again')
            else:
                self.cur_item = 0
                print('an epoch is done')
            raise StopIteration
                
        else:
            for item in range(self.cur_item, self.cur_item+self.batch):
                # print(self.cur_item, self.cur_item+self.batch, item)
                return_list.append(self.names_list[item])
            self.cur_item += self.batch
            sample = self.data_processing(return_list)
            return sample

    def data_processing(self, name_list):
        sample = {'raw_image':[], 'img':[], 'dsm_ndsm':[], 'label':[]}
        basename = 'D:\\Code\\LULC\\Hed_Seg\\'
        for item in range(len(name_list)):
            img_path = basename + name_list[item].split(',')[0]
            img_raw = gdal.Open(img_path)
            img_w = img_raw.RasterXSize
            img_h = img_raw.RasterYSize
            dsm_path = basename + name_list[item].split(',')[1]
            dsm_raw = gdal.Open(dsm_path)
            ndsm_path = basename + name_list[item].split(',')[2]
            ndsm_raw = gdal.Open(ndsm_path)
            label_path = basename + name_list[item].split(',')[3].strip('\n')
            label_raw = gdal.Open(label_path)

            img = np.array(img_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
            dsm = np.array(dsm_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
            ndsm = np.array(ndsm_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
            label = np.array(label_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h))

            #数据增广
            Data_Aug = DataAugmentation()
            Trans = Data_Aug.get_random_transform_params(img)
            img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
            img = cv2.warpPerspective(img, Trans, dsize=(img.shape[0], img.shape[1]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
            dsm = cv2.warpPerspective(dsm, Trans, dsize=(img.shape[1], img.shape[2]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            ndsm = cv2.warpPerspective(ndsm, Trans, dsize=(img.shape[1], img.shape[2]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpPerspective(label, Trans, dsize=(img.shape[1], img.shape[2]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

            #数据归一化
            img = (img-np.min(img))/(np.max(img)-np.min(img))
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

            dsm_ndsm = np.array([dsm, ndsm])

            img = torch.from_numpy(img)
            raw_image = img
            dsm_ndsm = torch.from_numpy(dsm_ndsm)
            label = torch.from_numpy(label)
            label = label.contiguous().view(label.size()[0],label.size()[1])
            #数据标准化Normalize
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5],std=[0.229, 0.224, 0.225, 0.5])(img)
            dsm_ndsm = transforms.Normalize(mean=[0.5,0.5],std=[0.5,0.5])(dsm_ndsm)

            if self.gpu == True:
                img = img.cuda()
                dsm_ndsm = dsm_ndsm.cuda()
                label = label.cuda()

        
            raw_image = raw_image.unsqueeze(0)
            img = img.unsqueeze(0)
            dsm_ndsm = dsm_ndsm.unsqueeze(0)
            label = label.unsqueeze(0)
            
            if item == 0:
                sample['raw_image'] = raw_image
                sample['img'] = img
                sample['dsm_ndsm'] = dsm_ndsm
                sample['label'] = label
            else:
                sample['raw_image'] = torch.cat([sample['raw_image'], raw_image], 0)
                sample['img'] = torch.cat([sample['img'], img], 0)
                sample['dsm_ndsm'] = torch.cat([sample['dsm_ndsm'], dsm_ndsm], 0)
                sample['label'] = torch.cat([sample['label'], label], 0)

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
    dataset = Dataset_myself('train_Deeplabv3plus.csv', 4)
    for i, data in enumerate(dataset):
        raw_images, images, dsm_ndsms, labels = data['raw_image'], data['img'], data['dsm_ndsm'], data['label']
        print(raw_images.shape, images.shape, dsm_ndsms.shape, labels.shape)
    # a = next(dataset)
    # print(a)
    # print(len(a))
if __name__ == '__main__':
    main()
