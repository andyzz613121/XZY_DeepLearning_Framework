import torch
import cv2
import os
import numpy as np
import torch.nn.functional as F
import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from PIL import Image
from osgeo import gdal
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset.XZY_dataset import ISPRS_dataset
from model.SegNet.SegNet_dis import SegNet

class ISPRS_test_dataset():
    def __init__(self, dataset_name, image_path, dsm_path, ndsm_path, label_path):
        self.image_path = image_path
        self.dsm_path = dsm_path
        self.ndsm_path = ndsm_path
        self.label_path = label_path
        self.dataset_name = dataset_name
        assert self.dataset_name == 'Vaihingen' or self.dataset_name == 'Potsdam', "ERROR:: test_image//dataset"

        self.dataset = ISPRS_dataset(None, dataset_name, train_flag=False)
        self.sample = self.open_test_data(self.image_path, self.dsm_path, self.ndsm_path, self.label_path)
        self.img_h = self.sample['img'].shape[2]
        self.img_w = self.sample['img'].shape[3]
    def open_test_data(self, image_path, dsm_path, ndsm_path, label_path):
        return self.dataset.open_and_procress_data(self.dataset_name, image_path, dsm_path, ndsm_path, label_path, label_path)

class test_model():
    def __init__(self, dataset_name, class_num=6):
        self.dataset_name = dataset_name
        self.class_num = 6

    def test_image(self, model_img, model_addition, patch_size=256, run_time=3000):
        if self.dataset_name == 'Vaihingen':
            self.test_Vaihingen(model_img, model_addition, patch_size, run_time)
        elif self.dataset_name == 'Potsdam':
            self.test_Potsdam(model_img, model_addition, patch_size, run_time)
        else:
            print("Unknow dataset_name")

    def test_Vaihingen(self, model_img, model_addition, patch_size, run_time):
        print("---testing model in Vaihingen---")
        self.total_pixel = 0
        self.pos_pixel = 0
        for img_NO in (30,5,7,23):
        
            image_path = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai\\test\\big_image\\image' + str(img_NO) + '.tif'
            dsm_path = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai\\test\\big_dsm\\big_dsm' + str(img_NO) + '.tif'
            ndsm_path = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai\\test\\big_ndsm\\big_ndsm' + str(img_NO) + '.tif'
            label_path = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai\\label_gray\\label' + str(img_NO) + '_gray.tif'
            predict_path = 'result\\pre' + str(img_NO) + '_'

            test_dataset = ISPRS_test_dataset('Vaihingen', image_path,  dsm_path, ndsm_path, label_path)
            self.image_probability = np.zeros((test_dataset.img_h, test_dataset.img_w)).astype(np.float32)
            self.predect_count_list = []
            for classes in range(self.class_num):
                self.predect_count_list.append(np.zeros((test_dataset.img_h, test_dataset.img_w))) #记录每个类的个数
            #########################################################################
            total_num, pos_num = self.test_patches(test_dataset, img_NO, model_img, model_addition, predict_path, patch_size, run_time)
            # total_num, pos_num = self.test_total_image(test_dataset, img_NO, model_img, model_addition, predict_path)
            self.total_pixel += total_num
            self.pos_pixel += pos_num
        acc = self.pos_pixel/self.total_pixel
        print('AC is %f'%acc)

    def test_Potsdam(self, model_img, model_addition, patch_size, run_time):
        print("---testing model in Potsdam---")
        self.total_pixel = 0
        self.pos_pixel = 0
        image_folder = 'D:\\dataset\\Postdam\\4_Ortho_RGBIR\\'
        dsm_folder = 'D:\\dataset\\Postdam\\1_DSM\\'
        ndsm_folder = 'D:\\dataset\\Postdam\\1_DSM_normalisation\\'
        label_folder = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_pos\\label_gray\\'
        for img_NO in ['2_11', '5_11', '4_10', '7_08']:
        # for img_NO in ['2_11']:
            image_path = image_folder + 'top_potsdam_' + img_NO + '_RGBIR.tif'
            dsm_path = dsm_folder + 'dsm_potsdam_0' + img_NO + '.tif'
            ndsm_path = ndsm_folder + 'dsm_potsdam_0' + img_NO + '_normalized_lastools.jpg'
            label_path = label_folder + 'label' + img_NO + '_gray.tif'
            predict_path = 'result\\pre' + str(img_NO) + '_'

            test_dataset = ISPRS_test_dataset('Potsdam', image_path,  dsm_path, ndsm_path, label_path)
            self.image_probability = np.zeros((test_dataset.img_h, test_dataset.img_w)).astype(np.float32)
            self.predect_count_list = []
            for classes in range(self.class_num):
                self.predect_count_list.append(np.zeros((test_dataset.img_h, test_dataset.img_w))) #记录每个类的个数
            #########################################################################
            total_num, pos_num = self.test_patches(test_dataset, img_NO, model_img, model_addition, predict_path, patch_size, run_time)
            self.total_pixel += total_num
            self.pos_pixel += pos_num
        acc = self.pos_pixel/self.total_pixel
        print('AC is %f'%acc)
    
    def test_patches(self, test_dataset, img_NO, model_img, model_addition, predict_path, patch_size, run_time):
        model_dis = SegNet(3, 1).cuda()
        # model_dis = torch.load('D:\\Code\\LULC\\Laplace\\result\\Dis_Pos\\image_model50.pkl').cuda()
        # model_dis.load_state_dict(torch.load('pretrained\\Dis\\Dis_Pos.pth'))
        model_dis.load_state_dict(torch.load('pretrained\\Dis\\Dis_Vai.pth'))
        model_dis.eval()
        img_model = model_img
        addition_model = model_addition
        img_model.eval()
        addition_model.eval()

        
        img_w = test_dataset.img_w
        img_h = test_dataset.img_h

        cur_h = 0
        while cur_h <= img_h: 
            start_h = cur_h 
            end_h = cur_h + patch_size
            if end_h >= img_h:
                end_h = img_h - 1
                start_h = end_h - patch_size
            cur_w = 0
            while cur_w <= img_w:
                start_w = cur_w
                end_w = cur_w + patch_size
                if end_w >= img_w:
                    end_w = img_w - 1
                    start_w = end_w - patch_size
                self.test_single_patch(test_dataset, start_h, end_h, start_w, end_w, img_model, addition_model, model_dis)

                cur_w+=int(patch_size/2)
            cur_h+=int(patch_size/2)

        for i in range(1000):
            start_h = np.random.randint(0, img_h-1)
            end_h = start_h + patch_size
            if end_h >= img_h:
                end_h = img_h - 1
                start_h = end_h - patch_size
            start_w = np.random.randint(0, img_w-1)
            end_w = start_w + patch_size
            if end_w >= img_w:
                end_w = img_w - 1
                start_w = end_w - patch_size
            self.test_single_patch(test_dataset, start_h, end_h, start_w, end_w, img_model, addition_model, model_dis)

        image_predect = np.array([self.predect_count_list[0], self.predect_count_list[1], self.predect_count_list[2], 
            self.predect_count_list[3], self.predect_count_list[4], self.predect_count_list[5]])
        image_predect1 = np.array([self.predect_count_list[i] for i in range(self.class_num)])    
        image_predect = np.argmax(image_predect, 0).astype(np.uint8)
        
        label = test_dataset.sample['label'].cpu().detach().numpy()
        pos_index = (label==image_predect)
        total_num = img_h * img_w
        self.total_pixel += total_num
        pos_num = pos_index.sum()
        self.pos_pixel += pos_num
        true_rate = pos_num/total_num
        print('image%s , acc is %f'%(str(img_NO), true_rate))
        # self.image_probability = Image.fromarray(self.image_probability)
        image_predect = Image.fromarray(image_predect)
        predict_path = predict_path + str(true_rate) + '.tif'
        image_predect.save(predict_path)
        return total_num, pos_num
        
    def test_total_image(self, test_dataset, img_NO, model_img, model_addition, predict_path):
        img_model = model_img
        addition_model = model_addition
        img_model.eval()
        addition_model.eval()
        
        model_dis = SegNet(3, 1).cuda()
        # model_dis = torch.load('D:\\Code\\LULC\\Laplace\\result\\Dis_Pos\\image_model50.pkl').cuda()
        model_dis.load_state_dict(torch.load('pretrained\\Dis\\Dis_Vai.pth'))
        model_dis.eval()

        img_w = test_dataset.img_w
        img_h = test_dataset.img_h
        w = None
        # AW_mlp = Weight_MLP.Auto_Weights(4, [25, 50, 25, 6])

        with torch.no_grad():
            sample = test_dataset.sample
            self.img = sample['img']
            self.dsm = sample['dsm']
            self.ndvi = sample['ndvi']
            self.label = sample['label']

            img = self.img
            dsm = self.dsm
            ndvi = self.ndvi
            label = self.label

            dis = model_dis(img)
            if torch.max(dis)==torch.min(dis):
                if torch.max(dis)!=0:
                    dis = dis/torch.max(dis)
            else:
                dis = (dis-torch.min(dis))/(torch.max(dis)-torch.min(dis))
            dis = (dis - 0.5)/0.5

            adds = torch.cat([img, dsm, dis, ndvi], 1)

            output = model_img(adds)[0].cpu().detach().numpy()
            
        image_predect = np.argmax(output, 0).astype(np.uint8)
        label = label[0].cpu().detach().numpy()
        pos_index = (label==image_predect)
        total_num = img_h * img_w
        self.total_pixel += total_num
        pos_num = pos_index.sum()
        self.pos_pixel += pos_num
        true_rate = pos_num/total_num
        print('image%s , acc is %f'%(str(img_NO), true_rate))
        image_predect = Image.fromarray(image_predect)
        predict_path = predict_path + str(true_rate) + '.tif'
        image_predect.save(predict_path)
        return total_num, pos_num

    def test_single_patch(self, test_dataset, start_h, end_h, start_w, end_w, model_img, model_addition, model_dis):
        with torch.no_grad():
            sample = test_dataset.sample
            self.img = sample['img']
            self.dsm = sample['dsm']
            self.dis = sample['dis']
            self.ndvi = sample['ndvi']
            self.ndwi = sample['ndwi']
            self.label = sample['label']
            
            img_patch = self.img[:, :, start_h:end_h, start_w:end_w]
            dsm_patch = self.dsm[:, :, start_h:end_h, start_w:end_w]
            dis_patch = self.dis[:, :, start_h:end_h, start_w:end_w]
            ndvi_patch = self.ndvi[:, :, start_h:end_h, start_w:end_w]
            label_patch = self.label[:, start_h:end_h, start_w:end_w]

            dis_patch = model_dis(img_patch)

            # driver = gdal.GetDriverByName('GTiff')
            # dataset = driver.Create("C:\\Users\\ASUS\\Desktop\\2.tif", 256, 256, 1, gdal.GDT_Float32)
            # dataset.GetRasterBand(1).WriteArray(dis_patch[0][0].cpu().numpy())

            if torch.max(dis_patch)==torch.min(dis_patch):
                if torch.max(dis_patch)!=0:
                    dis_patch = dis_patch/torch.max(dis_patch)
            else:
                dis_patch = (dis_patch-torch.min(dis_patch))/(torch.max(dis_patch)-torch.min(dis_patch))
            
            dis_patch = (dis_patch - 0.5)/0.5

            inputs = torch.cat([img_patch, dsm_patch, dis_patch, ndvi_patch], 1)
            output = model_img(inputs)
            
            max_probability, max_index = torch.max(output, 1)
            max_probability = max_probability.cpu().detach().numpy()
            max_index = max_index.cpu().detach().numpy()
            max_probability = max_probability.reshape((max_probability.shape[1], max_probability.shape[2]))
            max_index = max_index.reshape((max_index.shape[1], max_index.shape[2]))

            big_pb_index = (self.image_probability[start_h:end_h, start_w:end_w] < max_probability)
            self.image_probability[start_h:end_h, start_w:end_w][big_pb_index] = max_probability[big_pb_index]

            for classes in range(6):
                class_index = (max_index == classes)
                self.predect_count_list[classes][start_h:end_h, start_w:end_w][class_index] += 1



if __name__ == "__main__":
    
    for i in range(50, 0, -1):
        mlp_path= '..\\XZY_DeepLearning_Framework\\result\image_model' +str(i)+'.pkl'
        img_path = '..\\XZY_DeepLearning_Framework\\result\\image_model' +str(i)+'.pkl'
        # img_model = PSPNet_AW(6,6).cuda()
        # img_model.load_state_dict(torch.load(img_path))
        # mlp_model = PSPNet_AW(6,6).cuda()
        # mlp_model.load_state_dict(torch.load(mlp_path))
        img_model = torch.load(img_path)
        mlp_model = torch.load(mlp_path)
        test = test_model('Vaihingen', 6)
        test.test_image(img_model, mlp_model)

