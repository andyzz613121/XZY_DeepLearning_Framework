import torch
import cv2
import os
import numpy as np
import torch.nn.functional as F
import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from PIL import Image

from dataset.XZY_dataset import RS_dataset
from data_processing.Raster import gdal_write_tif
class IMG_Testdata():
    def __init__(self, image_path, HugeData_Flag=False):
        self.image_path = image_path

        self.dataset = RS_dataset(None, train_flag=False, gpu=(not HugeData_Flag))
        self.sample = self.open_test_data(self.image_path)
        self.img_h = self.dataset.img_h
        self.img_w = self.dataset.img_w

    def open_test_data(self, image_path):
        return self.dataset.open_and_procress_data(image_path, None)


class test_IMG():
    def __init__(self, model_img, class_num, TestImg_file, Pre_Folder = 'result\\'):
        print("---testing IMG---")
        
        predict_path = Pre_Folder + 'pre.png'
        test_data = IMG_Testdata(TestImg_file, True)
        self.class_num = class_num
        self.image_probability = np.zeros((test_data.img_h, test_data.img_w)).astype(np.float32)
        self.predect_count_list = []
        for classes in range(self.class_num):
            self.predect_count_list.append(np.zeros((test_data.img_h, test_data.img_w)).astype(np.uint8)) #记录每个类的个数
        self.test_patches(test_data, model_img, 256, 0)
        test_data = None
        self.save_pre(predict_path)

    def save_pre(self, predict_path):
        image_predect = np.array([self.predect_count_list[i] for i in range(self.class_num)])    
        image_predect = np.argmax(image_predect, 0).astype(np.uint8)
        img_rgb = self.cvtRGB(image_predect)
        image_predect = Image.fromarray(image_predect)
        image_predect.save(predict_path)
        # gdal_write_tif(predict_path, self.image_probability, self.image_probability.shape[1], self.image_probability.shape[0], 1, datatype=2)
        return True

    def test_patches(self, test_data, model_img, patch_size, run_time):
        img_model = model_img
        img_model.eval()

        img_w = test_data.img_w
        img_h = test_data.img_h

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
                self.test_single_patch(test_data, start_h, end_h, start_w, end_w, img_model)

                cur_w+=int(patch_size/2)
            cur_h+=int(patch_size/2)

        for i in range(run_time):
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
            self.test_single_patch(test_data, start_h, end_h, start_w, end_w, img_model)
        
        return True

    def test_single_patch(self, test_data, start_h, end_h, start_w, end_w, model_img):
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = test_data.sample
            self.img = sample['img']
            
            
            img_patch = self.img[:, :, start_h:end_h, start_w:end_w].cuda()
            if img_patch.is_cuda == False:
                img_patch = img_patch.cuda()
            
            img_outputs = model_img(img_patch)[0].unsqueeze(0)

            self.image_probability[start_h:end_h, start_w:end_w] = img_outputs[0][1].cpu().detach().numpy()
            max_probability, max_index = torch.max(img_outputs, 1)
            max_probability = max_probability.cpu().detach().numpy()
            max_index = max_index.cpu().detach().numpy()
            max_probability = max_probability.reshape((max_probability.shape[1], max_probability.shape[2]))
            max_index = max_index.reshape((max_index.shape[1], max_index.shape[2]))
            # big_pb_index = (self.image_probability[start_h:end_h, start_w:end_w] < max_probability)
            # self.image_probability[start_h:end_h, start_w:end_w][big_pb_index] = max_probability[big_pb_index]

            for classes in range(self.class_num):
                class_index = (max_index == classes)
                self.predect_count_list[classes][start_h:end_h, start_w:end_w][class_index] += 1

    def cvtRGB(self, img_gray):
        label_rgb = np.zeros((img_gray.shape[0],img_gray.shape[1],3)).astype(np.uint8)

        index0 = (img_gray == 0)
        index1 = (img_gray == 1)
        index2 = (img_gray == 2)
        index3 = (img_gray == 3)
        index4 = (img_gray == 4)
        index5 = (img_gray == 5)
        index6 = (img_gray == 6)

        if (index0.sum() > 0):
            label_rgb[index0, 0] = 0
            label_rgb[index0, 1] = 255
            label_rgb[index0, 2] = 255
        if (index1.sum() > 0):
            label_rgb[index1,0] = 255
            label_rgb[index1,1] = 255
            label_rgb[index1,2] = 0
        if (index2.sum() > 0):
            label_rgb[index2,0] = 255
            label_rgb[index2,1] = 0
            label_rgb[index2,2] = 255
        if (index3.sum() > 0):
            label_rgb[index3,0] = 0
            label_rgb[index3,1] = 255
            label_rgb[index3,2] = 0
        if (index4.sum() > 0):
            label_rgb[index4,0] = 0
            label_rgb[index4,1] = 0
            label_rgb[index4,2] = 255
        if (index5.sum() > 0):
            label_rgb[index5,0] = 255
            label_rgb[index5,1] = 255
            label_rgb[index5,2] = 255
        if (index6.sum() > 0):
            label_rgb[index6,0] = 0
            label_rgb[index6,1] = 0
            label_rgb[index6,2] = 0

        return label_rgb

if __name__ == "__main__":
    
    for i in range(50, 0, -1):
        img_path = '..\\XZY_DeepLearning_Framework\\result\\image_model' +str(i)+'.pkl'
        img_model = torch.load(img_path)
        test = test_IMG(img_model, 4, 'E:\\dataset\\连云港GF2数据\\1_RPC+全色融合\\GF2_PMS1_E119.1_N34.2_20210730_L1A0005787958-pansharp1\\GF2_PMS1_E119.1_N34.2_20210730_L1A0005787958-pansharp.tif')
        break
