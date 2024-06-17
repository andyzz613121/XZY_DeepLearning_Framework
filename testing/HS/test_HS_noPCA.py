
from cgi import test
from pickle import FALSE
import torch
from osgeo import gdal
import numpy as np
import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from PIL import Image

from dataset.HS_dataset_new import HS_dataset
from visual.Gray2RGB import GRAYcvtRGB
class HS_test():
    def __init__(self, dataset, model, norm, PCA):
        self.PCA = PCA
        self.norm = norm

        if dataset == 'Pavia':
            if self.PCA==False:
                self.image_path='E:\\dataset\\高光谱数据集\\Pavia\\Train\\big_image.tif'
            else:
                print('Warning: Using PCA images for test')
                self.image_path='E:\\dataset\\高光谱数据集\\Pavia\\Train\\PCA_image.tif'
            self.label_path='E:\\dataset\\高光谱数据集\\Pavia\\Train\\valid_label_pt.tif'
            self.class_num = 9

        elif dataset == 'Houston13':
            if self.PCA==False:
                self.image_path='E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\big_image.tif'
            else:
                print('Warning: Using PCA images for test')
                self.image_path='E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\PCA_image.tif'
            self.label_path='E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\valid_label.tif'
            self.class_num = 15

        elif dataset == 'Houston18':
            if self.PCA==False:
                self.image_path='E:\\dataset\\高光谱数据集\\2018IEEE_Contest\\Train\\HoustonU2018.tif'
            else:
                print('Warning: Using PCA images for test')
                self.image_path='E:\\dataset\\高光谱数据集\\2018IEEE_Contest\\Train\\PCA_image.tif'
            self.label_path='E:\\dataset\\高光谱数据集\\2018IEEE_Contest\\Train\\valid_label.tif'
            self.class_num = 20

        elif dataset == 'Salinas':
            if self.PCA==False:
                self.image_path='E:\\dataset\\高光谱数据集\\Salinas\\big_image.tif'
            else:
                print('Warning: Using PCA images for test')
                self.image_path='E:\\dataset\\高光谱数据集\\Salinas\\PCA_image.tif'
            self.label_path='E:\\dataset\\高光谱数据集\\Salinas\\Train\\valid_label.tif'
            self.class_num = 16

        self.model = model
    
        self.dataset = HS_dataset(dataset=dataset, pca_flag=self.PCA, norm_flag=self.norm, train_flag=False)
        self.sample = self.dataset.open_and_procress_data(self.image_path, None)
        self.img_h = self.sample['img'].shape[2]
        self.img_w = self.sample['img'].shape[3]

    def compute_position(self, x, y, half_window_size=5):
        start_x = x - half_window_size
        start_y = y - half_window_size
        end_x = x + half_window_size + 1
        end_y = y + half_window_size + 1
        
        if start_x < 0:
            start_x = 0
            end_x = 2*half_window_size + 1
        
        if start_y < 0:
            start_y = 0
            end_y = start_y + 2*half_window_size + 1

        if end_y >= self.img_w:
            end_y = self.img_w - 1
            start_y = end_y - 2*half_window_size - 1

        if end_x >= self.img_h:
            end_x = self.img_h - 1
            start_x = end_x - 2*half_window_size - 1

        assert (end_x - start_x) ==  (2*half_window_size + 1)
        assert (end_y - start_y) ==  (2*half_window_size + 1)
        
        return start_x, end_x, start_y, end_y

    def test_model(self):
        img = self.sample['img']
        pre_map = np.zeros([self.img_h, self.img_w])
        for x in range(self.img_h):
            y = 0
            while y < self.img_w - 1:
                #计算能用多大的batch进行计算
                batch = 32
                #存一个batch的图像
                img_batch = []   

                if y+batch >= self.img_w:
                    batch = self.img_w - y - 1
                    # print(batch)

                for step in range(batch):
                    start_x, end_x, start_y, end_y = self.compute_position(x, y)
                    if self.PCA ==False:
                        img_clip = img[:, :, start_x:end_x, start_y:end_y]
                    else:
                        img_clip = img[:, 0:30, start_x:end_x, start_y:end_y]
                    img_batch.append(img_clip)
                
                    y += 1
                    if y >= self.img_w:
                        break

                input_batch = torch.cat([item for item in img_batch], 0)

                out = self.model(input_batch)[0].cpu().detach().numpy()
                pre = np.argmax(out, 1)
                pre_map[x, y-batch: y] = pre

        pre_map = pre_map + 1
        return pre_map
    
    def compute_acc(self, pre_map, out_path):
        out = open(out_path, 'w')

        label = gdal.Open(self.label_path)
        img_w = label.RasterXSize
        img_h = label.RasterYSize
        label = np.array(label.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h))

        total_num = 0
        total_true_num = 0

        for c in range(1, self.class_num+1):
            lab_class_index = (label == c)
            class_total_num = lab_class_index.sum()

            pre_class_index = (pre_map == c)
            true_class_index = (pre_class_index * lab_class_index)
            true_class_num = true_class_index.sum()

            acc_class = true_class_num/class_total_num
            out.write('c%d: %f \n' %(c, acc_class))

            total_num += class_total_num
            total_true_num += true_class_num

        acc_all = total_true_num/total_num
        print('ACC: %f \n' %acc_all)
        out.write('all: %f \n' %acc_all)

    # def GRAYcvtRGB(self, dataset, img_gray):
    #     label_rgb = np.zeros((img_gray.shape[0],img_gray.shape[1],3)).astype(np.uint8)
    #     if dataset == 'Houston13':
    #         rgb_list = [[45,153,0], [81,255,0], [49, 153, 154], [27, 102, 0],
    #         [98, 50, 0], [21, 10, 208], [255, 255, 255], [255, 255, 0], 
    #         [160, 160, 160], [146, 0, 0], [147, 0, 155], [251, 204, 203], [246, 127, 0], 
    #         [245, 0, 255], [87, 255, 255]]
    #     if dataset == 'Houston18':
    #         rgb_list = [[81,255,0], [143,255,0], [80, 205, 32], [53, 139, 23], [26, 100, 0],
    #         [158, 40, 39], [29, 15, 255], [238, 238, 204], [190, 190, 190], [244, 0, 0],
    #         [238, 238, 223], [132, 139, 131], [134, 68, 9], [133, 0, 0], [250, 200, 0], 
    #         [255, 255, 0], [200, 148, 0], [250, 192, 203], [9, 3, 130], [108, 148, 240]]
    #     if dataset == 'Pavia':
    #         rgb_list = [[255,0,0], [81,255,0], [29, 15, 255], [254, 254, 0],
    #         [87, 255, 255], [244, 0, 255], [192, 192, 192], [128, 128, 128], [122, 0, 0]]

    #     for c in range(len(rgb_list)):
    #         index = ((img_gray-1) == c)
    #         if (index.sum() > 0):
    #             label_rgb[index, 0] = rgb_list[c][0]
    #             label_rgb[index, 1] = rgb_list[c][1]
    #             label_rgb[index, 2] = rgb_list[c][2]

    #     return label_rgb

def testHS(dataset, img_model, path, norm, PCA):
    Test = HS_test(dataset, img_model, norm, PCA)

    pre_path = path + '.tif'
    prergb_path = path + '_RGB.tif'
    acc_path = path + '.txt'

    pre_map = Test.test_model()
    pre_mapout = Image.fromarray(pre_map)
    pre_mapout.save(pre_path)

    pre_rgb = GRAYcvtRGB(dataset, pre_map)
    pre_rgbout = Image.fromarray(pre_rgb)
    pre_rgbout.save(prergb_path)

    Test.compute_acc(pre_map, acc_path)

if __name__ == "__main__":
    image_model_name = 'result\\HyperSpectral\\Pavia\\Normalized\\SpecAttenNet\\image_model400.pkl'
    img_model = torch.load(image_model_name).cuda().eval()
    testHS('Pavia', img_model, path = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\200', PCA=FALSE)
        
        

