import configparser
import torch
from osgeo import gdal
import numpy as np
import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from PIL import Image

from dataset.HS_dataset_new import HS_dataset

class HS_test():
    def __init__(self, dataset, model, norm, PCA):
        self.PCA = PCA
        self.norm = norm

        # Read Configs
        HS_config = configparser.ConfigParser()
        HS_config.read('dataset\\Configs\\HS_Config.ini',encoding='UTF-8')
        HS_key_list = HS_config.sections()
        HS_value_list = []
        for item in HS_key_list:
            HS_value_list.append(HS_config.items(item))
        HS_config_dict = dict(zip(HS_key_list, HS_value_list))

        if dataset == 'Pavia':
            self.image_path = HS_config_dict['Pavia'][5][1]
            if self.PCA==False:
                self.pca_path=None
            else:
                print('Warning: Using PCA images for test')
                self.pca_path = HS_config_dict['Pavia'][4][1]
            self.label_path = HS_config_dict['Pavia'][6][1]
            self.class_num = 9

        elif dataset == 'Houston13':
            self.image_path = HS_config_dict['Houston13'][5][1]
            if self.PCA==False:
                self.pca_path=None
            else:
                print('Warning: Using PCA images for test')
                self.pca_path = HS_config_dict['Houston13'][4][1]
            self.label_path = HS_config_dict['Houston13'][6][1]
            self.class_num = 15

        elif dataset == 'Houston18':
            self.image_path = HS_config_dict['Houston18'][5][1]
            if self.PCA==False:
                self.pca_path=None
            else:
                print('Warning: Using PCA images for test')
                self.pca_path = HS_config_dict['Houston18'][4][1]
            self.label_path = HS_config_dict['Houston18'][6][1]
            self.class_num = 20

        elif dataset == 'Salinas':
            self.image_path = HS_config_dict['Salinas'][5][1]
            if self.PCA==False:
                self.pca_path=None
            else:
                print('Warning: Using PCA images for test')
                self.pca_path = HS_config_dict['Salinas'][4][1]
            self.label_path = HS_config_dict['Salinas'][6][1]
            self.class_num = 16

        self.model = model
    
        self.dataset = HS_dataset(dataset=dataset, pca_flag=self.PCA, norm_flag=self.norm, train_flag=False)
        self.sample = self.dataset.open_and_procress_data(self.image_path, None)
        if self.PCA==True:
            self.sample_pca = self.dataset.open_and_procress_data(self.pca_path, None)
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

    def test_model_img(self):
        img = self.sample['img']
        if self.PCA==True:
            img = self.sample_pca['img']

        pre_map = np.zeros([self.img_h, self.img_w])
        cur_h = 0
        patch_size = 11
        while cur_h <= self.img_h: 
            start_h = cur_h 
            end_h = cur_h + patch_size
            if end_h >= self.img_h:
                end_h = self.img_h - 1
                start_h = end_h - patch_size
            cur_w = 0
            while cur_w <= self.img_w:
                start_w = cur_w
                end_w = cur_w + patch_size
                if end_w >= self.img_w:
                    end_w = self.img_w - 1
                    start_w = end_w - patch_size
                
                with torch.no_grad():
                    out = self.model(img)[0].cpu().detach().numpy()
                    pre = np.argmax(out, 1)
                    pre_map[start_h:end_h, start_w:end_w] = pre

        pre_map = pre_map + 1
        return pre_map

    def test_model_pixel(self, dataset):
        img = self.sample['img']
        if self.PCA==True:
            pca_img = self.sample_pca['img']

        pre_map = np.zeros([self.img_h, self.img_w])
        self.model.eval()
        # if dataset == 'Salinas': 
        #     model_AutoFE = torch.load('result\\Salinas\\0\\AutoFeatExct_Salinas_500.pkl').cuda()
        # elif dataset == 'Houston13':
        #     model_AutoFE = torch.load('result\\Houston13\\0\\AutoFeatExct_Houston13_500.pkl').cuda()
        # elif dataset == 'Houston18':
        #     model_AutoFE = torch.load('result\\Houston18\\0\\AutoFeatExct_Houston18_500.pkl').cuda()
        # elif dataset == 'Pavia':
        #     model_AutoFE = torch.load('result\\Pavia\\0\\AutoFE_model500.pkl').cuda()
        # model_AutoFE.eval()

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
                    
                    if self.PCA==False:
                        img_clip = img[:, :, start_x:end_x, start_y:end_y]
                    elif self.PCA==True:
                        img_clip = pca_img[:, :, start_x:end_x, start_y:end_y]
                    img_batch.append(img_clip)
                
                    y += 1
                    if y >= self.img_w:
                        break

                input_batch = torch.cat([item for item in img_batch], 0)
                with torch.no_grad():
                    # _, Feats = model_AutoFE(input_batch)
                    out = self.model(input_batch)[0][0].cpu().detach().numpy()
                    # out = self.model(Feats)[0][0].cpu().detach().numpy()

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

    def GRAYcvtRGB(self, dataset, img_gray):
        label_rgb = np.zeros((img_gray.shape[0],img_gray.shape[1],3)).astype(np.uint8)
        if dataset == 'Houston13':
            rgb_list = [[45,153,0], [81,255,0], [49, 153, 154], [27, 102, 0],
            [98, 50, 0], [21, 10, 208], [255, 255, 255], [255, 255, 0], 
            [160, 160, 160], [146, 0, 0], [147, 0, 155], [251, 204, 203], [246, 127, 0], 
            [245, 0, 255], [87, 255, 255]]
        if dataset == 'Houston18':
            rgb_list = [[81,255,0], [143,255,0], [80, 205, 32], [53, 139, 23], [26, 100, 0],
            [158, 40, 39], [29, 15, 255], [238, 238, 204], [190, 190, 190], [244, 0, 0],
            [238, 238, 223], [132, 139, 131], [134, 68, 9], [133, 0, 0], [250, 200, 0], 
            [255, 255, 0], [200, 148, 0], [250, 192, 203], [9, 3, 130], [108, 148, 240]]
        if dataset == 'Pavia':
            rgb_list = [[255,0,0], [81,255,0], [29, 15, 255], [254, 254, 0],
            [87, 255, 255], [244, 0, 255], [192, 192, 192], [128, 128, 128], [122, 0, 0]]
        if dataset == 'Salinas':
            rgb_list = [[143,255,0], [53, 139, 23], [26, 100, 0],
            [158, 40, 39], [29, 15, 255], [238, 238, 204], [190, 190, 190], [244, 0, 0],
            [238, 238, 223], [132, 139, 131], [133, 0, 0], [250, 200, 0], 
            [255, 255, 0], [250, 192, 203], [9, 3, 130], [108, 148, 240]]

        for c in range(len(rgb_list)):
            index = ((img_gray-1) == c)
            if (index.sum() > 0):
                label_rgb[index, 0] = rgb_list[c][0]
                label_rgb[index, 1] = rgb_list[c][1]
                label_rgb[index, 2] = rgb_list[c][2]

        return label_rgb

def testHS(dataset, img_model, path, norm, PCA, test_flag=0):
    '''
    test_flag: 0 表示模型输出结果是点的方式； 1 表示模型输出结果是图的方式
    '''
    Test = HS_test(dataset, img_model, norm, PCA)

    pre_path = path + '.tif'
    prergb_path = path + '_RGB.tif'
    acc_path = path + '.txt'

    if test_flag == 0:
        pre_map = Test.test_model_pixel(dataset)
    elif test_flag == 1:
        pre_map = Test.test_model_img()

    pre_mapout = Image.fromarray(pre_map)
    pre_mapout.save(pre_path)

    pre_rgb = Test.GRAYcvtRGB(dataset, pre_map)
    pre_rgbout = Image.fromarray(pre_rgb)
    pre_rgbout.save(prergb_path)

    Test.compute_acc(pre_map, acc_path)

if __name__ == "__main__":
    folder = 'result\\HyperSpectral\\Mlp_Auto\\bandwise\\多像素合成\\Bandwise+ReLU\\ratio4，scale1_2_4, 2Dx2D，5X5窗口，特征图求均值，3个波段，bandwise_mlp2层\\Base\\'
    for i in range(3):
        for dataset in ['Pavia', 'Houston13', 'Houston18', 'Salinas']:
            image_folder = folder + dataset + '\\' + str(i)
            image_model_name = image_folder + '\\image_model_xzy500.pkl'
            img_model = torch.load(image_model_name).cuda().eval()
            path = image_folder + '500_new'
            testHS(dataset, img_model, path, norm=True, PCA = False, test_flag=0)
        
        

