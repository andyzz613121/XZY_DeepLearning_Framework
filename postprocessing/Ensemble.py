import os
import numpy as np
from PIL import Image
import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from visual.Gray2RGB import GRAYcvtRGB
from data_processing.Raster import *
class ISPRS_dataset_Ensemble():   
    def Vaihingen_Ensemble(self, folder):
        print('Ensemble results in Vaihingen dataset')
        self.image_5 = np.zeros([6, 2557, 1887])
        self.image_7 = np.zeros([6, 2557, 1887])
        self.image_23 = np.zeros([6, 2546, 1903])
        self.image_30 = np.zeros([6, 2563, 1934])
        for item in os.listdir(folder):
            try:
                img = np.array(Image.open(folder + item))
                for c in range(6):
                    class_index = (img == c)
                    if 'pre5' in item:
                        self.image_5[c][class_index] += 1
                    elif 'pre7' in item:
                        self.image_7[c][class_index] += 1
                    elif 'pre23' in item:
                        self.image_23[c][class_index] += 1
                    elif 'pre30' in item:
                        self.image_30[c][class_index] += 1
            except:
                continue
        label_folder = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai\\label_gray\\'

        for img_NO in ['5', '7', '23', '30']:
            label_path = label_folder + 'label' + img_NO + '_gray.tif'
            label = np.array(Image.open(label_path))
            if img_NO == '5':
                cur_img = self.image_5
            elif img_NO == '7':
                cur_img = self.image_7
            elif img_NO == '23':
                cur_img = self.image_23
            elif img_NO == '30':
                cur_img = self.image_30
            ensembel_label = np.argmax(cur_img, 0)
            true_index = (label == ensembel_label)
            if img_NO == '5':
                ac = true_index.sum()/(2557*1887)
            elif img_NO == '7':
                ac = true_index.sum()/(2557*1887)
            elif img_NO == '23':
                ac = true_index.sum()/(2546*1903)
            elif img_NO == '30':
                ac = true_index.sum()/(2563*1934)
            
            print('image %s, acc is %f'%(img_NO, ac))
            ensembel_label = Image.fromarray(ensembel_label.astype(np.uint8))
            ensembel_label_outname = folder + str(img_NO) + '_ensemble_' + str(ac) + '.tif'
            ensembel_label.save(ensembel_label_outname)

    def Potsdam_Ensemble(self, folder):
        print('Ensemble results in Potsdam dataset')
        self.image_211 = np.zeros([6, 6000, 6000])
        self.image_410 = np.zeros([6, 6000, 6000])
        self.image_511 = np.zeros([6, 6000, 6000])
        self.image_78 = np.zeros([6, 6000, 6000])
        for item in os.listdir(folder):
            try:
                img = np.array(Image.open(folder + item))
                for c in range(6):
                    class_index = (img == c)
                    if 'pre2_11' in item:
                        self.image_211[c][class_index] += 1
                    elif 'pre4_10' in item:
                        self.image_410[c][class_index] += 1
                    elif 'pre5_11' in item:
                        self.image_511[c][class_index] += 1
                    elif 'pre7_08' in item:
                        self.image_78[c][class_index] += 1
            except:
                continue
        label_folder = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_pos\\label_gray\\'

        for img_NO in ['2_11', '5_11', '4_10', '7_08']:
            label_path = label_folder + 'label' + img_NO + '_gray.tif'
            label = np.array(Image.open(label_path))
            if img_NO == '2_11':
                cur_img = self.image_211
            elif img_NO == '5_11':
                cur_img = self.image_511
            elif img_NO == '4_10':
                cur_img = self.image_410
            elif img_NO == '7_08':
                cur_img = self.image_78
            ensembel_label = np.argmax(cur_img, 0)
            true_index = (label == ensembel_label)
            ac = true_index.sum()/(6000*6000)
            print('image %s, acc is %f'%(img_NO, ac))
            ensembel_label = Image.fromarray(ensembel_label.astype(np.uint8))
            ensembel_label_outname = folder + str(img_NO) + '_ensemble_' + str(ac) + '.tif'
            ensembel_label.save(ensembel_label_outname)

class HS_Ensemble():
    def Houston13_ensemble(self, folder, ensemble_list=[500]):
        print('Ensemble results in Houston13 dataset')
        image = np.zeros([16, 349, 1905])

        for item in ensemble_list:
            img_file = folder + str(item) + '.tif'
            try:
                img = np.array(Image.open(img_file))
                for c in range(16):
                    class_index = (img == c)
                    image[c][class_index] += 1
            except:
                continue

        ensemble_label = np.argmax(image, 0)

        ensemble_label_rgb = GRAYcvtRGB('Houston13', ensemble_label)
        ensemble_label_rgb = Image.fromarray(ensemble_label_rgb)
        ensemble_label_rgb_outname = folder + 'Ensemble_RGB' + '.tif'
        ensemble_label_rgb.save(ensemble_label_rgb_outname)

        ensemble_label = Image.fromarray(ensemble_label.astype(np.uint8))
        ensemble_label_outname = folder + 'Ensemble' + '.tif'
        ensemble_label.save(ensemble_label_outname)

    def Houston18_ensemble(self, folder, ensemble_list=[500]):
        print('Ensemble results in Houston18 dataset')
        image = np.zeros([21, 601, 2384])

        for item in ensemble_list:
            img_file = folder + str(item) + '.tif'
            try:
                img = np.array(Image.open(img_file))
                for c in range(21):
                    class_index = (img == c)
                    image[c][class_index] += 1
            except:
                continue

        ensemble_label = np.argmax(image, 0)

        ensemble_label_rgb = GRAYcvtRGB('Houston18', ensemble_label)
        ensemble_label_rgb = Image.fromarray(ensemble_label_rgb)
        ensemble_label_rgb_outname = folder + 'Ensemble_RGB' + '.tif'
        ensemble_label_rgb.save(ensemble_label_rgb_outname)

        ensemble_label = Image.fromarray(ensemble_label.astype(np.uint8))
        ensemble_label_outname = folder + 'Ensemble' + '.tif'
        ensemble_label.save(ensemble_label_outname)

    def Pavia_ensemble(self, folder, ensemble_list=[500]):
        print('Ensemble results in Pavia dataset')
        image = np.zeros([10, 610, 340])

        for item in ensemble_list:
            img_file = folder + str(item) + '.tif'
            try:
                img = np.array(Image.open(img_file))
                for c in range(10):
                    class_index = (img == c)
                    image[c][class_index] += 1
            except:
                continue

        ensemble_label = np.argmax(image, 0)

        ensemble_label_rgb = GRAYcvtRGB('Pavia', ensemble_label)
        ensemble_label_rgb = Image.fromarray(ensemble_label_rgb)
        ensemble_label_rgb_outname = folder + 'Ensemble_RGB' + '.tif'
        ensemble_label_rgb.save(ensemble_label_rgb_outname)

        ensemble_label = Image.fromarray(ensemble_label.astype(np.uint8))
        ensemble_label_outname = folder + 'Ensemble' + '.tif'
        ensemble_label.save(ensemble_label_outname)

    def Salinas_ensemble(self, folder, ensemble_list=[500]):
        print('Ensemble results in Salinas dataset')
        image = np.zeros([17, 512, 217])

        for item in ensemble_list:
            img_file = folder + str(item) + '.tif'
            try:
                img = np.array(Image.open(img_file))
                for c in range(17):
                    class_index = (img == c)
                    image[c][class_index] += 1
            except:
                continue

        ensemble_label = np.argmax(image, 0)

        ensemble_label_rgb = GRAYcvtRGB('Salinas', ensemble_label)
        ensemble_label_rgb = Image.fromarray(ensemble_label_rgb)
        ensemble_label_rgb_outname = folder + 'Ensemble_RGB' + '.tif'
        ensemble_label_rgb.save(ensemble_label_rgb_outname)

        ensemble_label = Image.fromarray(ensemble_label.astype(np.uint8))
        ensemble_label_outname = folder + 'Ensemble' + '.tif'
        ensemble_label.save(ensemble_label_outname)

class Ensemble():
    def voting(imgpath_list, same_idx=0):
        '''
            Input:
                imgpath_list: 待集成的图像名列表
                same_idx: 若每个类别数量都一样时选择哪个作为最终的类别
        '''
        imglist = []
        class_num = 0
        for imgpath in imgpath_list:
            img, _ = gdal_read_tif(imgpath)
            imglist.append(img)
            img_class = np.max(img)
            class_num = img_class if img_class > class_num else class_num

        h, w = img.shape[0], img.shape[1]
        img_count = np.zeros([class_num, h, w])

        for img in imglist:
            for c in range(class_num):
                class_index = (img == c)
                img_count[c][class_index] += 1

        ensemble_label = np.argmax(img_count, 0)
        # 判断集成后每个类别权重都一样的
        max_count = np.max(img_count, 0)
        negidx = (max_count == 1)
        ensemble_label[negidx] = imglist[same_idx][negidx]

        return ensemble_label

def main(): 
    base_folder = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\Spatial_Spectral\\9_12\\'
    imgpath_list = [base_folder+'pred_seg100_pre_s1.png',base_folder+'pred_seg100_pre_s2.png',base_folder+'pred_seg100_pre_ss.png',base_folder+'pred_9_12_pre.png']
    imgout = Ensemble.voting(imgpath_list, same_idx=1)
    gdal_write_tif(base_folder+'vote.png', imgout, imgout.shape[0], imgout.shape[1])

if __name__ == '__main__':
    main()