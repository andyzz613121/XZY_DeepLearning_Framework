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
from model.Self_Module.SRM_filter import spam11, minmax41
class IMG_Testdata():
    def __init__(self, image_path, norm=True):
        self.image_path = image_path

        self.dataset = RS_dataset(None, train_flag=False, norm=norm)
        self.sample = self.open_test_data(self.image_path)
        self.img_h = self.dataset.img_h
        self.img_w = self.dataset.img_w


    def open_test_data(self, image_path):
        return self.dataset.open_and_procress_data(image_path, None)


class test_IMG():
    def __init__(self, model_path, class_num, TestImg_file, Pre_Folder = 'result\\', pre_imgtag = '', norm=True):
        print("---testing---")
        predict_path = Pre_Folder + pre_imgtag + '_pre.png'
        test_data = IMG_Testdata(TestImg_file, norm)
        self.class_num = class_num
        self.predect_count_list = []
        for classes in range(self.class_num):
            self.predect_count_list.append(np.zeros((test_data.img_h, test_data.img_w)).astype(np.uint8)) #记录每个类的个数
        
        model_img = torch.load(model_path).eval()
        self.test_patches(test_data, model_img, predict_path, 256, 0)
 
    def test_patches(self, test_data, model_img, predict_path, patch_size, run_time):

        img_model = model_img
        img_model.eval()

        img_w = test_data.img_w
        img_h = test_data.img_h

        cur_h = 0
        while cur_h <= img_h: 
            start_h = cur_h 
            end_h = cur_h + patch_size

            if end_h >= img_h:
                end_h = img_h
                start_h = end_h - patch_size
            cur_w = 0
            while cur_w <= img_w:
                start_w = cur_w
                end_w = cur_w + patch_size
                if end_w >= img_w:
                    end_w = img_w
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

        image_predect = np.array([self.predect_count_list[i] for i in range(self.class_num)])    
        image_predect = np.argmax(image_predect, 0).astype(np.uint8)
        img_rgb = self.cvtRGB(image_predect)
        image_predect = Image.fromarray(image_predect)
        image_predect.save(predict_path)
        return True

    def test_single_patch(self, test_data, start_h, end_h, start_w, end_w, model_img):
        with torch.no_grad():
            sample = test_data.sample

            raw_imgs=sample['raw_image']
            img=sample['img']
            img_patch = img[:, :, start_h:end_h, start_w:end_w]
            raw_imgpatch = raw_imgs[:, :, start_h:end_h, start_w:end_w]

            spam_img = spam11(raw_imgpatch)
            minmax_img = minmax41(raw_imgpatch)
            input = torch.cat([img_patch, spam_img, minmax_img], 1)

            img_outputs = model_img(input)[0].unsqueeze(0)
            img_patch.cpu()

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
    # img_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\test\\img_gray\\'
    # for i in range(300, 0, -1):
    #     pre_folder = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\SRNetResult\\' + str(i) + '\\'
    #     model_path = pre_folder + '\\SRNet_model' +str(i)+'.pkl'
    #     img_model = torch.load(model_path)
    #     for item in os.listdir(img_folder):
    #         test = test_IMG(img_model, 2, img_folder+item, Pre_Folder=pre_folder, pre_imgtag=item, norm=True)
    #     break

    # img_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\test\\img\\'
    # for i in range(300, 0, -1):
    #     pre_folder = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\SRNetResult\\划分的数据集，带权交叉熵，灰度，Norm ,SRNET\\' + str(i) + '\\'
    #     model_path = pre_folder + '\\SRNet_model' +str(i)+'.pkl'
    #     img_model = torch.load(model_path)
        
    #     base_folder = 'E:\\dataset\\ImageBlur\\Fuzzy Data\\Img\\'
    #     for blur in os.listdir(base_folder):
    #         for complex in os.listdir(base_folder + blur):
    #             bc_folder = base_folder + blur + '\\' + complex + '\\'
    #             out_folder = pre_folder + blur + '\\' + complex + '\\'
    #             if os.path.exists(out_folder) == False:
    #                 os.makedirs(out_folder)
    #             for img_item in os.listdir(bc_folder):
    #                 img_file = bc_folder + img_item
    #                 test = test_IMG(img_model, 2, img_file, Pre_Folder=out_folder, pre_imgtag=img_item, norm=True)
    #     break
    img_path = 'E:\\dataset\\毕设数据\\img\\LayerStack_20220320.tif'
    model_path = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\deeplabv3_plus_image_model15.pkl'
    pre_path = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\pred.tif'
    test = test_IMG(model_path, 13, img_path, pre_imgtag='pred', norm=True)