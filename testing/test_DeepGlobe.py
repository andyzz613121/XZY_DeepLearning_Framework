import torch
import cv2
import os
import numpy as np
import torch.nn.functional as F
import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from PIL import Image

from dataset.XZY_dataset import DeepGlobe_dataset

class DeepGlobe_test_dataset():
    def __init__(self, image_path):
        self.image_path = image_path

        self.dataset = DeepGlobe_dataset(None, train_flag=False)
        self.sample = self.open_test_data(self.image_path)

    def open_test_data(self, image_path):
        return self.dataset.open_and_procress_data(image_path)

class test_DeepGlobe():
    def start_test(self, model_img, TestImg_Folder='E:\\dataset\\DeepGlobe\\land_valid_sat\\'):
        print("---testing model in DeepGlobe---")
        Pre_Folder = 'result\\DeepGlobe_PreImg\\'
        if os.path.isdir(Pre_Folder) == False:
            os.makedirs(Pre_Folder)

        for item in os.listdir(TestImg_Folder):
            if '.jpg' in item:
                img_No = item.split('_')[0]
                image_path = TestImg_Folder + item
                predict_path = Pre_Folder + img_No + '_mask.png'
                test_data = DeepGlobe_test_dataset(image_path)
                self.test_total_image(test_data, model_img, predict_path)

        
    def test_total_image(self, test_data, model_img, predict_path):
        img_model = model_img
        img_model.eval()

        with torch.no_grad():
            img = test_data.sample['img']
         
            confuse_matrix_input = torch.tensor([[0.0000e+00, 7.6764e-01, 2.4047e-01, 2.3106e-02, 1.0939e-02, 1.2468e-01,
         2.2079e-03],
        [2.6698e-01, 0.0000e+00, 4.0513e-01, 9.2363e-02, 7.5648e-02, 1.0090e-01,
         3.5319e-03],
        [2.1634e-01, 1.0000e+00, 0.0000e+00, 8.4595e-02, 8.8201e-02, 5.9723e-02,
         1.1307e-03],
        [4.6969e-02, 3.1077e-01, 1.0101e-01, 0.0000e+00, 2.3790e-02, 5.7801e-02,
         6.4655e-04],
        [9.2343e-03, 1.6031e-01, 7.1060e-02, 1.6735e-02, 0.0000e+00, 4.0839e-02,
         4.9317e-03],
        [1.2961e-01, 2.7318e-01, 5.6198e-02, 4.4702e-02, 4.3900e-02, 0.0000e+00,
         1.6672e-03],
        [1.6839e-04, 4.6278e-03, 3.8871e-04, 1.1992e-04, 2.6389e-04, 2.0499e-04,
         0.0000e+00]]).cuda()

            output, d1_map, d2_map, d3_map, d4_map = model_img(img, confuse_matrix_input)

        image_predect = np.argmax(output.cpu().detach().numpy()[0], 0).astype(np.uint8)
        # img_rgb = self.cvtRGB(image_predect)
        image_predect = Image.fromarray(image_predect)
        image_predect.save(predict_path)
        return True

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
    
    for i in range(20, 0, -1):
        img_path = '..\\XZY_DeepLearning_Framework\\result\\image_model' +str(i)+'.pkl'
        img_model = torch.load(img_path)
        test = test_DeepGlobe()
        test.start_test(img_model, 'E:\\dataset\\GID数据集\\test\\big_img\\')
        break
