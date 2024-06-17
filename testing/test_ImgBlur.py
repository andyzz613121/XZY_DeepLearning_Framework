import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

import os
import torch
import numpy as np
from PIL import Image

from testing.XZY_testImage_Base import XZY_testIMG_classification_Base

class XZY_testIMG_ImgBlur(XZY_testIMG_classification_Base):
    def __init__(self, model_img, class_num, TestImgfile_list, Pre_Folder = 'result\\', pre_imgtag = '', norm=True, gpu=True, randomtime=0):
        super(XZY_testIMG_ImgBlur, self).__init__(model_img, class_num, TestImgfile_list, Pre_Folder, pre_imgtag, norm, gpu, randomtime=randomtime)
        
    def test_single_patch(self, test_samples, start_h, end_h, start_w, end_w, model_img):
        with torch.no_grad():

            img = test_samples['img_0']
            # srm = test_samples['img_1']
            
            img_patch = img[:, :, start_h:end_h, start_w:end_w]
            # srm_patch = srm[:, :, start_h:end_h, start_w:end_w]
            # input = torch.cat([img_patch, srm_patch], 1)

            input = img_patch
            img_outputs = model_img[0](input)
            # img_outputs = model_img[0]([input], 'base')
            self.agg_vote(img_outputs, start_h, end_h, start_w, end_w)



if __name__ == "__main__":

    # for i in ['High IC', 'Low IC', 'Middle IC']:
    #     for r in [ 1.2, 1.4, 1.6, 1.8]:
    #         img_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\实验图像_(复杂度+模糊半径)\\选择的实验图像\\全部\\r' + str(r) + '\\' + i + '\\img_rgb\\'
    #         srm_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\实验图像_(复杂度+模糊半径)\\选择的实验图像\\全部\\r' + str(r) + '\\' + i + '\\minspam\\'
    #         model_path = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\SRNetResult\\2\\200\\SRNet_model200.pkl'
    #         pre_folder = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\SRNetResult\\2\\r' + str(r) + '\\' + i + '\\'
    #         if os.path.exists(pre_folder) == False:
    #             os.makedirs(pre_folder)

    #         img_model = torch.load(model_path)
    #         for item in os.listdir(img_folder):
    #             input_list = [img_folder+item, srm_folder+item]
    #             test = XZY_testIMG_ImgBlur(img_model, 2, input_list, Pre_Folder=pre_folder, pre_imgtag=item, norm=True)
        

    # img_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\训练图像\\test\\img\\'
    # srm_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\训练图像\\test\\minspam_gray\\'
    # out_folder = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\SRNetResult\\2\\predict\\'
    # model_path = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\SRNetResult\\2\\200\\SRNet_model200.pkl'
    # img_model = torch.load(model_path)
    # for img_item in os.listdir(img_folder):
    #     img_file = img_folder + img_item
    #     srm_file = srm_folder + img_item
    #     input_list = [img_file, srm_file]
    #     test = XZY_testIMG_ImgBlur(img_model, 2, input_list, Pre_Folder=out_folder, pre_imgtag=img_item, norm=True)
    #     print(test)

    for times in [2,3,4,5]:
        img_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\训练图像\\test\\img\\'
        srm_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\训练图像\\test\\minspam_gray\\'
        base_folder = 'result\\SRNetResult\\2\\'
        out_folder = base_folder + 'Predict\\200\\'
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        model_path = base_folder + 'Model\\SRNet_model200.pkl'
        # img_model = torch.load(model_path)
        for img_item in os.listdir(img_folder):
            img_file = img_folder + img_item
            srm_file = srm_folder + img_item
            input_list = [img_file]
            test = XZY_testIMG_ImgBlur([model_path], 2, input_list, Pre_Folder=out_folder, pre_imgtag=img_item, norm=True)