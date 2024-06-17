import os
import sys
import numpy as np
from PIL import Image

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from model.Self_Module.GLCM import compute_glcm

def computeIC(img):
    '''
        Input: Img_Gray(H, W)
        Output: Std of Image
    '''
    # img = np.reshape(img, -1)
    glcm_para = compute_glcm(img, props={'entropy'})    

    return glcm_para['entropy']

if __name__ == '__main__':
    import os
    basefolder = 'E:\\dataset\\ImageBlur\\Data\\train\\无模糊图像（复杂度+模糊半径）\\High IC\\'
    HICfolder = 'E:\\dataset\\ImageBlur\\Data\\train\\无模糊图像（复杂度+模糊半径）\\selected\\High IC\\'
    LICfolder = 'E:\\dataset\\ImageBlur\\Data\\train\\无模糊图像（复杂度+模糊半径）\\Low IC2\\'
    MICfolder = 'E:\\dataset\\ImageBlur\\Data\\train\\无模糊图像（复杂度+模糊半径）\\selected\\Middle IC\\'

    img_index = 0
    for item in os.listdir(basefolder):
        img_index += 1
        img = np.array(Image.open(basefolder+item))
        IC = computeIC(img)
        if IC > 16:
            img = Image.fromarray(img)
            img.save(HICfolder+item)
        # elif IC > 10 and IC <= 15:
        #     img = Image.fromarray(img)
        #     img.save(MICfolder+item)
        # elif IC > 15:
        #     img = Image.fromarray(img)
        #     img.save(HICfolder+item)
        # print(IC)


    # for item in os.listdir(basefolder):
    #     img = np.array(Image.open(basefolder+item))
    #     # IC = computeIC(img)
    #     pos_index = (img > 50)
    #     if pos_index.sum()/(256*256) > 0.50:
    #         img = Image.fromarray(img)
    #         img.save(LICfolder+item)
