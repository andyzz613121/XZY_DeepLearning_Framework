import os
import sys
import random
import numpy as np
from PIL import Image

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from data_processing.Raster import *

def split_imglabel(imgfile, labfile, outfolder, rate=0.7):
    img, paras_img = gdal_read_tif(imgfile)
    lab, paras_lab = gdal_read_tif(labfile)
    assert paras_img[0] == paras_lab[0] and paras_img[1] == paras_lab[1]

    # 计算最大最小值，并写入Metadata中
    max_list = []
    min_list = []
    for b in range(img.shape[0]):
        min, max = np.min(img[b]), np.max(img[b])
        max_list.append(max)
        min_list.append(min)
    add_metadata = {'max':max_list, 'min':min_list}
    # 获取训练图像和测试图像
    img_h = paras_img[0]
    train_h = int(img_h*rate)
    train_img = img[:,0:train_h,:]
    train_imgfile = outfolder + 'img_train.tif'
    test_img = img[:,train_h:,:]
    test_imgfile = outfolder + 'img_test.tif'
    train_lab = lab[0:train_h,:]
    train_labfile = outfolder + 'lab_train.tif'
    test_lab = lab[train_h:,:]
    test_labfile = outfolder + 'lab_test.tif'
    gdal_write_tif(train_imgfile, train_img, train_img.shape[1], train_img.shape[2], train_img.shape[0], datatype=3, add_metadata=add_metadata)
    gdal_write_tif(test_imgfile, test_img, test_img.shape[1], test_img.shape[2], test_img.shape[0], datatype=3, add_metadata=add_metadata)
    gdal_write_tif(train_labfile, train_lab, train_lab.shape[0], train_lab.shape[1], datatype=1)
    gdal_write_tif(test_labfile, test_lab, test_lab.shape[0], test_lab.shape[1], datatype=1)

def split_dataset(img_num, rate=0.7):
    '''
        Input: imgnum_list（数据集中的图像数量）
               rate: 训练集的比例
        
        Output: train_list（训练集的图像序号）
                test_list（测试集的图像序号）
    '''
    num_list = list(range(0, img_num))
    train_list = random.sample(num_list, int(len(num_list)*rate))
    test_list = [x for x in num_list if x not in train_list]
    return train_list, test_list

def save_splitimage(oriimg_folder, trainimg_folder, testimg_folder, train_list, test_list):
    '''
        Input: oriimg_folder（全部数据集文件夹，待分割）
               trainimg_folder（训练数据集文件夹）
               testimg_folder（测试数据集文件夹）
               train_list（split_dataset函数返回的训练集图像序号）
               test_list（split_dataset函数返回的测试集图像序号）
    '''
    filelist = os.listdir(oriimg_folder)
    for item in train_list:
        img = np.array(Image.open(oriimg_folder+filelist[item]))
        img = Image.fromarray(img)
        img.save(trainimg_folder+filelist[item])
    for item in test_list:
        img = np.array(Image.open(oriimg_folder+filelist[item]))
        img = Image.fromarray(img)
        img.save(testimg_folder+filelist[item])

if __name__ == '__main__':
    img_file = 'E:\\dataset\\毕设数据\\new\\2. MS\\Imgs\\6_29.tif'
    lab_file = 'E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\7. labels_with_months\\6. June.tif'
    outfolder = 'E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\'
    split_imglabel(img_file, lab_file, outfolder)

    # blur_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\按是否包含模糊分类图像\\有模糊\\'
    # blurimg_folder = blur_folder + 'blur_image\\'
    # blurimgGray_folder = blur_folder + 'blur_image_gray\\'
    # blurlab_folder = blur_folder + 'blur_label\\'

    # nonblur_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\按是否包含模糊分类图像\\无模糊\\'
    # nonblurimg_folder = nonblur_folder + 'nonblur_image\\'
    # nonblurimgGray_folder = nonblur_folder + 'nonblur_image_gray\\'
    # nonblurlab_folder = nonblur_folder + 'nonblur_label\\'
    
    
    
    # train_list_blur, test_list_blur = split_dataset(982)
    # train_list_nonblur, test_list_nonblur = split_dataset(4921)



    # save_splitimage(blurimg_folder, 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\train\\img\\', 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\test\\img\\', train_list_blur, test_list_blur)
    # save_splitimage(blurimgGray_folder, 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\train\\img_gray\\', 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\test\\img_gray\\', train_list_blur, test_list_blur)
    # save_splitimage(blurlab_folder, 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\train\\label\\', 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\test\\label\\', train_list_blur, test_list_blur)

    # save_splitimage(nonblurimg_folder, 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\train\\img\\', 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\test\\img\\', train_list_nonblur, test_list_nonblur)
    # save_splitimage(nonblurimgGray_folder, 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\train\\img_gray\\', 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\test\\img_gray\\', train_list_nonblur, test_list_nonblur)
    # save_splitimage(nonblurlab_folder, 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\train\\label\\', 'E:\\dataset\\ImageBlur\\Data\\train\\训练集划分\\test\\label\\', train_list_nonblur, test_list_nonblur)
