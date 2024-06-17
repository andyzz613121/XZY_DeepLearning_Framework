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
    if len(lab.shape) == 2:
        train_lab = lab[0:train_h,:]
        train_labfile = outfolder + 'lab_train.tif'
        test_lab = lab[train_h:,:]
        test_labfile = outfolder + 'lab_test.tif'
        gdal_write_tif(train_labfile, train_lab, train_lab.shape[0], train_lab.shape[1], datatype=1)
        gdal_write_tif(test_labfile, test_lab, test_lab.shape[0], test_lab.shape[1], datatype=1)
    else:
        train_lab = lab[:,0:train_h,:]
        train_labfile = outfolder + 'lab_train.tif'
        test_lab = lab[:,train_h:,:]
        test_labfile = outfolder + 'lab_test.tif'
        print(test_lab.shape, train_lab.shape)
        gdal_write_tif(train_labfile, train_lab, train_lab.shape[1], train_lab.shape[2], train_lab.shape[0], datatype=1)
        gdal_write_tif(test_labfile, test_lab, test_lab.shape[1], test_lab.shape[2], test_lab.shape[0], datatype=1)

    gdal_write_tif(train_imgfile, train_img, train_img.shape[1], train_img.shape[2], train_img.shape[0], datatype=3, add_metadata=add_metadata)
    gdal_write_tif(test_imgfile, test_img, test_img.shape[1], test_img.shape[2], test_img.shape[0], datatype=3, add_metadata=add_metadata)
    

def split_HSimg(labfile, outfolder, class_train_num=100):
    '''
    按离散点采样
    '''
    label = gdal.Open(labfile)
    img_w = label.RasterXSize
    img_h = label.RasterYSize
    label = np.array(label.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.int8)
    # label = label[5,:,:]
    class_num = np.max(label)
    train_label = np.zeros_like(label) - 1

    #遍历每个类，找到每个类的所有像素，并按照比例选择
    # for c in range(1, class_num + 1):
    for c in range(class_num+1):
        class_index = np.array(np.where(label == c))
        class_train_idx = np.random.choice(np.arange(len(class_index[0])), size=class_train_num, replace=False)
        class_train_xy = class_index[:,class_train_idx]
        for i in range(len(class_train_xy[0])):
            x, y = class_train_xy[0][i], class_train_xy[1][i]
            train_label[x][y] = c
            
    #验证集数据是全部的label-训练的label
    valid_label = label
    diff_index = (valid_label == train_label)
    valid_label[diff_index] = -1

    valid_label = Image.fromarray(valid_label)
    train_label = Image.fromarray(train_label)
    valid_label.save(outfolder+'lab_val.tif')
    train_label.save(outfolder+'lab_train.tif')


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
    # img_file = 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Imgs\\Spectral_pred\\Spectral_layerstack.tif'
    # lab_file = 'E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\8. labels_with_year\\labels_with_year.tif'
    # outfolder = 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Imgs\\Spectral_pred\\'
    # split_imglabel(img_file, lab_file, outfolder)

    outfolder = 'E:\\dataset\\毕设数据\\new\\2. MS\\SS_month_Pixs_200\\10_17\\'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    label_path = 'E:\\dataset\\毕设数据\\new\\2. MS\\SS_month\\labelmonth_layerstack.tif'
    split_HSimg(label_path, outfolder, 500)

    # outfolder = 'E:\\dataset\\毕设数据\\new\\2. MS\\SS_month\\'
    # split_imglabel('E:\\dataset\\毕设数据\\new\\2. MS\\SS_month\\SS_layerstack.tif', 'E:\\dataset\\毕设数据\\new\\2. MS\\SS_month\\labelmonth_layerstack.tif', outfolder)