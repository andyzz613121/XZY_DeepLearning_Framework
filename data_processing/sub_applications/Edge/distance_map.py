import cv2
import numpy as np
import os
from osgeo import gdal
from sklearn import preprocessing

base_folder = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai\\label_gray\\'
out_folder = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai_new\\distance\\'
def make_one_hot(data, class_num):
    array = []
    for i in range(class_num):
        pos_index = (data==i)
        array.append(pos_index.astype(np.uint8))
    array = np.array(array)
    return array

for item in os.listdir(base_folder):
    label_name = base_folder + item
    label_name = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai\\train\\train_image_label\\5657.tif'
    out_name = out_folder + item
    label = cv2.imread(label_name)
    label = label[:,:,0].astype(np.uint8)
    one_hot = make_one_hot(label,6)

    dis_final = np.zeros([label.shape[0], label.shape[1]]).astype(np.float32)
    driver = gdal.GetDriverByName('GTiff')
    data = driver.Create(out_name, one_hot.shape[2], one_hot.shape[1], 1, gdal.GDT_Float32)
    for i in range(one_hot.shape[0]):
        pos_index = (label == i)
        dis = cv2.distanceTransform(one_hot[i], cv2.DIST_L2, 0)
        dis_final[pos_index] = dis[pos_index]
    data.GetRasterBand(1).WriteArray(dis_final)

    