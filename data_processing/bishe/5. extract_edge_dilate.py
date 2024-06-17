import os
import cv2
import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
import numpy as np
from data_processing.Raster import *
from dataset.create_dataset.write_csv import *

base_folder = 'E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\Train\\10.17\\'
lab_folder = base_folder + 'lab\\'
edge_folder = base_folder + 'edge\\'
edgedi_folder = base_folder + 'edge_dilate\\'
if not os.path.exists(edge_folder):
    os.makedirs(edge_folder)
if not os.path.exists(edgedi_folder):
    os.makedirs(edgedi_folder)

total_num = 0
for item in os.listdir(lab_folder):
    name_gray = lab_folder + item
    name_edge = edge_folder + item
    name_dilate_edge = edgedi_folder + item

    img, para = gdal_read_tif(name_gray)
    img_copy = np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cur_class = img[i,j]
            #print(cur_class)
            leftx = i-1
            lefty = j
            upx = i
            upy = j-1 
            rightx = i+1
            righty = j
            downx = i
            downy = j+1
            if i-1 < 0:
                leftx = 0
            if j-1 < 0:
                upy = 0
            if i+1 >= img.shape[0]:
                rightx = img.shape[0]-1
            if j+1 >= img.shape[1]:
                downy = img.shape[1]-1
            if img[leftx,lefty] != cur_class or img[upx,upy] != cur_class or img[downx,downy] != cur_class or img[rightx,righty] != cur_class:
                img_copy[i,j] = 1

    gdal_write_tif(name_edge, img_copy, para[0], para[1], para[2])
    # cv2.imwrite(name_edge,img_copy)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
    img_copy = cv2.dilate(img_copy, kernel)

    gdal_write_tif(name_dilate_edge, img_copy, para[0], para[1], para[2])
    # cv2.imwrite(name_dilate_edge, img_copy)
    total_num += 1

writecsv_number(total_num, base_folder, 'train_edge_dilate', ['img', 'edge_dilate', 'lab'])
writecsv_number(total_num, base_folder, 'train_edge', ['img', 'edge', 'lab'])