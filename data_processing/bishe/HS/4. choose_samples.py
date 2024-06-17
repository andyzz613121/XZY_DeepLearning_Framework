import numpy as np
from PIL import Image
from osgeo import gdal
import random

'''
按离散点采样
'''
label_path = 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Pixs\\Spatial_pred\\lab_train.tif'
# training_rate = 0.01

label = gdal.Open(label_path)
img_w = label.RasterXSize
img_h = label.RasterYSize
label = np.array(label.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
class_num = np.max(label)
train_label = np.zeros_like(label)

#每个像素的坐标列表
xy_list = []     
for x in range(img_h):
    for y in range(img_w):
        xy_list.append([x, y])

#遍历每个类，找到每个类的所有像素，并按照比例选择
# for c in range(1, class_num + 1):
for c in range(class_num):
    class_list = []
    class_index = (label == c)
    class_total_num = class_index.sum()
    # class_train_num = int(class_total_num * training_rate)
    class_train_num = 100 #每个类固定100个

    filter_result = filter(lambda x: class_index[x[0], x[1]]==True, xy_list)
    class_list = list(filter_result)

    class_train_list = random.sample(class_list, class_train_num)

    for item in class_train_list:
        train_label[item[0], item[1]] = c

#验证集数据是全部的label-训练的label
valid_label = label
diff_index = (valid_label == train_label)
valid_label[diff_index] = 0

valid_label = Image.fromarray(valid_label)
train_label = Image.fromarray(train_label)
valid_label.save('E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\HS\\Train\\lab_val.tif')
train_label.save('E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\HS\\Train\\lab_train.tif')