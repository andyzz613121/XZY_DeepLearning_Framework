import cv2
import sys
import numpy as np

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from data_processing.Raster import gdal_read_tif, gdal_write_tif

def do_kmeans(img, class_num):
    '''
        Input: img(C*H*W)
    '''
    print('Using Kmeans: class_num = %d' %class_num)
    c, h, w = img.shape
    img = img.reshape([img.shape[0], -1]).swapaxes(0,1).astype(np.float32)
    # 进行Kmeans聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0) #停止迭代条件
    flags = cv2.KMEANS_RANDOM_CENTERS   #初始中心选择（随机中心点）

    # compactness, label, center分别是紧密度（每个点到相应聚类中心距离平方和），标志数组，由聚类中心组成的数组
    # compactness: float
    # label: H*W, 1
    # center: class_num, C
    compactness, label, center = cv2.kmeans(img, class_num, None, criteria, 1, flags)
    label = label.reshape([h, w])
    out = np.zeros([c, h, w])
    for i in range(np.max(label)):
        idx = (label == i)
        out[0][idx] = center[i][0]
        out[1][idx] = center[i][1]
        out[2][idx] = center[i][2]

    return out

if __name__ == '__main__':
    img, para = gdal_read_tif('F:\\新建文件夹\\test\\img.tif')
    for class_num in [10, 20, 40, 80, 160, 320]:
        kmeans = do_kmeans(img, class_num)
        out_file = 'F:\\新建文件夹\\test\\pykmeans_' + str(class_num) + '.tif'
        gdal_write_tif(out_file, kmeans, kmeans.shape[2], kmeans.shape[1], 3, para[3], para[4])