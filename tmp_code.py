from collections import Counter
from data_processing.Raster import *
import numpy as np

img1, paras = gdal_read_tif('E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\6. merge\\5. merge_others.tif')

# 将类别进行合并：6月冬小麦与大豆：26应设置为裸地
img2 = np.zeros_like(img1)
img2 = img2 + 300
idx = (img1 == 121) | (img1 == 122) | (img1 == 123) # 建筑
img2[idx] = 0
idx = (img1 == 1) | (img1 == 13) # 玉米
img2[idx] = 1
idx = (img1 == 2)   # 棉花
img2[idx] = 7
idx = (img1 == 3)   # 水稻
img2[idx] = 3
idx = (img1 == 5) | (img1 == 26)  # 大豆
img2[idx] = 7
idx = (img1 == 10) # 花生
img2[idx] = 7
idx = (img1 == 24) # 冬小麦
img2[idx] = 7
idx = (img1 == 61) | (img1 == 37) # 裸地
img2[idx] = 7
idx = (img1 == 111) # 水体
img2[idx] = 8
idx = (img1 == 141) | (img1 == 142) | (img1 == 143) # 森林
img2[idx] = 9
idx = (img1 == 190) | (img1 == 195) # 湿地
img2[idx] = 10
idx = (img1 == 176) # 草地
img2[idx] = 11

# 将类别进行修改（根据图像及物候）
idx1 = (img2 == 1)
idx2 = (img2 == 2)
idx3 = (img2 == 3)
idx4 = (img2 == 4)
idx5 = (img2 == 5)
idx6 = (img2 == 6)
idx7 = (img2 == 7)
idx8 = (img2 == 8)
idx9 = (img2 == 9)
idx10 = (img2 == 10)
idx11 = (img2 == 11)

img2[idx1] = 1
img2[idx3] = 3
img2[idx7] = 2
img2[idx8] = 4
img2[idx9] = 5
img2[idx10] = 6
img2[idx11] = 7

gdal_write_tif('E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\7. labels_with_months\\6. June.tif', img2, paras[0], paras[1], paras[2], paras[3], paras[4], 1)