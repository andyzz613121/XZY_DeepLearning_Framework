import os
import numpy as np
from osgeo import gdal

import os
import numpy as np
from glob import glob
from PIL import Image


valid_label = gdal.Open('E:\\dataset\\高光谱数据集\\Pavia\\Train\\valid_label.tif')
pre_label = gdal.Open('D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\HyperSpectral\\Pavia\\Only_SI\\200.tif')
w = pre_label.RasterXSize
h = pre_label.RasterYSize
pre_label = np.array(pre_label.ReadAsArray(0,0,w,h,buf_xsize=w,buf_ysize=h)).astype('float32')
valid_label = np.array(valid_label.ReadAsArray(0,0,w,h,buf_xsize=w,buf_ysize=h)).astype('float32')
label_rgb = np.zeros((pre_label.shape[0],pre_label.shape[1],3)).astype(np.uint8)

# rgb_list = [[45,153,0], [81,255,0], [49, 153, 154], [27, 102, 0],
# [98, 50, 0], [21, 10, 208], [255, 255, 255], [255, 255, 0], 
# [160, 160, 160], [146, 0, 0], [147, 0, 155], [251, 204, 203], [246, 127, 0], 
# [245, 0, 255], [87, 255, 255]]

rgb_list = [[255,0,0], [81,255,0], [29, 15, 255], [254, 254, 0],
[87, 255, 255], [244, 0, 255], [192, 192, 192], [128, 128, 128], [122, 0, 0]]

for c in range(len(rgb_list)):
    index = ((pre_label-1) == c)
    if (index.sum() > 0):
        label_rgb[index, 0] = rgb_list[c][0]
        label_rgb[index, 1] = rgb_list[c][1]
        label_rgb[index, 2] = rgb_list[c][2]
    
zero_index = (valid_label == 0)
print(zero_index.sum())
label_rgb[zero_index, 0] = 0
label_rgb[zero_index, 1] = 0
label_rgb[zero_index, 2] = 0

out = Image.fromarray(label_rgb)
out.save('D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\HyperSpectral\\Pavia\\Only_SI\\200_rgb_background.tif')