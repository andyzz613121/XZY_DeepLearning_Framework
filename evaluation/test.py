import numpy as np
from osgeo import gdal
from sklearn.metrics import confusion_matrix

base_path = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\HyperSpectral\\Houston18\\No_Norm_没权重有sideout\\500.tif'
val_path = 'E:\\dataset\\高光谱数据集\\2018IEEE_Contest\\Train\\valid_label.tif'

img = gdal.Open(base_path)
img_w = img.RasterXSize
img_h = img.RasterYSize
img = np.array(img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)

val = gdal.Open(val_path)
val_w = val.RasterXSize
val_h = val.RasterYSize
val= np.array(val.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)

img_list = []
val_list = []
for i in range(val.shape[0]):
    for j in range(val.shape[1]):
        if val[i][j] != 0:
            img_list.append(img[i][j])
            val_list.append(val[i][j])
img_np = np.array(img_list)
val_np = np.array(val_list)

a = np.array(confusion_matrix(val_np, img_np))
for i in range(a.shape[0]):
    print(a[i])
    print('\n')
