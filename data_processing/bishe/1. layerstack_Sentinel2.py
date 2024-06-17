import os
import cv2
import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from data_processing.Raster import *

outimgsize = 10980
outfile = 'E:\\dataset\\毕设数据\\new\\2. MS\\Imgs\\10_02.tif'
base_folder = 'E:\\dataset\\毕设数据\\new\\2. MS\\S2A_MSIL2A_20221002T164131_N0400_R126_T16SBF_20221002T223756.SAFE\\'
folder_10m = base_folder + 'GRANULE\\'
for item in os.listdir(folder_10m):
    folder_10m = folder_10m + item + '\\IMG_DATA\\' 
    break

folder_20m = folder_10m + 'R20m\\'
folder_60m = folder_10m + 'R60m\\'
folder_10m = folder_10m + 'R10m\\'

name_10m = {}
for item in os.listdir(folder_10m):
    if 'B02' in item:
        name_10m['B02'] = folder_10m + item
    elif 'B03' in item:
        name_10m['B03'] = folder_10m + item
    elif 'B04' in item:
        name_10m['B04'] = folder_10m + item
    elif 'B08' in item:
        name_10m['B08'] = folder_10m + item

name_20m = {}
for item in os.listdir(folder_20m):
    if 'B01' in item:
        name_20m['B01'] = folder_20m + item
    elif 'B05' in item:
        name_20m['B05'] = folder_20m + item
    elif 'B06' in item:
        name_20m['B06'] = folder_20m + item
    elif 'B07' in item:
        name_20m['B07'] = folder_20m + item
    elif 'B8A' in item:
        name_20m['B8A'] = folder_20m + item
    elif 'B11' in item:
        name_20m['B11'] = folder_20m + item
    elif 'B12' in item:
        name_20m['B12'] = folder_20m + item

name_60m = {}
for item in os.listdir(folder_60m):
    if 'B09' in item:
        name_60m['B09'] = folder_60m + item

res_img = []
for item in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']:
    print(item)
    if item in name_10m.keys():
        img, para = gdal_read_tif(name_10m[item])
        res_img.append(img)
    
    if item in name_20m.keys():
        img, _ = gdal_read_tif(name_20m[item])
        img = cv2.resize(img, (outimgsize, outimgsize), cv2.INTER_NEAREST)
        res_img.append(img)
    
    if item in name_60m.keys():
        img, _ = gdal_read_tif(name_60m[item])
        img = cv2.resize(img, (outimgsize, outimgsize), cv2.INTER_NEAREST)
        res_img.append(img)

res_img = np.array(res_img)
gdal_write_tif(outfile, res_img, outimgsize, outimgsize, 12, para[3], para[4], datatype=3)