import cv2
from PIL import Image
import numpy as np
from osgeo import gdal
import os
base_name = 'D:\\dataset\\suichang\\suichang_round1_dilate_edge\\'
for item in os.listdir(base_name):
    name_dilate_edge = base_name + item
    name_edge_dis = 'D:\\dataset\\suichang\\suichang_round1_dilate_edge_dis\\' + item
    #name_edge_dis_label = 'data\\RS_image_paper\\label_edge_distance_label\\' + str(num) + 'edge_dis_label.tif'
    img=gdal.Open(name_dilate_edge)

    w = img.RasterXSize
    h = img.RasterYSize
    band_num = img.RasterCount

    img = img.ReadAsArray(0,0,w,h)
    img_datatype = img.dtype.name

    img = np.array(img).astype(np.float32)
    #label = np.zeros((img.shape[0],img.shape[1]))
    #print(img.shape,label.shape)
    num = 0

    for i in range(h):
        for j in range(w):
            if img[i,j] == 1:
                dis = 1
                flag = 0
                for size in range(10):
                    for x in range(i-size,i+size+1):
                        for y in range(j-size,j+size+1):
                            if flag==0:
                                x_temp = x
                                y_temp = y
                                if x < 0:
                                    x_temp = 0
                                elif x >= h:
                                    x_temp = h-1
                                if y < 0:
                                    y_temp = 0
                                elif y >= w:
                                    y_temp = w-1

                                if img[x_temp,y_temp]==0:
                                    if size == 3:
                                        dis = 1
                                    elif size == 4:
                                        dis = 0
                                    else:
                                        dis = size
                                    flag = 1
                flag = 0
                img[i,j] = dis    
        
    if 'int8' in img_datatype:
        datatype = gdal.GDT_Byte
    elif 'int16' in img_datatype:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32


   
    img = Image.fromarray(img.astype(np.uint8))
    img.save(name_edge_dis)
    # driver = gdal.GetDriverByName("GTiff")
    # dataset = driver.Create(name_edge_dis, w, h, 1, datatype)
    # dataset.GetRasterBand(1).WriteArray(img)


# name_edge = 'C:\\Users\\ASUS\\Desktop\\edge4.tif'
# name_dilate_edge = 'C:\\Users\\ASUS\\Desktop\\edge2.tif'
# img=gdal.Open(name_dilate_edge)
# w = img.RasterXSize
# h = img.RasterYSize
# band_num = img.RasterCount

# img = img.ReadAsArray(0,0,w,h)
# img_datatype = img.dtype.name

# img = np.array(img).astype(np.float32)

# img1=gdal.Open(name_edge)
# w = img1.RasterXSize
# h = img1.RasterYSize
# band_num = img1.RasterCount

# img1 = img1.ReadAsArray(0,0,w,h)
# img_datatype = img1.dtype.name

# img1= np.array(img1).astype(np.float32)

# for i in range(img1.shape[0]):
#     for j in range(img1.shape[1]):
#         if img[0,i,j]==0 and img[1,i,j] == 255 and img[2,i,j]==0:
#             img1[i,j] += 10
#         elif img[0,i,j]== 0 and img[1,i,j] == 255 and img[2,i,j]==255:
#             img1[i,j] += 30
#         elif img[0,i,j]== 0 and img[1,i,j] == 0 and img[2,i,j]==255:
#             img1[i,j] += 50
# datatype = gdal.GDT_Byte
# driver = gdal.GetDriverByName("GTiff")
# dataset = driver.Create('C:\\Users\\ASUS\\Desktop\\edge3.tif', w, h,1, datatype)

# dataset.GetRasterBand(1).WriteArray(img1)