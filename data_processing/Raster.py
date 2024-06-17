import gdal
import gdalconst
# import TiffImagePlugin
from PIL import TiffImagePlugin

import numpy as np
from osgeo import ogr
from osgeo import gdal

GDAL_DATATYPE = \
{
    1: gdal.GDT_Byte,
    2: gdal.GDT_Float32,
    3: gdal.GDT_UInt16
}

def gdal_read_tif(filename):
    '''
    img: c*h*w
    return img, [img_h, img_w, img_c, GeoTransform, Spatial_Ref, img_gdal]
    '''
    img_gdal = gdal.Open(filename)
    img_w = img_gdal.RasterXSize
    img_h = img_gdal.RasterYSize
    img_c = img_gdal.RasterCount

    GeoTransform = img_gdal.GetGeoTransform()
    Spatial_Ref = img_gdal.GetSpatialRef()
    # Proj = img.GetProjection()

    img = np.array(img_gdal.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h))
    return img, [img_h, img_w, img_c, GeoTransform, Spatial_Ref, img_gdal]

def xy2geo(Trans, row, col):
    '''
    图像坐标转地理坐标
    Inputs:
        Trans：仿射变换系数
        row：行号(img_h)
        col：列号(img_w)
    Outputs:
        geox：地理坐标x值
        geoy：地理坐标y值
    '''
    px = Trans[0] + col * Trans[1] + row * Trans[2]
    py = Trans[3] + col * Trans[4] + row * Trans[5]
    return px, py

def geo2xy(Trans, geox, geoy):
    '''
    图像坐标转地理坐标
    Inputs:
        Trans：仿射变换系数
        geox：地理坐标x值
        geoy：地理坐标y值
    Outputs:
        row：行号
        col：列号

    注意：是像素中心的位置（+0.5）
    '''
    delta = Trans[1] * Trans[5] - Trans[2] * Trans[4]
    row = (Trans[5] * (geox - Trans[0]) - Trans[2] * (geoy - Trans[3])) / delta + 0.5
    col = (Trans[1] * (geoy - Trans[3]) - Trans[4] * (geox - Trans[0])) / delta + 0.5
    return row, col

def get_imgbox(img_h, img_w, Trans):
    '''
    获取图像的地理坐标box
    Inputs:
        img：图像
        Trans：仿射变换系数

    Outputs:
        4个box的点坐标
    '''
    sml_x, sml_y = xy2geo(Trans, 0, 0)
    big_x, big_y = xy2geo(Trans, img_h, img_w)
    return (sml_x, sml_y), (sml_x, big_y), (big_x, big_y), (big_x, sml_y)

def gdal_write_tif(filename, img, img_h, img_w, img_c=1, GeoTransform=None, Spatial_Ref=None, datatype=1, add_metadata=None):#img:[c,h,w]
    '''
        Input:
                (1) img: (C, H, W)
                (2) GDAL_DATATYPE:
                    {
                        1: gdal.GDT_Byte,
                        2: gdal.GDT_Float32
                        3: gdal.GDT_UInt16
                    }
    '''
    datatype = GDAL_DATATYPE[datatype]
    driver = gdal.GetDriverByName("GTiff") 
    dataset = driver.Create(filename, img_w, img_h, img_c, datatype)
    if GeoTransform is not None:
        dataset.SetGeoTransform(GeoTransform)
    if Spatial_Ref is not None:
        dataset.SetSpatialRef(Spatial_Ref)
    if add_metadata is not None:
        create_metadata(dataset, add_metadata)

    if len(img.shape) == 2:
        dataset.GetRasterBand(1).WriteArray(img)
    elif len(img.shape) == 3:
        for b in range(img_c):
            dataset.GetRasterBand(1+b).WriteArray(img[b])
    else:
        print('Error: gdal_write_tif//unknow img shape')

def create_metadata(ds, add_metadata):
    '''
        为GDAL图像新建元文件
        metadata(dict):{'max':[1,2,3,4], 'min':[0,1,2,3]...}
    '''
    # return ds.SetMetadata(metadata, "")
    for metadata_att in add_metadata.keys():
        metadata_val = add_metadata[metadata_att]

        if type(metadata_val) == list:
            metadata_val = [str(x) for x in metadata_val]
            metadata_val = " ".join(metadata_val)
        elif type(metadata_val) != str:
            print('The metadata_val of metadata_att: %s should be str or list'%metadata_att)
            continue
        ds.SetMetadataItem(metadata_att, metadata_val)

def get_metadata(img_gdal, metadata_att):
    return img_gdal.GetMetadataItem(metadata_att)

def resampling(tif_input, tif_out, scale):
    '''
    重采样图像分辨率（地理坐标）
    Inputs:
        tif_input: 输入图像路径
        tif_out: 输出图像路径
        scale: 输出的分辨率大小
    '''
    gdal.Warp(tif_out, tif_input,
              resampleAlg=gdalconst.GRA_NearestNeighbour,
              xRes=scale,
              yRes=scale
              )

def imgfilter(img, area_T):
    from skimage.measure import label
    from collections import Counter
    component_lab, component_num = label(img, background=0, return_num=True)
    num_list = Counter(component_lab.reshape([-1]))
    print(num_list)
    for i in range(component_num):
        if num_list[i] < area_T:
            xy_set = np.where(component_lab==i)
            dirs = [[-1,-1], [-1,1],[1,-1],[1,1]]
            find_flag, new_v = 0, -1
            for idx in range(len(xy_set[0])):
                x, y = xy_set[0][idx], xy_set[1][idx]
                for dir in dirs:
                    new_x, new_y = x+dir[0], y+dir[1]
                    new_x, new_y = max(0, new_x), max(0, new_y)
                    new_x, new_y = min(img.shape[0], new_x), min(img.shape[1], new_y)
                    if num_list[component_lab[new_x][new_y]] >= area_T:
                        find_flag = 1
                        new_v = component_lab[new_x][new_y]
                if find_flag == 1:
                    component_lab[xy_set] = new_v
                    break
                        
    return component_lab

def setproject(noref_imgpath, ref_imgpath, outfile):
    nonref_img, _ = gdal_read_tif(noref_imgpath)
    _, para = gdal_read_tif(ref_imgpath)
    gdal_write_tif(outfile, nonref_img, para[0], para[1], para[2], para[3], para[4])

def layer_stack(imgfolder, outfile):
    img_list = []
    for img_path in os.listdir(imgfolder):
        img, para = gdal_read_tif(imgfolder+img_path)
        img = img[None,:,:]
        img_list.append(img)
    img_stack = np.concatenate([img for img in img_list], 0)
    gdal_write_tif(outfile, img_stack, para[0], para[1], img_stack.shape[0], para[3], para[4])
    return True



# def add_tag(tif_img, tag_name, tag_value):
    # file = 'E:\\dataset\\毕设数据\\new\\2. MS\\Imgs\\9_12.tif'
    # tif_img = TiffFile(file)
    # tif_tags = {}
    # for tag in tif_img.pages[0].tags.values():
    #     name, value = tag.name, tag.value
    #     tif_tags[name] = value
    #     print(tag)



if __name__ == '__main__':
    import os
    from PIL import Image
    import cv2
    import torch
    import torch.utils.data.dataset as Dataset
    import torchvision.transforms as transforms
            
    folder = 'D:\\毕业\\博士论文\\毕业论文\\图\\第二章\\基于统计的粗标签处理\\新建文件夹\\'
    outfolder = 'C:\\Users\\25321\\Desktop\\新建文件夹\\'
    for item in os.listdir(folder):
        path = folder + item
        img, _ = gdal_read_tif(path)
        print(img.shape, path)
        img = img[:50,-50:,:]
        img = Image.fromarray(img)
        img.save(outfolder+item)
