import os
from osgeo import gdal
from osgeo import ogr
import numpy as np
from Raster import gdal_read_tif, gdal_write_tif, get_imgbox
from Vector import Vector
def Coordinate_alignment_oneclass(raster_filename, shape_filename, outfilename, ATTRIBUTE='CodeXZY'):
    '''
    1. 新建一个layer（layer_new）
    2. 在其中新建一个与栅格空间box一致的Feature，增加一列属性值，并修改属性值为0
    3. 拷贝待转换矢量Vect.Layer中的所有矢量，增加一列属性值，并修改属性值为1
    4. 将新的待拷贝矢量添加到新的layer中，并转换为栅格，从而使得两张图像大小一致
    Inputs：
        raster_filename：栅格文件名
        shp_filename：输入的原始矢量文件名
        outfilename：矢量转栅格后输出的文件名
        ATTRIBUTE：矢量转换的关键字段
    Outputs:
        二值图像（0，1），1代表原始矢量有的地方，0代表原始矢量没有的地方
    '''
    
    # 读遥感影像 & 矢量
    _, att = gdal_read_tif(raster_filename)  
    [img_h, img_w, img_c, Trans, Spatial_Ref, img_gdal]  = att
    Vect = Vector(shape_filename)         
    
    # 创建新的矢量
    new_ds = ogr.GetDriverByName('Memory').CreateDataSource('shapefile')
    # new_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource('D:\\1111.shp')
    layer_new = new_ds.CreateLayer('poly', srs=Spatial_Ref, geom_type=ogr.wkbPolygon)
    Vect.Add_Field(layer_new, ATTRIBUTE)

    # 找到遥感影像的最大范围BOX，并在layer_new中新增一个多边形
    Img_box = get_imgbox(img_h, img_w, Trans)

    # 对空的layer_new先增加范围多边形，并将范围多边形属性设置为0
    Vect.create_poly_with_position(Img_box, layer_new, ATTRIBUTE, 0)
    feat_list = Vect.Get_feature_total(Vect.Layer)
    
    # 对空的layer_new再增加原始矢量多边形，并将除了第一个范围多边形外其它的属性设置为1
    # 不直接改了属性再加到layer_new里面是因为原始的Vect.layer没有新增的属性名
    # 这里注意，修改矢量名称后，一定要layer_new.CreateFeature(feat)！！！
    # 还要注意，add_feature_to_layer函数里面一定要用旧的矢量的Geometry重新生成一遍，不然会找不到属性表，报ERROR 1: Invalid index : -1错，但是好像不用重新SetField，可以在CreateFeature
    # 之后函数外面SetField，然后重新CreateFeature
    Vect.add_feature_to_layer(feat_list, layer_new)
    feat_list_new = Vect.Get_feature_total(layer_new)
    first_flag = True
    for feat in feat_list_new:
        if first_flag == False:
            feat.SetField(ATTRIBUTE, 1)
            layer_new.CreateFeature(feat)
        first_flag = False

    # 输出图像
    Vect.convert2raster(att, layer_new, outfilename, ATTRIBUTE)
    return

def Coordinate_alignment_mulclass(raster_filename, shape_filename, outfilename, ATTRIBUTE='Code'):
    '''
    1. 新建一个layer（layer_new）
    2. 在其中新建一个与栅格空间box一致的Feature，增加一列属性值，并修改属性值为0
    3. 拷贝待转换矢量Vect.Layer中的所有矢量，增加一列属性值，并修改属性值为1
    4. 将新的待拷贝矢量添加到新的layer中，并转换为栅格，从而使得两张图像大小一致
    Inputs：
        raster_filename：栅格文件名
        shp_filename：输入的原始矢量文件名
        outfilename：矢量转栅格后输出的文件名
        ATTRIBUTE：矢量转换的关键字段
    Outputs:
        多值图像（0，1），不同值代表原始矢量对应ATTRIBUTE的值
    '''
    
    # 读遥感影像 & 矢量
    _, att = gdal_read_tif(raster_filename)  
    [img_h, img_w, img_c, Trans, Spatial_Ref, img_gdal]  = att
    Vect = Vector(shape_filename)         
    
    # 创建新的矢量
    new_ds = ogr.GetDriverByName('Memory').CreateDataSource('shapefile')
    # new_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource('D:\\1111.shp')
    layer_new = new_ds.CreateLayer('poly', srs=Spatial_Ref, geom_type=ogr.wkbPolygon)
    for field in Vect.Field:
        Vect.Add_Field(layer_new, field)
    
    # 找到遥感影像的最大范围BOX，并在layer_new中新增一个多边形
    Img_box = get_imgbox(img_h, img_w, Trans)

    # 对空的layer_new先增加范围多边形，并将范围多边形属性设置为0
    Vect.create_poly_with_position(Img_box, layer_new, ATTRIBUTE, 0)
    feat_list = Vect.Get_feature_total(Vect.Layer)
    for feat in feat_list:
        feat_field = feat.GetField(ATTRIBUTE)
        
    # 对空的layer_new再增加原始矢量多边形，并将除了第一个范围多边形外其它的属性设置为1
    # 不直接改了属性再加到layer_new里面是因为原始的Vect.layer没有新增的属性名
    # 这里注意，修改矢量名称后，一定要layer_new.CreateFeature(feat)！！！
    # 还要注意，add_feature_to_layer函数里面一定要用旧的矢量的Geometry重新生成一遍，不然会找不到属性表，报ERROR 1: Invalid index : -1错，但是好像不用重新SetField，可以在CreateFeature
    # 之后函数外面SetField，然后重新CreateFeature

    # 之前报错ERROR 1: Invalid field name: '面积'是因为没有增加82-83行 在新的图层中增加属性的问题
    Vect.add_feature_to_layer(feat_list, layer_new)
    feat_list_new = Vect.Get_feature_total(layer_new)
    first_flag = True
    for feat in feat_list_new:
        if first_flag == False:
            feat_field = feat.GetField(ATTRIBUTE)
            # print(feat_field, ATTRIBUTE)
            # feat.SetField(ATTRIBUTE, feat_field)
            layer_new.CreateFeature(feat)
        first_flag = False

    # 输出图像
    Vect.convert2raster(att, layer_new, outfilename, ATTRIBUTE)
    return

def raster2vector(shape_filename, attribute, raster_filename):
    _, para = gdal_read_tif(raster_filename)
    img = para[5].GetRasterBand(1)

    vect = Vector()
    ds, shpLayer = vect.create_empty_shp(shape_filename, para[4])
    vect.Add_Field(shpLayer, attribute)
    gdal.Polygonize(img, None, shpLayer, 0, [], callback=None)
    return True


if __name__ == '__main__':
    # shape_filename = 'C:\\Users\\25321\\Desktop\\test\\result1.shp'
    # raster_filename = 'C:\\Users\\25321\\Desktop\\test\\kmeans.tif'
    # out_filename = 'C:\\Users\\25321\\Desktop\\test\\result1.tif'
    # Coordinate_alignment_mulclass(raster_filename, shape_filename, out_filename, ATTRIBUTE='code_1')

    raster_filename = 'D:\\Code\\XZY_Detect\\result\\111\\returnmask1.tif'
    out_filename = 'C:\\Users\\25321\\Desktop\\test\\新建文件夹 (2)\\111.shp'
    raster2vector(out_filename, 'code', raster_filename)
    