from msilib.schema import Feature
from dataclasses import field
from osgeo import ogr
from osgeo import gdal, gdalconst

import xlwt
import random

class Vector():
    def __init__(self, filename=None):
        
        # gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")
        # # 为了使属性表字段支持中文，请添加下面这句
        # gdal.SetConfigOption("SHAPE_ENCODING","")
        # # 注册所有的驱动
        # ogr.RegisterAll()

        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GB2312")
        ogr.RegisterAll()
        
        self.Layer = None
        self.Spatial_Ref = None
        self.Feature_num = 0
        self.ds = None
        self.attribute_table = None
        self.Field = {}
        self.Field_num = 0

        if filename is not None:
            self.gdal_read_shp(filename)
            self.attribute_table, self.Field_num, self.Field = self.Get_attribute_table(self.Layer)

    def  __del__(self):
        if self.ds != None:
            self.ds.Destroy()

    def gdal_read_shp(self, filename, layer_index=0): 
        '''
        return: shp layer, spatial reference, feature_num
        '''
        driver = ogr.GetDriverByName('ESRI Shapefile')
        self.ds = driver.Open(filename, 1)
        if self.ds == None:
            print('Error in function(gdal_read_shp)//open failed')
            return False
        self.Layer = self.ds.GetLayerByIndex(layer_index)
        self.Spatial_Ref = self.Layer.GetSpatialRef()
        self.Feature_num = self.Layer.GetFeatureCount()  #Feature number
        
        print('Read shapefile success!')
        return True

    def Get_attribute_table(self, layer):
        '''
        return: attribute_table(layer_defn)， Field_num
        通过 self.attribute_table.GetNextFeature() 来找到某个属性名在属性表中的编号
        '''
        if layer == None:
            print('Error in function(Get_attribute_table)//layer is None')
            return False
       
        attribute_table = layer.GetLayerDefn() 
        Field_num = attribute_table.GetFieldCount()
        Field = {}
        for i in range(Field_num):
            field_defn = attribute_table.GetFieldDefn(i)
            Field_dict_tmp = {field_defn.GetName():{'Field_Width': field_defn.GetWidth(), 'Field_Type': field_defn.GetType(), 'Field_Precision': field_defn.GetPrecision()}}
            Field.update(Field_dict_tmp)

        print('Get attribute_table success!')
        return attribute_table, Field_num, Field

    def Add_Field(self, layer, field_name, field_type=ogr.OFTInteger64, field_percision=0):
        '''
        return: attribute_table(layer_defn)
        field_type:
        ogr.OFTInteger64        12
        ogr.OFTInteger64List    13
        ogr.OFTReal             2
        ogr.OFTString           4
        ogr.OFTTime             10
        ogr.OFSTNone            0
        ogr.OFTBinary           8
        ogr.OFTDate             9
        '''
        if layer == None:
            print('Error in function(Add_Field)//Layer is None')
            return False

        defn = layer.GetLayerDefn()
        field_index = defn.GetFieldIndex(field_name)
        if field_index >= 0: 
            print('Warning in function(Add_Field)//already had same field_name "%s"'%field_name)
            return False
        else:
            field_defn = ogr.FieldDefn(field_name, field_type)
            field_defn.SetPrecision(field_percision)
            layer.CreateField(field_defn, 1)

        test_field_index = defn.GetFieldIndex(field_name)
        if test_field_index >= 0:
            print('Create Field "%s", datatype "%s", success!'%(field_name, field_type))
        else:
            print('Create Field "%s", datatype "%s", Error!'%(field_name, field_type))
        return True
    
    def get_field_value(self, feature, ATTR):
        return feature.GetField(ATTR)

    def create_empty_shp(self, filename, Spatial_Ref=None, geom_type=ogr.wkbPolygon, ATTR_Table=None):
        """
        docstring
        """
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds = driver.CreateDataSource(filename)
        if ds == None:
            print('Error in function(create_empty_shp)//ds is None')
            return False

        shpLayer=ds.CreateLayer("Polygon_xzy", srs=Spatial_Ref, geom_type=geom_type)
        if shpLayer == None:
            print('Error in function(create_empty_shp)//shpLayer is None')
            return False

        if ATTR_Table is not None:
            for field in ATTR_Table.keys():
                self.Add_Field(shpLayer, field, field_type=ATTR_Table[field]['Field_Type'], field_percision=ATTR_Table[field]['Field_Precision'])
        return ds, shpLayer

    def create_poly_with_position(self, pt_list, layer, ATTRIBUTE, value):
        # 设置多边形坐标
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for pt in pt_list:
            ring.AddPoint(pt[0], pt[1])
        ring.CloseRings()

        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
    
        # # 在layer图层中新建一个字段
        # self.Add_Field(layer, ATTRIBUTE, ogr.OFTString)
        
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(polygon)
        feature.SetField(ATTRIBUTE, value) # 添加属性名

        # 添加多边形
        layer.CreateFeature(feature)
    
    def create_poly_with_feature(self, feature_list, layer):
        # 添加多边形
        for feature in feature_list:
            layer.CreateFeature(feature)

    def create_point_with_position(self, pt_list, layer, ATTR_list, value_list, poly = None):
        '''
            pt_list: [x0, y0]
            poly: 判断点是否在poly内部，如果不在则返回
        '''
        # 图层中增加点文件
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.AddPoint(pt_list[0], pt_list[1])
        geometry = poly.GetGeometryRef()

        if poly is not None:
            if not pt.Within(geometry):
                return False
        # 在layer图层中新建一个字段
        for att in ATTR_list:
            self.Add_Field(layer, att, ogr.OFTString)
        
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(pt)
        for i in range(len(value_list)):
            feature.SetField(ATTR_list[i], value_list[i]) # 添加属性名

        # 添加多边形
        layer.CreateFeature(feature)
        return True

    def add_feature_to_layer(self, feature_list, layer):
        '''
        对layer图层插入feature矢量
        '''
        for feat in feature_list:
            geo_tmp = feat.GetGeometryRef()
            feat_tmp = ogr.Feature(layer.GetLayerDefn())

            for field in self.Field:
                field_value = feat.GetField(field)
                feat_tmp.SetField(field, field_value)

            feat_tmp.SetGeometry(geo_tmp)
            # feat_tmp.SetField('CodeXZY', 1)
            layer.CreateFeature(feat_tmp)

    def Get_feature_total(self, layer):
        '''
        for i, data in enumerate(shp1_features):
            print(data.GetField(0),data.GetField(1),data.GetField(2),data.GetField(3),data.GetField(4))

        for item in feature:
            print(item.GetGeometryRef().GetX(), item.GetGeometryRef().GetY())
        '''
        Feature_num = layer.GetFeatureCount()
        Feature_list = []
        for i in range(Feature_num):
            feature = layer.GetNextFeature()
            Feature_list.append(feature)
        return Feature_list
        
    def Get_feature_by_index(self, layer, feature_index):
        # if self.attribute_table == None:
        #     print('Error in function(Get_feature_by_index)//attritube_table is None')
        #     return False

        # if feature_index >= self.Feature_num:
        #     print('Error in function(Get_feature_by_index)//feature_index >= Feature_num')
        return layer.GetFeature(feature_index)
    
    def Get_feature_by_value(self, layer, ATTR, value):
        Feature_num = layer.GetFeatureCount()
        Feature_list = []
        for i in range(Feature_num):
            feature = layer.GetNextFeature()
            field_value = feature.GetField(ATTR)
            if field_value == value:
                Feature_list.append(feature)
        return Feature_list

    def Set_feature_value_by_name(self, feature, name, value):
        feature.SetField(name, value)
        self.Layer.SetFeature(feature)
        return True
    
    def Get_feature_boundary(self, feature):
        geometry = feature.GetGeometryRef()
        boundary = geometry.GetBoundary()
        point_num = boundary.GetPointCount()
        point_list = []
        for i in range(point_num):
            point_list.append([boundary.GetX(i), boundary.GetY(i)])
        return point_list

    def Get_feature_envelope(self, feature):
        '''
        return: 'minX': envelope[0], 'maxX': envelope[1], 'minY': envelope[2], 'maxY': envelope[3]
        '''
        geometry = feature.GetGeometryRef()
        envelope = geometry.GetEnvelope()
        
        return envelope

    def convert2raster(self, raster_att, layer, outfilename, ATTRIBUTE):
        '''
        转换矢量为栅格图像
        Inputs：
            raster_att：栅格的属性[img_h, img_w, img_c, GeoTransform, Spatial_Ref]
            layer：矢量待转换的图层
            outfilename：输出的名称

        '''    
        Spatial_Ref = raster_att[4]
        geo_transform = raster_att[3]
        cols = raster_att[1]  # 列数
        rows = raster_att[0]  # 行数
        
        x_min = geo_transform[0]
        y_min = geo_transform[3]
        pixel_width = geo_transform[1]
        
        target_ds = gdal.GetDriverByName('GTiff').Create(outfilename, xsize=cols, ysize=rows, bands=1, eType=gdal.GDT_Byte)
        target_ds.SetGeoTransform(geo_transform)
        target_ds.SetSpatialRef(Spatial_Ref)
        
        band = target_ds.GetRasterBand(1)
        # band.SetNoDataValue(0)
        band.FlushCache()
        cvt_keyvalue = "ATTRIBUTE=" + str(ATTRIBUTE)
        print(cvt_keyvalue)
        gdal.RasterizeLayer(target_ds, [1], layer, options=[cvt_keyvalue]) # 跟shp字段给栅格像元赋值
        # gdal.RasterizeLayer(target_ds, [1], m_layer) # 多边形内像元值的全是255
        del target_ds

    def rand_samp_point_within_polygon(self, out_filename, pt_num, polygons, ATTR_list = []):
        '''
            在输入的多边形内部随机选择K个点，如吉林省白城市矢量中选择K个点进行验证
            polygons: list --> [poly_1, poly_2... poly_n]
        '''
        _, out_layer = self.create_empty_shp(out_filename, self.Spatial_Ref, ogr.wkbPoint)

        for polygon in polygons:
            cur_num = 0
            envelope = self.Get_feature_envelope(polygon)
            minX, maxX, minY, maxY = envelope[0], envelope[1], envelope[2], envelope[3]
            while cur_num < pt_num:
                rand_x = random.uniform(minX, maxX)
                rand_y = random.uniform(minY, maxY)
                out_value_list = []
                out_ATTR_list = []
                for item in ATTR_list:
                    out_value_list.append(self.get_field_value(polygon, item))
                    out_ATTR_list.append(item + '_OUT')
                if self.create_point_with_position([rand_x, rand_y], out_layer, out_ATTR_list, out_value_list, polygon):
                    cur_num += 1

    def rand_samp_point_within_att(self, out_filename, pt_num, pt_list, ATTR):
        '''
            在输入的点中，根据属性值，对每个属性值随机选择K个点
            pt_list: list --> [poly_1, poly_2... poly_n]
        '''
        _, out_layer = self.create_empty_shp(out_filename, self.Spatial_Ref, ogr.wkbPoint)
        pt_att_dict = {}
        for pt in pt_list:
            v = self.get_field_value(pt, ATTR)
            if v not in pt_att_dict.keys():
                pt_att_dict[v] = [pt]
            else:
                pt_att_dict[v].append(pt)
        
        insert_pt_list = []
        for key in pt_att_dict.keys():
            rand_list = list(range(0, len(pt_att_dict[key])))
            if len(rand_list) < pt_num:
                print('Pt number < Select number')
                continue
            select_list = random.sample(rand_list, pt_num)
            for idx in select_list:
                pt_select = pt_att_dict[key][idx]
                out_layer.CreateFeature(pt_select)
           
        return True
                    
    # def rand_samp_point_within_region(self, out_filename, poly_num, pt_num, regions, polygons, ATTR_list = []):
    #     '''
    #         在输入的多边形区域里面，随机选择K个多边形，并在多边形中选择n个点，如吉林省白城市提取结果10000个图斑，选择100个图斑，取每个图斑中的1个点进行验证
    #         region_polygons: list --> [poly_1, poly_2... poly_n]
    #     '''
    #     _, out_layer = self.create_empty_shp(out_filename, self.Spatial_Ref, ogr.wkbPoint)
    #     poly_regions = [[] for i in range(len(regions))]
    #     for i in range(len(polygons)):
    #         for j in range(len(regions)):
    #             poly_geo = polygons[i].GetGeometryRef()
    #             region_geo = regions[j].GetGeometryRef()
    #             union_geo = poly_geo.Union(region_geo)
    #             if union_geo.Equal(region_geo):
    #                 poly_regions[j].append(polygons[i])
    #                 break
    #     print('11111', len(poly_regions), len(poly_regions[0]))
    #     for i in range(len(poly_regions)):
    #         poly_region = poly_regions[i]
    #         if len(poly_region) < poly_num:
    #             print('Polygon number < Select number')
    #             continue
    #         poly_list = list(range(0, len(poly_region)))
    #         select_list = random.sample(poly_list, poly_num)

    #         for idx in select_list:
    #             cur_num = 0
    #             envelope = self.Get_feature_envelope(poly_regions[i][idx])
    #             minX, maxX, minY, maxY = envelope[0], envelope[1], envelope[2], envelope[3]
    #             while cur_num < pt_num:
    #                 rand_x = random.uniform(minX, maxX)
    #                 rand_y = random.uniform(minY, maxY)
    #                 out_value_list = []
    #                 out_ATTR_list = []
    #                 for item in ATTR_list:
    #                     out_value_list.append(self.get_field_value(poly_regions[i][idx], item))
    #                     out_ATTR_list.append(item + '_OUT')
    #                 if self.create_point_with_position([rand_x, rand_y], out_layer, out_ATTR_list, out_value_list, poly_regions[i][idx]):
    #                     cur_num += 1

    def write_position_to_excel(self, exl_filename, features, ATTR_list = []):
        workbook = xlwt.Workbook(encoding = 'utf-8')
        worksheet = workbook.add_sheet('XZY')
        line_num = 0
        for feat in features:
            envelope = self.Get_feature_envelope(feat)
            minX, maxX, minY, maxY = envelope[0], envelope[1], envelope[2], envelope[3]
            ct_X = (minX + maxX)/2
            ct_Y = (minY + maxY)/2
            value_list = []
            for attr in ATTR_list:
                v = self.get_field_value(feat, attr)
                v = 0 if v == None else v
                value_list.append(v)

            worksheet.write(line_num, 0, label = ct_X)
            worksheet.write(line_num, 1, label = ct_Y)
            for i in range(2, 2+len(value_list)):
                worksheet.write(line_num, i, label = value_list[i-2])
            line_num += 1
        workbook.save(exl_filename)

    def delete_feature_by_attibute(self, oldlayer, ATTRIBUTE, value, rules=1):
        '''
        Inputs:
                ATTRIBUTE:  要删除的属性
                value:  要对比的属性值
                rules:  
                    1: 删除 < value的;
                    2: 删除 > value的;
                    3: 删除 <= value的;
                    4: 删除 >= value的;
                    5: 删除 == value的;
                    6: 删除 != value的。
        Outputs:
                selected feature idx list
        Usage:
                1. new_shpfilename = 'C:\\Users\\25321\\Desktop\\test\\新建文件夹\\shaixuan.shp'             # 新建一个空矢量
                2. _, _, attr_table = vect.Get_attribute_table(vect.Layer)                                  # 获取旧矢量的属性表
                3. _, newlayer = vect.create_empty_shp(new_shpfilename, vect.Layer.GetSpatialRef(), ATTR_Table=attr_table)  # 把旧矢量属性表并写入新的空矢量中
            对单个筛选条件：
                4. select_featureidxlist = vect.delete_feature_by_attibute(vect.Layer, "Match_Dist", 0.3, 2)        # 调用该函数获取按规则筛选过的feature idx
                5. select_featurelist = [vect.Get_feature_by_index(vect.Layer, i) for i in select_featureidxlist]   # 根据idx找到对应的矢量Feature
                6. vect.create_poly_with_feature(select_featurelist, newlayer)                                      # 把筛选过的Feature写入新的空矢量中
            对多个筛选条件：
                4. select_featureidxlist1 = vect.delete_feature_by_attibute(vect.Layer, "面积", 1000, 3)            # 调用该函数获取按规则筛选过的feature idx
                5. select_featureidxlist2 = vect.delete_feature_by_attibute(vect.Layer, "Match_Dist", 0.3, 2)       # 调用该函数获取按规则筛选过的feature idx
                6. select_featureidxlist = list(set(select_featureidxlist1)&set(select_featureidxlist2))            # 合并两个规则得到的feature idx (与)

                7. select_featurelist = [vect.Get_feature_by_index(vect.Layer, i) for i in select_featureidxlist]   # 根据idx找到对应的矢量Feature
                8. vect.create_poly_with_feature(select_featurelist, newlayer)                                      # 把筛选过的Feature写入新的空矢量中
        '''
        Feature_num = oldlayer.GetFeatureCount()
        Feature_list = []
        for i in range(Feature_num):
            feature = self.Get_feature_by_index(oldlayer, i)
            feat_value = self.get_field_value(feature, ATTRIBUTE)
            if type(value) in [int, float]:
                if feat_value == None:
                    feat_value = 0
                feat_value = float(feat_value)
                
            if rules == 1 and feat_value >= value:
                Feature_list.append(i)
            elif rules == 2 and feat_value <= value:
                Feature_list.append(i)
            elif rules == 3 and feat_value > value:
                Feature_list.append(i)
            elif rules == 4 and feat_value < value:
                Feature_list.append(i)
            elif rules == 5 and feat_value != value:
                Feature_list.append(i)
            elif rules == 6 and feat_value == value:
                Feature_list.append(i)
        return Feature_list

if __name__ == '__main__':

    vect = Vector('C:\\Users\\25321\\Desktop\\test\\新建文件夹\\result1.shp')
    new_shpfilename = 'C:\\Users\\25321\\Desktop\\test\\新建文件夹\\shaixuan.shp'             
    _, _, attr_table = vect.Get_attribute_table(vect.Layer)                                
    _, newlayer = vect.create_empty_shp(new_shpfilename, vect.Layer.GetSpatialRef(), ATTR_Table=attr_table)
    select_featureidxlist2 = vect.delete_feature_by_attibute(vect.Layer, "面积", 1000, 3)
    select_featureidxlist1 = vect.delete_feature_by_attibute(vect.Layer, "Match_Dist", 0.15, 2)
    select_featureidxlist = list(set(select_featureidxlist1)&set(select_featureidxlist2))

    select_featurelist = [vect.Get_feature_by_index(vect.Layer, i) for i in select_featureidxlist]
    vect.create_poly_with_feature(select_featurelist, newlayer)

    