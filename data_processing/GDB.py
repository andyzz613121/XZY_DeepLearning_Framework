
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from data_processing.Vector import Vector

class GDB_XZY():
    def __init__(self, filename=None, readmode=0):
        '''
            readmode = 0 只读
            readmode = 1 读写
        '''
        # gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")
        # # 为了使属性表字段支持中文，请添加下面这句
        # gdal.SetConfigOption("SHAPE_ENCODING","")
        # # 注册所有的驱动
        # ogr.RegisterAll()

        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GB2312")
        ogr.RegisterAll()
        
        self.gdb = None
        self.iLayerCount = None

        if filename is not None:
            self.open_gdb(filename, readmode)

    def  __del__(self):
        if self.gdb != None:
            self.gdb.Destroy()
            del self.gdb

    def open_gdb(self, gdb_path, readmode = 0):
        '''
            readmode = 0 只读
            readmode = 1 读写
        '''
        # 使用ogr特定异常
        ogr.UseExceptions()
        # 获取驱动
        driver = ogr.GetDriverByName("FileGDB")
        # 打开gdb文件
        try:
            self.gdb = driver.Open(gdb_path, readmode)
        except Exception as e:
            print(e)
            sys.exit()
        print(self.gdb)
        # 获取图层个数
        self.iLayerCount = self.gdb.GetLayerCount()
        print("Layer number = ", self.iLayerCount)

    # 读gdb文件
    def read_layer(self, layer_name):
        # 根据名称获取图层
        oLayer = self.gdb.GetLayerByName(layer_name)
        if oLayer == None:
            print("Get layer failed !")
            sys.exit()
        # 对图层进行初始化
        oLayer.ResetReading()
        return oLayer

    def delete_feature(self, layer_name, delete_list):
        '''
            Input: 
                layer_num: 要删除对象的图层名称
                delete_list: [['feature_att', number, operation], ....]}  
                    list中分别是要判断对象的属性名，number/str是判断的条件值，operation是判断运算符（> < == !=）,且多个条件取或，即满足一个就删除
                    例如 [[code, 0, ==]]}表示删除满足code字段为0的对象
        '''
        oLayer = self.read_layer(layer_name)

        # 输出图层中的要素个数
        num = oLayer.GetFeatureCount(0)
        print("Feature number = ", num)
        
        # 读取矢量文件的字段
        V = Vector()
        _, _, Fields = V.Get_attribute_table(oLayer)
        F_k = list(Fields.keys())
        print(F_k)
        # gdb的layer中feature编号从1开始
        for i in range(1, num+1):
            # ofeature = oLayer.GetFeature(i)
            ofeature = oLayer.GetNextFeature()
            if ofeature == None:
                continue
            if i%1000000 == 0:
                print(i)
            det_flag = False
            for det_item in delete_list:
                name, para, opr = det_item[0], str(det_item[1]), det_item[2]
                value = str(ofeature.GetFieldAsString(name))
                FIDs = ofeature.GetFieldAsString(F_k[0])
                if opr == '==':
                    if value == para:
                        det_flag = True
                        break
                elif opr == '!=':
                    if value != para:
                        det_flag = True
                        break
                elif opr == '<=':
                    if float(value) <= float(para):
                        det_flag = True
                        break
                elif opr == '>=':
                    if float(value) >= float(para):
                        det_flag = True
                        break
                else:
                    print('Unknown operation')
                    return

            if det_flag == True:
                oLayer.DeleteFeature(int(FIDs))
        return
# def getGdbLayerList(gdb_path):
#     # 使用ogr特定异常
#     ogr.UseExceptions()
#     # 获取驱动
#     driver = ogr.GetDriverByName("OpenFileGDB")
#    # 打开gdb文件
#     try:
#         gdb = driver.Open(gdb_path, 0)
#     except Exception as e:
#         print(e)
#         sys.exit()
#     # 存储图层名称的列表
#     layerList = []
#     # 获取图层名称
#     for index in range(gdb.GetLayerCount()):
#         layer = gdb.GetLayerByIndex(index)
#         layerList.append(layer.GetName())
#     # 清除文件
#     del gdb
#     return layerList


if __name__ == '__main__':
    # readGdb('F:\\新建文件夹\\我的数据\\2\\xzy2.gdb', 'RasterT_tif1')
    delete_key = [['gridcode', 0, '=='], ['gridcode', 2, '=='], ['Shape_Area', 1000, '<=']]
    aa = GDB_XZY('F:\\新建文件夹\\我的数据\\3\\xzy3_new.gdb', 1)
    aa.delete_feature('RasterT_tif1', delete_key)
    from osgeo import ogr

    # gdb_driver=ogr.GetDriverByName("FileGDB")
    # gdb = gdb_driver.Open('F:\\新建文件夹\\我的数据\\2\\xzy2.gdb', 0)
    # 使用ogr特定异常
    # ogr.UseExceptions()
    # # 获取驱动
    # driver = ogr.GetDriverByName("FileGDB")
    # # 打开gdb文件
    # try:
    #     gdb = driver.Open('F:\\新建文件夹\\我的数据\\2\\xzy2.gdb', 0)
    # except Exception as e:
    #     print(e)
    #     sys.exit()
    # print(gdb)