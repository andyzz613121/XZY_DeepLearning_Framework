import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from data_processing.Raster import gdal_read_tif, gdal_write_tif
from data_processing.excel import read_excel, get_total_item, write_excel
from data_processing.Vector import Vector

'''
    Step1: 读取数据统计ColorMap里面各类别的像素数
'''
# img_path = 'E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\5. clip\\5. clip.tif'
# img, para = gdal_read_tif(img_path)

# ColorMap = read_excel('E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\6. merge\\1. CDL_ColorMap.xlsx')
# ColorMap = get_total_item(ColorMap)[1:]

# merge_list = []
# ColorMap_Num = []

# for item in ColorMap:
#     value = int(item[0])
#     idx = (img == value)
#     number = idx.sum()
#     ColorMap_Num.append([value, number, item[4]])
# write_excel('E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\6. merge\\2. CDL_ColorMap_sta.xls', ColorMap_Num)
# print(ColorMap_Num, len(ColorMap_Num))

'''
    Step2: 读取CDL数据转矢量后, 矢量里面各类别的图斑数
'''
vect_sta = {}
vect = Vector('E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\6. merge\\CDL_shp\\cdl_shp.shp')
feats = vect.Get_feature_total(vect.Layer)
for feat in feats:
    area = vect.get_field_value(feat, '面积')
    category = vect.get_field_value(feat, 'Code')
    if category not in vect_sta.keys():
        tmp_list = [1, area]
        vect_sta[category] = tmp_list
    else:
        vect_sta[category][0] += 1
        vect_sta[category][1] += area

out_list = []
for key in vect_sta.keys():
    out_list.append([key, vect_sta[key][0], vect_sta[key][1]])
write_excel('E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\6. merge\\3. CDL_vect_sta.xls', out_list)


'''
    Step2: 合并类别
'''
# img_path = 'E:\\dataset\\毕设数据\\label\\4. CDL_CLIP_FINAL.tif'
# img, para = gdal_read_tif(img_path)
# class_list = [121, 111, 61, 1, 54, 76, 75, 24, 69, 36, 142, 176, 152]
# idx0 = (img == 122)
# idx1 = (img == 123)
# idx2 = (img == 124)
# idx3 = (img == 37)
# img[idx0] = 121
# img[idx1] = 121
# img[idx2] = 121
# img[idx3] = 36

# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         if img[i][j] not in class_list:
#             start_x, end_x = i-3, i+3
#             start_y, end_y = j-3, j+3
#             if start_x < 0 or end_x >= img.shape[0]:
#                 start_x, end_x = 0, img.shape[1]-1
#             if start_y < 0 or end_y >= img.shape[0]:
#                 start_y, end_y = 0, img.shape[1]-1
            
#             find = 0
#             for tmp_i in range(start_x, end_x):
#                 if find == 1:
#                     break
#                 for tmp_j in range(start_y, end_y):
#                     if img[tmp_i][tmp_j] in class_list:
#                         img[i][j] = img[tmp_i][tmp_j]
#                         find = 1
#                         break
                    
# gdal_write_tif('E:\\dataset\\毕设数据\\label\\5. CDL_Class_Merge.tif', img, para[0], para[1], para[2], para[3], para[4])

            