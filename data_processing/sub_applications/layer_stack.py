import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from data_processing.Raster import gdal_read_tif, gdal_write_tif
import numpy as np
def layer_stack(name_list, out_path):
    img_list = []
    for name in name_list:
        img, para = gdal_read_tif(name)
        if len(img.shape) == 2:
            img = img[np.newaxis,:,:]
        img_list.append(img)
    imgs = np.concatenate([np.array(x) for x in img_list])
    print(imgs.shape)
    gdal_write_tif(out_path, imgs, para[0], para[1], len(name_list), para[3], para[4], datatype=3)
    return imgs

if __name__ == '__main__':
    import os
    # folder = 'E:\\dataset\\test\\旧金山哨兵影像L2A\\S2B_MSIL2A_20220320T184959_N0400_R113_T10SFH_20220320T231722.SAFE\\GRANULE\\L2A_T10SFH_A026308_20220320T190033\\IMG_DATA\\R10m\\'
    # name_list = []
    # for item in os.listdir(folder):
    #     if 'B02' in item or 'B03' in item or 'B04' in item or 'B08' in item:
    #         name_list.append(folder + item)
    # imgs = layer_stack(name_list, folder + 'LayerStack_20220320.tif')

    img = layer_stack(['E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\7. labels_with_months\\4&5. Apr_May.tif', 
                       'E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\7. labels_with_months\\6. June.tif',
                       'E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\7. labels_with_months\\7. July(7.14).tif', 
                       'E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\7. labels_with_months\\8. August.tif', 
                       'E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\7. labels_with_months\\9. Sep.tif', 
                       'E:\\dataset\\毕设数据\\new\\2. MS\\Labels\\7. labels_with_months\\10. Oct(10.17).tif', ], 
                      'E:\\dataset\\毕设数据\\new\\2. MS\\SS_month\\SS_layerstack.tif')
    