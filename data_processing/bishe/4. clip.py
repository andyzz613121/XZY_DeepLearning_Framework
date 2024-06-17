import os
import sys
import numpy as np
from osgeo import gdal

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from data_processing.Raster import *

class Clip():
    def open_img(self, img_path):
        img = gdal.Open(img_path)
        img_w = img.RasterXSize
        img_h = img.RasterYSize
        img = np.array(img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h))

        if len(img.shape) == 2:
            img = img[np.newaxis, :]

        print('Read Img Done!')
        return img, img_w, img_h

    def write_tif(self, filename, img, img_w, img_h, GeoTransform=None, Spatial_Ref=None, datatype=gdal.GDT_UInt16):#img:[c,h,w]
        driver = gdal.GetDriverByName("GTiff")
        bands = img.shape[0]

        dataset = driver.Create(filename, img_w, img_h, bands, datatype)
        if GeoTransform is not None:
            dataset.SetGeoTransform(GeoTransform)
        if Spatial_Ref is not None:
            dataset.SetSpatialRef(Spatial_Ref)
        
        for b in range(img.shape[0]):
            dataset.GetRasterBand(1+b).WriteArray(img[b])
        del dataset

    # 判断图像中是否存在背景值
    def exist_bkvalue(self, img, bk_value, bk_rate):
        assert len(bk_value) == img.shape[0], 'bk_value not match'
        # 设置一个全为1的图像
        exist_flag = np.ones([img.shape[1], img.shape[2]])
        total_num = exist_flag.sum()

        for b in range(img.shape[0]):
            img_b = img[b]
            bkvalue_b = bk_value[b]
            # 这样相乘，是背景值的地方还是1，不是背景值的地方就是0
            pos_index = (img_b == bkvalue_b)
            exist_flag *= pos_index
        
        bk_num =  exist_flag.sum()
        # print(total_num, bk_num)
        if bk_num/total_num > bk_rate:
            return True
        else:
            return False

    # 判断图像中是否全都是背景值
    def all_bkvalue(self, img, bk_value):
        assert len(bk_value) == img.shape[0], 'bk_value not match'
        all_bkFlag = True
        for b in range(img.shape[0]):
            img_b = img[b]
            bkvalue_b = bk_value[b]
            if np.max(img_b) != bkvalue_b or np.min(img_b) != bkvalue_b:
                all_bkFlag = False
        return all_bkFlag
        
    def clip_singleIMG(self, img_path, out_folder, step, overlap=0, bk_value:list=None, bk_rate=0.2, datatype=gdal.GDT_UInt16, start_imgnum=0):
        '''
        Paras
            img_path: 图像1路径(如遥感影像)
            out_folder: 图像1裁剪后输出路径
            step: 裁剪步长
            overlap=0: 重叠
            bk_value:list=None: 图像要剔除的背景值(用list表示每个波段的值)
            bk_rate=None: 背景值超出多少比例的图像要被剔除(0:存在就剔除, 1:全部是背景值才剔除, else:满足一定比例后剔除)
            datatype=gdal.GDT_UInt16: 输出图像数据类型
            start_imgnum=0: 从第几号开始编号（是否连续计算图像数目, 即图像编号连续, 用于多张图像裁剪的时候）
        '''

        img, para = gdal_read_tif(img_path)
        GeoTransform_old = para[3]
        GeoTransform_new = list(para[3])
        Spatial_Ref = para[4]

        start_x = 0
        start_y = 0
        for start_x in range(0, img.shape[1], step-overlap):
            for start_y in range(0, img.shape[2], step-overlap):
                
                # 计算图像块起始终止位置
                end_x = step + start_x
                end_y = step + start_y
                
                if GeoTransform_old != None:
                    GeoTransform_new[0], GeoTransform_new[3] = xy2geo(GeoTransform_old, start_x, start_y)

                if (step + start_x) >= img.shape[1]:
                    end_x = img.shape[1] - 1
                    start_x = end_x - step
                if (step + start_y) >= img.shape[2]:
                    end_y = img.shape[2] - 1
                    start_y = end_y - step
                
                # 图像块裁剪
                img_outname = out_folder + str(start_imgnum) + '.tif'
                img_split = img[:,start_x:end_x,start_y:end_y]

                # 忽略背景值
                if bk_value != None:
                    if self.exist_bkvalue(img_split, bk_value, bk_rate) == True:
                        continue

                # 图像块存储
                # gdal_write_tif(img_outname, img_split, step, step, img_split.shape[0], tuple(GeoTransform_new), Spatial_Ref, datatype=datatype)
                gdal_write_tif(img_outname, img_split, step, step, img_split.shape[0], datatype=datatype)
                start_imgnum += 1

    def clip_coupleIMG(self, img1_path, img2_path, out_folder, step, overlap=0, bk_value1:list=None, bk_value2:list=None, bk_rate=0.2, datatype=gdal.GDT_UInt16, start_imgnum=0): 
        '''
        Paras
            img1_path: 图像1路径(如遥感影像)
            img2_path: 图像2路径(如标签数据)
            out_folder: 图像裁剪后输出文件夹（将自动构建img和lab文件夹）
            step: 裁剪步长
            overlap=0: 重叠
            bk_value1:list=None: 图像1要剔除的背景值(用list表示每个波段的值)
            bk_value2:list=None: 图像2要剔除的背景值(用list表示每个波段的值)
            bk_model=None: 剔除背景值图像的模式(1:存在就剔除, 2:全部是背景值才剔除)
            datatype=gdal.GDT_UInt16: 输出图像数据类型
            GDAL_DATATYPE:
                    {
                        1: gdal.GDT_Byte,
                        2: gdal.GDT_Float32
                        3: gdal.GDT_UInt16
                    }
            start_imgnum=0: 从第几号开始编号（是否连续计算图像数目, 即图像编号连续, 用于多张图像裁剪的时候）

            注意：裁剪时会将Img1的最大值输入到每个裁剪的Img1子图像中，Img2因为是标签，所以不做这样的处理
        '''
        # /////////////////////////////////////////////////////////////
        # 判断文件和文件夹是否存在
        if os.path.exists(img1_path)==False:
            print('ERROR: img1_path not exist')
            return
        if os.path.exists(img2_path)==False:
            print('ERROR: img2_path not exist')
            return
        if os.path.exists(out_folder)==False:
            print('ERROR: outfolder not exist')
            return
        # 判断图像对大小是否一致
        img1, [h1, w1, _, _, _, img1_gdal] = gdal_read_tif(img1_path)
        img2, [h2, w2, _, _, _, _] = gdal_read_tif(img2_path)
        if len(img1.shape) == 2:
            img1 = img1[np.newaxis, :]
        if len(img2.shape) == 2:
            img2 = img2[np.newaxis, :]

        if w1!=w2 or h1!=h2:
            print('Img size not match')
            return
        # /////////////////////////////////////////////////////////////
        # 新建输出文件夹及输出文件
        outimg_folder = out_folder + 'img\\'
        outlab_folder = out_folder + 'lab\\'
        if os.path.exists(outimg_folder)==False:
            os.mkdir(outimg_folder)
        if os.path.exists(outlab_folder)==False:
            os.mkdir(outlab_folder)

        csv_name = out_folder + 'train.csv'
        csv_file = open(csv_name, 'w')

        # # /////////////////////////////////////////////////////////////
        # 读写最大值最小值
        max_list = get_metadata(img1_gdal, 'max')
        min_list = get_metadata(img1_gdal, 'min')
        if max_list and min_list:
            max_list = list(map(int, max_list.strip().split()))
            min_list = list(map(int, min_list.strip().split()))
            print('MaxMin: Using the original maxmin in Metadata of (%s)'%img1_path)
        else:
            max_list = []
            min_list = []
            for b in range(img1.shape[0]):
                min, max = np.min(img1[b]), np.max(img1[b])
                max_list.append(max)
                min_list.append(min)
            print('MaxMin: Computing the maxmin of (%s)'%img1_path)
        add_metadata = {'max':max_list, 'min':min_list}
        # /////////////////////////////////////////////////////////////
        # 开始裁剪
        start_x = 0
        start_y = 0
        for start_x in range(0, img1.shape[1], step-overlap):
            for start_y in range(0, img1.shape[2], step-overlap):
                # 计算图像块起始终止位置
                end_x = step + start_x
                end_y = step + start_y
                if (step + start_x) >= img1.shape[1]:
                    end_x = img1.shape[1] - 1
                    start_x = end_x - step
                if (step + start_y) >= img1.shape[2]:
                    end_y = img1.shape[2] - 1
                    start_y = end_y - step
                
                
                img1_outname = outimg_folder + str(start_imgnum) + '.tif'
                img2_outname = outlab_folder + str(start_imgnum) + '.tif'

                # 图像块裁剪
                img1_split = img1[:,start_x:end_x,start_y:end_y]
                img2_split = img2[:,start_x:end_x,start_y:end_y]
                # 忽略背景值
                if bk_value1 != None:
                    if self.exist_bkvalue(img1_split, bk_value1, bk_rate) == True:
                        continue

                if bk_value2 != None:
                    if self.exist_bkvalue(img2_split, bk_value2, bk_rate) == True:
                        continue
                
                # 图像块存储
                gdal_write_tif(img1_outname, img1_split, step, step, img1_split.shape[0], datatype=datatype, add_metadata=add_metadata)
                gdal_write_tif(img2_outname, img2_split, step, step, img2_split.shape[0], datatype=1)
                # self.write_tif(img1_outname, img1_split, step, step, datatype=datatype)
                # self.write_tif(img2_outname, img2_split, step, step, datatype=datatype)
                str1 = str(img1_outname) + ',' + str(img2_outname) + '\n'               
                csv_file.write(str1)
                start_imgnum += 1

        csv_file.close()
    
    def clip_HSIMG(self, img_path, lab_path, out_folder, half_window_size=5):
        '''
            针对高光谱图像分类的裁剪方式：输入样本点数据（lab_path），对存在样本点的像素以其为窗口中心进行裁剪，并输出csv文件（一张图对应一个值）
        '''
        # /////////////////////////////////////////////////////////////
        # 判断文件和文件夹是否存在
        if os.path.exists(img_path)==False:
            print('ERROR: img_path not exist')
            return
        if os.path.exists(lab_path)==False:
            print('ERROR: lab_path not exist')
            return
        if os.path.exists(out_folder)==False:
            print('ERROR: outfolder not exist')
            return
        # 判断图像对大小是否一致
        img, w1, h1 = self.open_img(img_path)
        lab, w2, h2 = self.open_img(lab_path)
        if w1!=w2 or h1!=h2:
            print('Img size not match')
            return
        # /////////////////////////////////////////////////////////////
        # 新建输出文件夹及输出文件
        outimg_folder = out_folder + 'img\\'
        if os.path.exists(outimg_folder)==False:
            os.mkdir(outimg_folder)

        csv_name = out_folder + 'train.csv'
        train_csv = open(csv_name, 'w')
        # /////////////////////////////////////////////////////////////
        # 开始裁剪

        line_num = 0
        print(lab.shape, img.shape)
        for x in range(lab.shape[0]):
            for y in range(lab.shape[1]):
                if lab[x, y] != 0:
                    start_x = x - half_window_size
                    start_y = y - half_window_size
                    end_x = x + half_window_size + 1
                    end_y = y + half_window_size + 1
                    
                    if start_x < 0:
                        start_x = 0
                        end_x = 2*half_window_size + 1
                    
                    if start_y < 0:
                        start_y = 0
                        end_y = start_y + 2*half_window_size + 1

                    if end_y >= label.shape[1]:
                        end_y = label.shape[1] - 1
                        start_y = end_y - 2*half_window_size - 1

                    if end_x >= label.shape[0]:
                        end_x = label.shape[0] - 1
                        start_x = end_x - 2*half_window_size - 1

                    assert (end_x - start_x) ==  (2*half_window_size + 1)
                    assert (end_y - start_y) ==  (2*half_window_size + 1)
                    
                    img_clip = img[:, start_x:end_x, start_y:end_y]
                    print(img_clip.shape)
                    outname = out_folder + str(line_num) + '.tif'
                    gdal_write_tif(outname, img_clip, img_clip.shape[1], img_clip.shape[2], img_clip.shape[0], datatype=2)

                    str1 = str(outname) + ',' + str(label[x, y]-1) + '\n'           #没有第0类背景类，所有类往前进一个
                    # print(str1, start_x, end_x, start_y, end_y)
                    train_csv.write(str1)

                    line_num+=1

if __name__ == '__main__':
    clip = Clip()
    img1_path = 'E:\\dataset\\毕设数据\\new\\2. MS\\SS_month\\img_train.tif'
    img2_path = 'E:\\dataset\\毕设数据\\new\\2. MS\\SS_month\\lab_train.tif'
    out_folder = 'E:\\dataset\\毕设数据\\new\\2. MS\\SS_month\\'
    clip.clip_coupleIMG(img1_path, img2_path, out_folder, step=256)

