from osgeo import gdal
import numpy as np

def output_maxmin(imgpath_list, maxmin_path):
    '''
        imgpath_list: 所有输入图像的路径list
        maxmin_path: 所有图像的最大值于最小值, 用ImgX间隔表示
    '''
    with open(maxmin_path,'w') as file:
        for idx in range(len(imgpath_list)):
            img_path = imgpath_list[idx]
            img_raw = gdal.Open(img_path)
            img_w = img_raw.RasterXSize
            img_h = img_raw.RasterYSize
            img = np.array(img_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
            tmpstr = 'Img' + str(idx) + '\n'
            file.write(tmpstr)
            for b in range(img.shape[0]):
                min, max = np.min(img[b]), np.max(img[b])
                str1 = str(min) + ',' + str(max) + '\n'
                file.write(str1)

if __name__ == '__main__':
    
    img_path1 = 'E:\\dataset\\高光谱数据集\\Salinas\\big_image.tif'
    img_path2 = 'E:\\dataset\\毕设数据\\new\\2. MS\\Imgs\\9_12.tif'
    max_file = 'E:\\dataset\\毕设数据\\new\\2. MS\\Imgs\\9_12.txt'
    imgpath_list = [img_path1, img_path2]
    output_maxmin(imgpath_list, max_file)


