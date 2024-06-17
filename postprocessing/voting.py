import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

from data_processing.Raster import *
from collections import Counter
def voting(imgpath_list):
    imglist = []
    for imgpath in imgpath_list:
        img, _ = gdal_read_tif(imgpath)
        img = img[None,:,:]
        imglist.append(img)
    img_stack = np.concatenate([img for img in imglist], 0)
    imgout = np.zeros_like(img)[0]
    for i in range(img_stack.shape[1]):
        for j in range(img_stack.shape[2]):
            count = Counter(img_stack[:,i,j])
            # print('1,',count, count.most_common(1))
            mostclass = count.most_common(1)[0][0]
            if count[mostclass] != 1:
                imgout[i][j] = mostclass
            else:
                imgout[i][j] = img_stack[1][i][j]
            # print(count, count.most_common(1)[0][0])
            # break
    return imgout

if __name__ == '__main__':
    base_folder = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\Spatial_Spectral\\9_12\\'
    imgpath_list = [base_folder+'pred_seg100_pre_s1.png',base_folder+'pred_seg100_pre_s2.png',base_folder+'pred_seg100_pre.png']
    imgout = voting(imgpath_list)
    gdal_write_tif(base_folder+'vote.png', imgout, imgout.shape[0], imgout.shape[1])