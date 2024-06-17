import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def spam11(img):
    '''
        Input: img[B, C, H, W]
        Output: spam_img[B, C, H, W]
    '''

    b, c, h, w = img.size()

    img_cp = torch.zeros_like(img)

    spam11_conv = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]).unsqueeze(0).unsqueeze(0).float().repeat(b, 1, 1, 1)
    spam11_conv = spam11_conv.cuda()
    for channel in range(c):
        img_cp[:, channel, :, :] = F.conv2d(img[:, channel, :, :].unsqueeze(1), spam11_conv, padding=2)[:, 0, :, :] + 128
    return img_cp

def minmax41(img):
    '''
        Input: img[B, C, H, W]
        Output: minmax41_img[B, 2C, H, W] -> 2C(min1, min2, ..., min C, max1, max2, ..., max C)
    '''
    b, c, h, w = img.size()

    left_conv = torch.tensor([[-1, 2, 0], [2, -4, 0], [-1, 2, 0]]).unsqueeze(0).unsqueeze(0).float().repeat(b, 1, 1, 1)
    right_conv = torch.tensor([[0, 2, -1], [0, -4, 2], [0, 2, -1]]).unsqueeze(0).unsqueeze(0).float().repeat(b, 1, 1, 1)
    top_conv = torch.tensor([[-1, 2, -1], [2, -4, 2], [0, 0, 0]]).unsqueeze(0).unsqueeze(0).float().repeat(b, 1, 1, 1)
    bottom_conv = torch.tensor([[0, 0, 0], [2, -4, 2], [-1, 2, -1]]).unsqueeze(0).unsqueeze(0).float().repeat(b, 1, 1, 1)

    left_conv, right_conv, top_conv, bottom_conv = left_conv.cuda(), right_conv.cuda(), top_conv.cuda(), bottom_conv.cuda()

    img_min, img_max = torch.zeros_like(img), torch.zeros_like(img)

    for channel in range(c):
        img_left = F.conv2d(img[:, channel, :, :].unsqueeze(1), left_conv, padding=1)
        img_right = F.conv2d(img[:, channel, :, :].unsqueeze(1), right_conv, padding=1)
        img_top = F.conv2d(img[:, channel, :, :].unsqueeze(1), top_conv, padding=1)
        img_bottom = F.conv2d(img[:, channel, :, :].unsqueeze(1), bottom_conv, padding=1)
        img_stack = torch.cat([img_left, img_right, img_top, img_bottom], 1)

        img_min_c, _ = torch.min(img_stack, 1)
        img_max_c, _ = torch.max(img_stack, 1)

        img_min[:, channel, :, :] = img_min_c + 128
        img_max[:, channel, :, :] = img_max_c + 128

    img_maxmin = torch.cat([img_min, img_max], 1)

    return img_maxmin

def srmfilter(img):
    '''
        Input: img[B, C, H, W]
        Output: minmax41_img[B, 2C, H, W] -> 2C(min1, min2, ..., min C, max1, max2, ..., max C)
    '''
    conv1 = np.array([[0,0,0,0,0],[0,-0.25,0.5,-0.25,0],[0,0.5,-1,0.5,0],[0,-0.25,0.5,-0.25,0],[0,0,0,0,0]])
    conv2 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]])
    conv2 = conv2/12
    conv3 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0.5,-1,0.5,0],[0,0,0,0,0],[0,0,0,0,0]])

    b, c, _, _ = img.size()

    conv11 = np.repeat(conv1[np.newaxis,:], c, 0)
    conv22 = np.repeat(conv2[np.newaxis,:], c, 0)
    conv33 = np.repeat(conv3[np.newaxis,:], c, 0)

    conv11 = np.repeat(conv11[np.newaxis,:], b, 0)
    conv22 = np.repeat(conv22[np.newaxis,:], b, 0)
    conv33 = np.repeat(conv33[np.newaxis,:], b, 0)
    
    weight1 = torch.from_numpy(conv11).cuda().float()
    weight2 = torch.from_numpy(conv22).cuda().float()
    weight3 = torch.from_numpy(conv33).cuda().float()

    img = img.float()
    out1 = F.conv2d(img, weight1, padding=2)
    out2 = F.conv2d(img, weight2, padding=2)
    out3 = F.conv2d(img, weight3, padding=2)
    
    return torch.cat([out1, out2, out3], 1)

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    import os
    import os
    import sys
    import time
    

    import torch
    import torch.nn as nn
    from torch.utils import data
    import torchvision.transforms as transforms

    base_path = '..\\XZY_DeepLearning_Framework\\'
    sys.path.append(base_path)

    from data_processing.Raster import gdal_write_tif, gdal_read_tif
    for r in [1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5]:
        base = 'E:\\dataset\\ImageBlur\\Data\\train\\实验图像_(复杂度+模糊半径)\\选择的实验图像\\全部\\r' + str(r) + '\\Middle IC\\img_rgb\\'
        out = 'E:\\dataset\\ImageBlur\\Data\\train\\实验图像_(复杂度+模糊半径)\\选择的实验图像\\全部\\r' + str(r) + '\\Middle IC\\minspam\\'
        if os.path.exists(out) == False:
            os.makedirs(out)

        for item in os.listdir(base):
            imgfile = base + item
            outfile = out + item
            imgfile = 'E:\\dataset\\ImageBlur\\Data\\train\\训练图像\\train\\img\\15.png'
            outfile = 'D:\\Papers\\投稿\\【撰写中】Automatic detection of blurred areas for remote sensing image\\图片\\流程图\\15.png'
            img, _ = gdal_read_tif(imgfile)
            # img = np.array(img).astype(np.float32)
            # img = np.swapaxes(img, 0, 2)
            # img = img[np.newaxis,:]
            img = torch.from_numpy(img).cuda().unsqueeze(0).float()
            # print(img.shape)
            img = (img[:, 0, :, :]*0.84 + img[:, 1, :, :]*0.11 + img[:, 2, :, :]*0.05).unsqueeze(1)
            # aa = srmfilter(img)[0]
            minmax_img = minmax41(img)[0]
            spam11_img = spam11(img)[0]

            srm = torch.cat([minmax_img, spam11_img], 0).cpu().numpy()     # 3, 256, 256
            # srm = aa.cpu().numpy()     # 3, 256, 256

            gdal_write_tif(outfile, srm, 256, 256, 3, datatype=2)
            # out_minmax = np.swapaxes(minmax_img, 0, 2).astype(np.uint8)

            # # out1 = Image.fromarray(out[:,:,0])
            # # out1.save('C:\\Users\\25321\\Desktop\\11.tif')
            # # out2 = Image.fromarray(out[:,:,1])
            # # out2.save('C:\\Users\\25321\\Desktop\\22.tif')

            # spam11_img = spam11(img).cpu().numpy()[0]
            # out_spam11 = np.swapaxes(spam11_img, 0, 2).astype(np.uint8)
            # print(out_spam11)
            # out_spam11 = Image.fromarray(out_spam11[:,:,0])
            # out_spam11.save('C:\\Users\\25321\\Desktop\\33.tif')

            # img = np.swapaxes(img[0][0], 0, 1)
            # img = Image.fromarray(img.cpu().numpy())
            # img.save('C:\\Users\\25321\\Desktop\\44.tif')

            break