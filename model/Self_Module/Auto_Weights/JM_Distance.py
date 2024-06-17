import numpy as np
from numpy.lib.function_base import cov
import torch
from torch import nn
import torch.nn.functional as F
from torch.linalg import det

def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True, dtype = torch.float64)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()
 
def cof1(M,index):
    zs = M[:index[0]-1,:index[1]-1]
    ys = M[:index[0]-1,index[1]:]
    zx = M[index[0]:,:index[1]-1]
    yx = M[index[0]:,index[1]:]
    s = torch.cat((zs,ys),axis=1)
    x = torch.cat((zx,yx),axis=1)
    return det(torch.cat((s,x),axis=0))
 
def alcof(M,index):
    return pow(-1,index[0]+index[1])*cof1(M,index)
 
def adj(M):
    result = torch.zeros((M.shape[0],M.shape[1])).cuda()
    for i in range(1,M.shape[0]+1):
        for j in range(1,M.shape[1]+1):
            result[j-1][i-1] = alcof(M,[i,j])
    return result
 
def invmat(M):
    return 1.0/det(M)*adj(M)
 
def JMD_np(c1_vector, c2_vector, feature_dim):
    c1_vector = np.reshape(c1_vector,[feature_dim, -1])
    c2_vector = np.reshape(c2_vector,[feature_dim, -1])

    c1_mean = np.mean(c1_vector, 1, dtype = np.float64)
    c2_mean = np.mean(c2_vector, 1, dtype = np.float64)
    mean12 = c1_mean-c2_mean
    print(c1_mean, c2_mean, mean12)
    if feature_dim > 1:
        c1_cov = np.cov(c1_vector)
        c2_cov = np.cov(c2_vector)
    else:
        c1_cov = np.var(c1_vector)
        c2_cov = np.var(c2_vector)
        c1_cov = np.reshape(c1_cov, [1,1])
        c2_cov = np.reshape(c2_cov, [1,1])

    cov12 = (c1_cov+c2_cov)/2
    cov12_inv = np.linalg.inv(cov12)

    a1 = (1/8) * np.core.dot(mean12.T, cov12_inv)
    a1 = np.core.dot(a1, mean12)
    # print(a1)
    a2 = (1/2) * np.log(np.linalg.det(cov12)/np.sqrt(np.linalg.det(c1_cov)*np.linalg.det(c2_cov)))
    a = a1 + a2
    JM = 2*(1-np.exp(-1*a))
    print('JMD_np', JM)
    
    return JM

def JMD_torch(c1_vector, c2_vector, feature_dim):
    c1_vector = torch.reshape(c1_vector,[feature_dim, -1])
    c2_vector = torch.reshape(c2_vector,[feature_dim, -1])
    # c1_vector = (c1_vector-torch.min(c1_vector))/(torch.max(c1_vector)-torch.min(c1_vector))
    # c2_vector = (c2_vector-torch.min(c2_vector))/(torch.max(c2_vector)-torch.min(c2_vector))

    c1_mean = torch.mean(c1_vector, 1, dtype = torch.float64)
    c2_mean = torch.mean(c2_vector, 1, dtype = torch.float64)
    mean12 = (c1_mean-c2_mean).float()
    # print(c1_mean, c2_mean, mean12)
    if feature_dim > 1:
        c1_cov = cov(c1_vector)
        c2_cov = cov(c2_vector)
    else:
        c1_cov = torch.var(c1_vector)
        c2_cov = torch.var(c2_vector)

    c1_cov = torch.reshape(c1_cov, [feature_dim,feature_dim])
    c2_cov = torch.reshape(c2_cov, [feature_dim,feature_dim])
    mean12 = torch.reshape(mean12, [feature_dim,1])
    cov12 = (c1_cov+c2_cov)/2

    if det(cov12)==0:
        print('The matrix is singular')
        # print(cov12)
        return False
    print('The matrix is not singular')
    # print(mean12, cov12)
    cov12_solve = torch.linalg.solve(cov12.float(), mean12)
    a1 = (1/8) * torch.mm(mean12.t(), cov12_solve)
    a2 = (1/2) * torch.log(det(cov12)/torch.sqrt(det(c1_cov)*det(c2_cov)))
    a = torch.log(a1 + a2)
    print(a1 + a2, a, torch.exp(-1*a))
    JM = 2*(1-torch.exp(-1*a))
    print('JMD_torch', JM)
    return JM

def JMD_1D(c1_vector, c2_vector, feature_dim):
    c1_vector = np.reshape(c1_vector,[feature_dim, -1])
    c2_vector = np.reshape(c2_vector,[feature_dim, -1])

    c1_mean = np.mean(c1_vector, 1)
    c2_mean = np.mean(c2_vector, 1)
    
    c1_cov = np.var(c1_vector)
    c2_cov = np.var(c2_vector)
    mean12 = c1_mean-c2_mean
    cov12 = (c1_cov+c2_cov)/2
    cov12_inv = 1/cov12
    # print('JMD_1D', mean12, c1_cov, c2_cov)

    a1 = (1/8) * np.core.dot(mean12.T, cov12_inv)
    a1 = np.core.dot(a1, mean12)
    # print(a1)
    a2 = (1/2) * np.log(cov12/np.sqrt(c1_cov*c2_cov))
    a = a1 + a2
    JM = 2*(1-np.exp(-1*a))
    print('JMD_1D', JM)
    return

def Simple_JMD(c1_vector, c2_vector, feature_dim):
    c1_vector = torch.reshape(c1_vector,[feature_dim, -1])
    c2_vector = torch.reshape(c2_vector,[feature_dim, -1])
    # print(c1_vector)
    c1_vector = (c1_vector-torch.min(c1_vector))/(torch.max(c1_vector)-torch.min(c1_vector))
    c2_vector = (c2_vector-torch.min(c2_vector))/(torch.max(c2_vector)-torch.min(c2_vector))

    # torch.set_printoptions(profile="full")
    c1_mean = torch.mean(c1_vector, 1, dtype = torch.float64)
    c2_mean = torch.mean(c2_vector, 1, dtype = torch.float64)
    mean12 = (c1_mean-c2_mean).float()

    c1_var = torch.var(c1_vector)
    c2_var = torch.var(c2_vector)
    
    a = mean12 * mean12 / (c1_var+c2_var)
    print(a, mean12*mean12, c1_var+c2_var)
    JMD = 2*(1-torch.exp(-1*a))
    # print('Simple_JDM', JMD)
    print(JMD)
    return JMD

class auto_weight(nn.Module):
    def __init__(self, input_channels):
        super(auto_weight, self).__init__()
        self.auto_weight = nn.Parameter(torch.ones([input_channels], requires_grad=True))
        # self.auto_weight = nn.Parameter(torch.rand([input_channels], requires_grad=True))
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(x, self.auto_weight)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

def JMD_Loss(featuremap, label, c1):
    '''
    input: featuremap [Batch, Channel, :, :], label [Batch, :, :]
    '''
    featuremap = F.interpolate(featuremap, size=(label.shape[1], label.shape[2]), mode='bilinear')
    featuremap = featuremap.permute(1, 0, 2, 3).contiguous()# featuremap[Channel, Batch, :, :], label[Batch, :, :]

    print(featuremap.shape)

    JMD_Sum = 0
    feature_dim = featuremap.shape[0]
    
    c1_index = (label == c1)
    if c1_index.sum()==0:
        return False
    c1_pixel = featuremap[:,c1_index]
    print(torch.max(c1_pixel),torch.min(c1_pixel))
    for c2 in range(5):
        if c2 != c1:
            c2_index = (label == c2)
            if c2_index.sum()==0:
                continue
            c2_pixel = featuremap[:,c2_index]
            JMD = JMD_torch(c1_pixel, c2_pixel, feature_dim)
            if JMD == False:
                continue
            
            # from osgeo import gdal
            # aa = featuremap.cpu().detach().numpy()
            # driver = gdal.GetDriverByName('GTiff')
            # dataset = driver.Create("C:\\Users\\25321\\Desktop\\2.tif", 256, 256, 64, gdal.GDT_Float32)
            # for i in range(64):
            #     dataset.GetRasterBand(1+i).WriteArray(aa[i][0])
            # del dataset

            JMD_Sum = JMD_Sum + (2-JMD)
    print('JMD_Sum', JMD_Sum)
    return JMD_Sum

if __name__ == "__main__":
    import cv2
    from osgeo import gdal

    img = gdal.Open('C:\\Users\\25321\\Desktop\\1.bmp')
    img_w = img.RasterXSize
    img_h = img.RasterYSize
    img = np.array(img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
    # img = img[:,:,2]
    # print(img.shape)
    # img[266:398,26:157] = 255
    # img[56:179,283:399] = 255        # 2  6，1  3
    
    # cv2.imwrite('C:\\Users\\25321\\Desktop\\1.tif',img)
    # c1_index = img[266:398,26:157,:]
    # c2_index = img[56:179,283:399,:]
    c1_index = img[:,266:398,26:157]
    c2_index = img[:,56:179,283:399]
    # c3_index = img[:,50:175,37:167]
    # c4_index = img[:,262:339,237:401]
    c1_index = torch.from_numpy(c1_index)
    c2_index = torch.from_numpy(c2_index)
    JMD_torch(c1_index, c2_index, 3)


    # img = gdal.Open('C:\\Users\\25321\\Desktop\\image1.tif')
    # label = gdal.Open('C:\\Users\\25321\\Desktop\\label1_gray.tif')
    # img_w = img.RasterXSize
    # img_h = img.RasterYSize
    # img = np.array(img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
    # label = np.array(label.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
    
    # zero_0_index = (img[0,:,:]==0)
    # zero_1_index = (img[1,:,:]==0)
    # zero_2_index = (img[2,:,:]==0)
    # img[0,zero_0_index] = 1
    # img[1,zero_1_index] = 1
    # img[2,zero_2_index] = 1
    # ndvi = (img[0,:,:] - img[1,:,:])/(img[0,:,:] + img[1,:,:])
    # print(ndvi.shape)
    # c0_index = (label==1)
    # c1_index = (label==3)
    # print(img.shape,c0_index.shape)
    # # img = img[:,:,2]
    # # print(img.shape)
    # # img[266:398,26:157] = 255
    # # img[56:179,283:399] = 255        # 2  6，1  3
    
    # # cv2.imwrite('C:\\Users\\25321\\Desktop\\1.tif',img)
    # # c1_index = img[266:398,26:157,:]
    # # c2_index = img[56:179,283:399,:]

    # c0 = ndvi[c0_index]
    # c1 = ndvi[c1_index]
    # c0 = img[1,c0_index]
    # c1 = img[1,c1_index]
    # JMD_1D(c0, c1, 1)

    aw = auto_weight(10)