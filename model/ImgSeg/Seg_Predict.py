import torch
from torch import nn

import sys
import numpy as np

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from data_processing.Raster import gdal_read_tif, gdal_write_tif

class MLP(nn.Module):
    def __init__(self, channel):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, 5, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(5, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

k_means20, _ = gdal_read_tif('F:\\新建文件夹\\test\\pykmeans_20.tif')
k_means40, _ = gdal_read_tif('F:\\新建文件夹\\test\\pykmeans_40.tif')
k_means80, _ = gdal_read_tif('F:\\新建文件夹\\test\\pykmeans_80.tif')
model = torch.load('F:\\新建文件夹\\test\\model.pkl').cuda()

img_result = torch.from_numpy(np.zeros_like(k_means20[0]) - 1)
cur_segidx = 1
for i in range(k_means20.shape[1]):
    for j in range(k_means20.shape[2]):
        # print(i, j)
        # fv1 = [k_means20[i][j], k_means40[i][j], k_means80[i][j]]
        fv1 = np.concatenate((k_means20[:,i,j]/255, k_means40[:,i,j]/255, k_means80[:,i,j]/255), 0)
        sample1 = torch.from_numpy(np.array(fv1)).float().cuda()
        start_x = (i-1) if (i-1)>=0 else 0
        end_x = (i+1) if (i-1)<k_means20.shape[1] else k_means20.shape[1]-1
        start_y = (j-1) if (j-1)>=0 else 0
        end_y = (j+1) if (j-1)<k_means20.shape[2] else k_means20.shape[2]-1

        candidate_segidx = cur_segidx + 1       # 初始化情况下，当前像素归属的图斑为cur_segidx + 1，即是一个不合并的新图斑
        cur_prob = 0                            # 8联通范围内合并的最大概率
        merge_flag = 0                          # 合并标志，如果没有与8联通的合并，说明是新的图斑，cur_segidx要+1
        for sub_i in range(start_x, end_x):   # 遍历8联通区域
            for sub_j in range(start_y, end_y):
                if (img_result[sub_i][sub_j] == -1):    # 只将待合并像元与已经确定类别的像元进行合并判断
                    continue
                else:
                    # fv2 = [k_means20[sub_i][sub_j], k_means40[sub_i][sub_j], k_means80[sub_i][sub_j]]
                    fv2 = np.concatenate((k_means20[:,sub_i,sub_j]/255, k_means40[:,sub_i,sub_j]/255, k_means80[:,sub_i,sub_j]/255), 0)
                    sample2 = torch.from_numpy(np.array(fv2)).float().cuda()
                    sample = torch.cat([sample1, sample2], 0)
                    pred = model(sample)
                    print(pred, sample)
                    # if (pred<0.5):
                    #     print('aaaaaaaaaaaaaaaaaaaaaaaa', sub_i, sub_j)
                    if (pred>cur_prob):# 更新最大概率与图斑下标
                        merge_flag = 1
                        cur_prob = pred
                        candidate_segidx = img_result[sub_i][sub_j]
        # print(candidate_segidx)
        img_result[i][j] = candidate_segidx
        if merge_flag == 0:
            cur_segidx += 1
                
img_result = img_result.numpy()
gdal_write_tif('F:\\新建文件夹\\test\\result.tif', img_result, img_result.shape[1], img_result.shape[0], 1)

