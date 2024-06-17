import torch
from torch import nn
import sys
import numpy as np

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from data_processing.Raster import gdal_read_tif


k_means20, _ = gdal_read_tif('F:\\新建文件夹\\test\\pykmeans_20.tif')
k_means40, _ = gdal_read_tif('F:\\新建文件夹\\test\\pykmeans_40.tif')
k_means80, _ = gdal_read_tif('F:\\新建文件夹\\test\\pykmeans_80.tif')
label, _ = gdal_read_tif('F:\\新建文件夹\\test\\lab.tif')
out_file = 'F:\\新建文件夹\\test\\feature.csv'

# # 针对类别号
# seg_dict = {0: [], 1: [], 2: [], 3: [], 4: []}
# for i in range(label.shape[0]):
#     for j in range(label.shape[1]):
#         fv = [k_means20[i][j]/20, k_means40[i][j]/40, k_means80[i][j]/80]
#         seg_idx = label[i][j]
#         insert_flag = 1
#         for item in seg_dict[seg_idx]:
#             if (item[0] == fv[0]) and (item[1] == fv[1]) and (item[2] == fv[2]):
#                 insert_flag = 0
#                 break
#         if insert_flag == 1:
#             seg_dict[seg_idx].append(fv)

# 针对类中心
seg_dict = {0: [], 1: [], 2: [], 3: [], 4: []}
for i in range(label.shape[0]):
    for j in range(label.shape[1]):
        fv = np.concatenate((k_means20[:,i,j]/255, k_means40[:,i,j]/255, k_means80[:,i,j]/255), 0)
        seg_idx = label[i][j]
        insert_flag = 1
        for item in seg_dict[seg_idx]:
            # print(fv, item, (item == fv).all())
            if (item == fv).all():
                insert_flag = 0
                break
        if insert_flag == 1:
            seg_dict[seg_idx].append(fv)
print(len(seg_dict[0]),len(seg_dict[1]),len(seg_dict[2]),len(seg_dict[3]))
pos_sample, neg_sample = [], []
for pair1 in range(4):
    seg_list1 = seg_dict[pair1]
    for pair2 in range(4):
        seg_list2 = seg_dict[pair2]
        for segs1 in seg_list1:
            for segs2 in seg_list2:
                sample1 = torch.from_numpy(np.array(segs1)).unsqueeze(0)
                sample2 = torch.from_numpy(np.array(segs2)).unsqueeze(0)
                sample = torch.cat([sample1, sample2], 1)
                if pair1 == pair2: 
                    pos_sample.append(sample)
                else:
                    neg_sample.append(sample)
print(len(pos_sample), len(neg_sample))
with open(out_file, 'w') as f:
    for i in range(len(pos_sample)):
        fv = pos_sample[i].numpy()[0]
        for num in range(fv.shape[0]):
            f.write(str(fv[num]))
            f.write(',')
        f.write('1')
        f.write('\n')
    for i in range(len(neg_sample)):
        fv = neg_sample[i].numpy()[0]
        for num in range(fv.shape[0]):
            f.write(str(fv[num]))
            f.write(',')
        f.write('0')
        f.write('\n')

