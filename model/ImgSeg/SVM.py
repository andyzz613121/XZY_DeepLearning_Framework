import os
import sys
import time
import pickle
import numpy as np
import configparser
from PIL import Image

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from dataset.ImgSeg_dataset import ImgSeg_dataset
from data_processing.Raster import gdal_read_tif, gdal_write_tif

import torch
from torch.utils import data
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def train_svm(times):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 300

    # Train Data
    ############################################################
    train_dst = ImgSeg_dataset('F:\\新建文件夹\\test\\feature.csv')
    train_loader = data.DataLoader(train_dst, batch_size = batch_size, shuffle = True)
    train_x, train_y = [], []
    for i, sample in enumerate(train_loader, 0):
        fv = sample['fv']
        lab = sample['lab']
        train_x.append(fv)
        train_y.append(lab)
    train_x = torch.cat([x for x in train_x], 0).cpu().numpy()
    train_y = torch.cat([x for x in train_y], 0).cpu().numpy()
    print(train_x.shape)
    # Train Model
    ############################################################
    model = SVC(kernel='rbf', C=1) #rbf poly  linear sigmod
    # model = MLPClassifier(hidden_layer_sizes=(16,16,16), max_iter=2000000)
    model.fit(train_x, train_y)
    
    # Save Model
    ############################################################
    image_model_name = 'F:\\新建文件夹\\test\\svm_model_xzy111.pkl'
    save_SVMmodel(model, image_model_name)

def test_svm(model_folder, dataset):
    k_means20, _ = gdal_read_tif('F:\\新建文件夹\\test\\pykmeans_20.tif')
    k_means40, _ = gdal_read_tif('F:\\新建文件夹\\test\\pykmeans_40.tif')
    k_means80, _ = gdal_read_tif('F:\\新建文件夹\\test\\pykmeans_80.tif')
    model = load_SVMmodel('F:\\新建文件夹\\test\\svm_model_xzy.pkl')

    img_result = np.zeros_like(k_means20[0]) - 1
    cur_segidx = 1
    for i in range(k_means20.shape[1]):
        for j in range(k_means20.shape[2]):
            fv1 = np.concatenate((k_means20[:,i,j]/255, k_means40[:,i,j]/255, k_means80[:,i,j]/255), 0)
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
                        fv2 = np.concatenate((k_means20[:,sub_i,sub_j]/255, k_means40[:,sub_i,sub_j]/255, k_means80[:,sub_i,sub_j]/255), 0)
                        sample = np.concatenate((fv1, fv2), 0)
                        sample = sample[np.newaxis, : ]
                        # print(sample.shape)
                        pred = model.predict(sample)
                        print(pred)
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

    gdal_write_tif('F:\\新建文件夹\\test\\result.tif', img_result, img_result.shape[1], img_result.shape[0], 1)


def load_SVMmodel(path):
    with open(path, 'rb') as f:
        model = pickle.loads(f.read())
    return model

def save_SVMmodel(model, path):
    s = pickle.dumps(model)
    with open(path, 'wb+') as f:
        f.write(s)

    
if __name__ == '__main__':
    train_svm(1)
    test_svm(1, 1)
