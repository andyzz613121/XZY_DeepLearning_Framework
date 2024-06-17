import os
import sys
import time
import pickle
import numpy as np
import configparser
from PIL import Image
import pandas as pd
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from dataset.HS_dataset_new import HS_dataset
from testing.HS.test_HS import HS_test
from model.HyperSpectral.Basic_Operation import center_pixel
from data_processing.Raster import gdal_read_tif, gdal_write_tif
from dataset.XZY_dataset_new import XZY_train_dataset, XZY_test_dataset

import torch
from torch.utils import data
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def train_svm(times, dataset_num):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 1
    PCA = False
    norm = True

    if dataset_num == 1:
        dataset = 'Houston13'
    elif dataset_num == 2:
        dataset = 'Salinas'
    elif dataset_num == 3:
        dataset = 'Pavia'
    elif dataset_num == 4:
        dataset = 'Houston18'
    elif dataset_num == 5:
        dataset = '6_29'
    elif dataset_num == 6:
        dataset = '7_14'
    elif dataset_num == 7:
        dataset = '8_04'
    elif dataset_num == 8:
        dataset = '9_12'
    elif dataset_num == 9:
        dataset = '5_20'
    elif dataset_num == 10:
        dataset = '10_17'
    else:
        return

    # Train Data
    ############################################################
    # train_dst = HS_dataset(dataset, PCA, norm)
    train_dst = XZY_train_dataset('E:\\dataset\\毕设数据\\new\\2. MS\\HS\\'+dataset+'\\train.csv', norm_list=[True, False], type=['img', 'value'])
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)
    train_x, train_y = [], []
    for i, sample in enumerate(train_loader, 0):
        pt = center_pixel(sample['img_0'])
        label = sample['lab_0']
        train_x.append(pt)
        train_y.append(label)
    train_x = torch.cat([x for x in train_x], 0).cpu().numpy()
    train_y = torch.cat([x for x in train_y], 0).cpu().numpy()

    # Train Model
    ############################################################
    model = SVC(kernel='linear', C=1) #rbf poly  linear sigmod
    # model = MLPClassifier(hidden_layer_sizes=(16,16,16), max_iter=2000000)
    model.fit(train_x, train_y)
    
    # Save Model
    ############################################################
    model_folder = 'result\\HS_new\\' + dataset + '\\' + str(times) + '\\'
    if os.path.exists(model_folder) == False:
        os.makedirs(model_folder)
    image_model_name = model_folder + '\\svm_model_xzy.pkl'
    save_SVMmodel(model, image_model_name)

def test_svm(model_folder, dataset):
    norm = True
    PCA = False

    model_path = model_folder + 'svm_model_xzy.pkl'
    model = load_SVMmodel(model_path)
    Test = XZY_test_dataset()
    imgpath = 'E:\\dataset\\毕设数据\\new\\2. MS\\Imgs\\'+dataset+'.tif'
    sample = Test.get_test_samples([imgpath])
    test_x = sample['img_0']
    img_h, img_w = test_x.shape[2], test_x.shape[3]
    test_x = test_x.view(test_x.shape[1], -1).transpose(0, 1).cpu().numpy()
    test_y = model.predict(test_x).reshape([img_h, img_w])+1
    # print(test_y.shape)
    # Save Predicts
    ############################################################
    pre_path = model_folder + '\\pre.tif'
    prergb_path = model_folder + '\\pre_RGB.tif'
    acc_path = model_folder + '\\pre.txt'

    # Test.compute_acc(test_y, acc_path)
    pre_mapout = Image.fromarray(test_y)
    pre_mapout.save(pre_path)

    # pre_rgb = Test.GRAYcvtRGB(dataset, test_y)
    # pre_rgbout = Image.fromarray(pre_rgb)
    # pre_rgbout.save(prergb_path)

def load_SVMmodel(path):
    with open(path, 'rb') as f:
        model = pickle.loads(f.read())
    return model

def save_SVMmodel(model, path):
    s = pickle.dumps(model)
    with open(path, 'wb+') as f:
        f.write(s)

def center_pixel_svm(img):
    l, h, w = img.shape
    pth = int((h - 1)/2)
    ptw = int((w - 1)/2)
    pt = img[:,pth,ptw]
    return pt

def train_svm_nums(path):
    
    csv = pd.read_csv(path).values
    csv = csv[1:,1:]
    train_x = csv[:,:-1]
    train_y = csv[:,-1]
    print(train_x.shape, train_y.shape)
    # Train Model
    ############################################################
    model = SVC(kernel='linear', C=1) #rbf poly  linear sigmod
    # model = MLPClassifier(hidden_layer_sizes=(16,16,16), max_iter=2000000)
    model.fit(train_x, train_y)
    
    # Save Model
    ############################################################
    model_folder = 'result\\SVMTEST\\'
    if os.path.exists(model_folder) == False:
        os.makedirs(model_folder)
    image_model_name = model_folder + '\\svm_model_xzy.pkl'
    save_SVMmodel(model, image_model_name)

def test_svm_num(model_folder, data_path):

    model_path = model_folder + 'svm_model_xzy.pkl'
    model = load_SVMmodel(model_path)
    csv = pd.read_csv(data_path).values
    # test_x = csv[:,:48]
    # lab_y = csv[:,48]
    csv = csv[1:,1:]
    test_x = csv[:,:-1]
    lab_y = csv[:,-1]
    test_y = model.predict(test_x)
    print(test_y, lab_y, abs(test_y-lab_y))
    

def train_svm_img():
    img = gdal_read_tif('E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\big_image.tif')[0]
    lab = gdal_read_tif('E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\train_label.tif')[0]
    val = gdal_read_tif('E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\valid_label.tif')[0]

    # max, min = np.max(img, 0), np.min(img, 0)
    # img = (img - min)/(max - min)

    pos_index = (lab > 0)
    x = np.transpose(img[:, pos_index], (1, 0))
    y = lab[pos_index]

    model = SVC(kernel='rbf') #rbf
    model.fit(x, y)

    pos_index = (val > -1)
    val_x = np.transpose(img[:, pos_index], (1, 0))
    pre = model.predict(val_x).reshape([img.shape[1], img.shape[2]])+1

    model_folder = 'result\\'
    pre_path = model_folder + '\\pre.tif'
    prergb_path = model_folder + '\\pre_RGB.tif'
    acc_path = model_folder + '\\pre.txt'
    

    Test = HS_test('Houston13', None, norm = True, PCA = False)

    Test.compute_acc(pre, acc_path)
    pre_mapout = Image.fromarray(pre)
    pre_mapout.save(pre_path)

    pre_rgb = Test.GRAYcvtRGB('Houston13', pre)
    pre_rgbout = Image.fromarray(pre_rgb)
    pre_rgbout.save(prergb_path)

    
if __name__ == '__main__':
    ###########################
    # Train
    # print('SVM')
    # for times in range(5):
    #     for dataset_num in range(1, 5):
    #         train_svm(times, dataset_num)

    ###########################
    # Test
    # for times in range(5):
    #     for dataset in ['Houston13', 'Salinas', 'Pavia', 'Houston18']: #, 'Pavia', 'Houston18'
    #         test_folder = 'result\\' + dataset + '\\' + str(times) + '\\'
    #         test_svm(test_folder, dataset)

    # for times in range(5):
    #     for dataset_num in range(5, 11):
    #         train_svm(times, dataset_num)

    # for times in range(5):
    #     for dataset in ['6_29', '7_14', '8_04', '9_12', '5_20', '10_17']:
    #         test_folder = 'result\\HS_new\\' + dataset + '\\' + str(times) + '\\'
    #         test_svm(test_folder, dataset)

    train_svm_nums('C:\\Users\\25321\\Desktop\\1\\landslides.csv')
    test_svm_num('result\\SVMTEST\\', 'C:\\Users\\25321\\Desktop\\1\\landslides1c.csv')