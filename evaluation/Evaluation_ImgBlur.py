import os
import sys
import numpy as np
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

from data_processing.Raster import gdal_read_tif

def recall_with_rate(lab, pred, rate, ignor_zero=True):
    lab_pos = (lab == 1)
    pred_pos = (pred == 1)

    true_index = lab_pos * pred_pos
    true_num = true_index.sum()

    lab_num = lab_pos.sum()
    pred_num = pred_pos.sum()

    total_num = lab_num
    print(true_num/total_num)
    if total_num == 0:
        if ignor_zero == False:
            if pred_num == 0:
                return 1
            else:
                return 0
        else:
            return -1
    else:
        if true_num/total_num > rate:
            return 1
        else:
            return 0

def acc_with_rate_zero(lab, pred, rate):
    lab_pos = (lab == 1)
    pred_pos = (pred == 1)

    true_index = lab_pos * pred_pos
    true_num = true_index.sum()

    total_num = pred_pos.sum()
    print(true_num/total_num)
    if total_num == 0:
        if true_num == 0:
            return 1
        else:
            return 0
    else:
        if true_num/total_num > rate:
            return 1
        else:
            return 0

def acc_with_rate_nozero(lab, pred, rate):
    lab_pos = (lab == 1)
    pred_pos = (pred == 1)

    true_index = lab_pos * pred_pos
    true_num = true_index.sum()

    total_num = lab_pos.sum()
    # if total_num == 0:
    #     if true_num == 0:
    #         return 1
    #     else:
    #         return 0
    # else:
    if total_num == 0:
        return 0
    if true_num/total_num > rate:
        return 1
    else:
        return 0

if __name__ == '__main__':

    pre_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\上传数据\\7. predict_results_in_validation_images\\'
    val_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\训练图像\\test\\label\\'
    # for i in ['High IC', 'Low IC', 'Middle IC']:
    #     for r in [1, 2, 3, 4, 5]:
            # pre_folder = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\SRNetResult\\SRNet_spam_norm\\200\\r' + str(r) + '\\' + i + '\\'
            # print('-------------------------------')
            # print(i, r)
    for rate in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        true_num = 0
        total_num = 0
        for item in os.listdir(pre_folder):
            PreImg_file = pre_folder + item
            if '.png' in PreImg_file or '.jpg' in PreImg_file or '.tiff' in PreImg_file or '.tif' in PreImg_file:
                img_num = item.split('.')[0]
                ValImg_file = val_folder + img_num + '.png'
                # ValImg_file = 'E:\\dataset\\ImageBlur\\Data\\train\\无模糊图像（复杂度+模糊半径）\\selected\\r_1\\train\\lab\\lab.png'
                
                # 打开图像, 并检查预测图与验证图大小是否一致
                pre, _ = gdal_read_tif(PreImg_file)
                lab, _ = gdal_read_tif(ValImg_file)
                assert pre.shape == lab.shape
                
                lab = np.resize(lab, (lab.shape[0]*lab.shape[1]))
                pre = np.resize(pre, (pre.shape[0]*pre.shape[1]))
                # lab_pos = (lab == 1)
                # if lab_pos.sum() == 0:
                #     continue
                # total_num += 1
                # true_num += acc_with_rate_nozero(lab, pre, rate)
                acc = acc_with_rate_zero(lab, pre, rate)
                print(item, acc)
                break
                if acc == -1:
                    continue
                else:
                    total_num += 1
                    true_num += acc
        print(true_num, total_num, true_num/total_num)
        