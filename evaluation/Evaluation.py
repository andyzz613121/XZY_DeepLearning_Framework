
import numpy as np
from osgeo import gdal
import configparser

import numpy as np
from collections import Counter
from postprocessing.Ensemble import HS_Ensemble
class Evaluation():
    def __init__(self, PreImg_file, ValImg_file, Out_folder, class_num, 
    oa=True, aa=True, mF1=True, mIoU=True, Kappa=True, CM=True, precision=True, recall=True,
    bk_value=None):

        # 各指标计算标识
        self.oa_flag = oa
        self.aa_flag = aa
        self.mF1_flag = mF1
        self.mIoU_flag = mIoU
        self.Kappa_flag = Kappa
        self.CM_flag = CM
        self.precision = precision
        self.recall = recall

        # 类别数
        self.class_num = class_num

        # 忽略的背景值
        self.bk_value = bk_value
        if bk_value != None and bk_value > class_num:
            print('ERROR: bk_value > class_num')
            return

        # 打开图像, 并检查预测图与验证图大小是否一致
        self.pre = self.open_img(PreImg_file)
        self.lab = self.open_img(ValImg_file)
        assert self.pre.shape == self.lab.shape

        self.compute_metrics(Out_folder)

    def open_img(self, path):
        img = gdal.Open(path)
        img_w = img.RasterXSize
        img_h = img.RasterYSize
        img = np.array(img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
        return img

    def cal_confu_matrix(self, label, predict, class_num):
        confu_list = []
        for i in range(class_num):
            c = Counter(predict[np.where(label == i)])
            single_row = []
            for j in range(class_num):
                single_row.append(c[j])
            confu_list.append(single_row)
        return np.array(confu_list).astype(np.int32)

    def metrics(self, confu_mat_total, save_path=None):
        '''
        param confu_mat_total: 总的混淆矩阵
        return: txt写出混淆矩阵, precision,recall,IOU,f-score
        '''

        class_num = confu_mat_total.shape[0]
        confu_mat = confu_mat_total.astype(np.float32) + 0.0001

        col_sum = np.sum(confu_mat, axis=1)  # 按行求和
        raw_sum = np.sum(confu_mat, axis=0)  # 每一列的数量

        '''计算各类面积比, 以求OA值'''
        oa = 0
        for i in range(class_num):
            oa = oa + confu_mat[i, i]
        oa = oa / confu_mat.sum()
    
        '''Kappa'''
        pe_fz = 0
        for i in range(class_num):
            pe_fz += col_sum[i] * raw_sum[i]
        pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
        kappa = (oa - pe) / (1 - pe)
    
        # 将混淆矩阵写入excel中
        TP = []  # 识别中每类分类正确的个数
    
        for i in range(class_num):
            TP.append(confu_mat[i, i])
    
        # 计算f1-score
        TP = np.array(TP)
        FN = col_sum - TP
        FP = raw_sum - TP
    
        # 计算并写出precision，recall, f1-score，f1-m以及mIOU
        f1_m = []
        iou_m = []
        for i in range(class_num):
            # 写出f1-score
            f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
            f1_m.append(f1)
            iou = TP[i] / (TP[i] + FP[i] + FN[i])
            iou_m.append(iou)
    
        f1_m = np.array(f1_m)
        iou_m = np.array(iou_m)
        
        if save_path is not None:
            with open(save_path, 'w') as f:
                if self.oa_flag == True:
                    f.write('OA:\t%.3f\n' % (oa*100))
                if self.Kappa_flag == True:
                    f.write('kappa:\t%.3f\n' % (kappa*100))
                if self.mF1_flag == True:
                    f.write('mf1-score:\t%.3f\n' % (np.mean(f1_m)*100))
                if self.mIoU_flag == True:
                    f.write('mIou:\t%.3f\n' % (np.mean(iou_m)*100))
        
                # 写出precision
                if self.precision == True:
                    f.write('precision:\n')
                    for i in range(class_num):
                        f.write('%.1f\t' % (float(TP[i]/raw_sum[i])*100))
                        # f.write('%.1f\t' % (raw_sum[i]))
                    f.write('\n\n')
    
                # 写出recall
                if self.recall == True:
                    f.write('recall:\n')
                    for i in range(class_num):
                        f.write('%.1f\t' % (float(TP[i] /col_sum[i])*100))
                        # f.write('%.1f\t' % (col_sum[i]))
                    f.write('\n\n')

                if self.mF1_flag == True:
                    # 写出f1-score
                    f.write('f1-score:\n')
                    for i in range(class_num):
                        f.write('%.1f\t' % (float(f1_m[i])*100))
                    f.write('\n\n')

                if self.mIoU_flag == True:
                    # 写出 IOU
                    f.write('Iou:\n')
                    for i in range(class_num):
                        f.write('%.1f\t' % (float(iou_m[i])*100))
                    f.write('\n\n')

                if self.CM_flag == True:
                    f.write('Confuse Matrix 横向相加为Label的数量，纵向相加为Predict的数量:\n')
                    f.write('XXXXX  Predict\n')
                    f.write('Label\n')
                    for i in range(class_num):
                        for j in range(class_num):
                            f.write('%.1f\t' % (float(confu_mat[i][j])))
                        f.write('\n')
        else:
            print('The out path is false')

    def compute_metrics(self, pre_folder):
        self.lab = np.resize(self.lab, (self.lab.shape[0]*self.lab.shape[1]))
        self.pre = np.resize(self.pre, (self.pre.shape[0]*self.pre.shape[1]))
        
        # # 去除背景值
        # if self.bk_value != None:
        #     pos_index = (self.lab!=0)
        #     lab_evl = self.lab[pos_index]
        #     pre_evl = self.pre[pos_index]
        # else:
        #     lab_evl = self.lab
        #     pre_evl = self.pre
        
        cm = self.cal_confu_matrix(self.lab, self.pre, self.class_num)
        # 去除背景值（先统计不去除背景值的CM, 然后把背景值那一列和那一行去掉）
        if self.bk_value != None:
            cm = np.delete(cm, self.bk_value, axis=0)
            cm = np.delete(cm, self.bk_value, axis=1)
        print(cm)
        out_acc_filename = pre_folder + 'Accuracy.txt'
        self.metrics(cm, out_acc_filename)

# 针对Houston13，Houston18，Pavia数据集的验证类
class Evaluation_HS():
    def __init__(self, dataset, folder, 
    oa=True, aa=True, mF1=True, mIoU=True, Kappa=True, CM=True, precision=True, recall=True,
    bk_value=None):

        # 输入数据集以及验证图像存的文件夹，自动在文件夹下输出精度评价结果
        # 输入的预测图像必须在该文件夹下，且名称为：Ensemble.tif（经过集成学习）
        Ensemble = HS_Ensemble()

        # 读取配置文件
        HS_config = configparser.ConfigParser()
        HS_config.read('dataset\\Configs\\HS_Config.ini',encoding='UTF-8')
        HS_key_list = HS_config.sections()
        HS_value_list = []
        for item in HS_key_list:
            HS_value_list.append(HS_config.items(item))
        HS_config_dict = dict(zip(HS_key_list, HS_value_list))

        img_path = folder + 'Ensemble.tif'
        bk_value = 0
        if dataset == 'Houston13':
            class_num = 15
            val_path = HS_config_dict['Houston13'][6][1]
            Ensemble.Houston13_ensemble(folder)
        elif dataset == 'Houston18':
            class_num = 20
            val_path = HS_config_dict['Houston18'][6][1]
            Ensemble.Houston18_ensemble(folder)
        elif dataset == 'Pavia':
            class_num = 9
            val_path = HS_config_dict['Pavia'][6][1]
            Ensemble.Pavia_ensemble(folder)
        elif dataset == 'Salinas':
            class_num = 16
            val_path = HS_config_dict['Salinas'][6][1]
            Ensemble.Salinas_ensemble(folder)
        else:
            print('Unknown Dataset for eval')
            return

        # class_num需要加上1，因为有一个背景
        Eval = Evaluation(img_path, val_path, folder, class_num+1, oa, aa, mF1, mIoU, Kappa, CM, precision, recall, bk_value)
        return Eval
    
if __name__ == '__main__':
    # base_path = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\Response\\no_spatial\\Pavia\\771\\'
    # Eval = Evaluation_HS('Pavia', base_path)
    
    from data_processing.excel import *
    content = []
    for date in ['Houston2013', 'Houston2018', 'Pavia', 'Salinas']:
        for net in ['AE_1D', 'AE_LSF', 'AE_GSF', 'AE_CO', 'AE_SO']:
            find_flag = 0
            basefolder = 'D:\\毕业\\博士论文\\毕业论文\\图\\第四章\\实验结果\\HS数据\\' + date + '\\' + net + '\\' 
            aa = Evaluation_HS(date, basefolder)
                
            # pa, oa, K, = aa['precision'], aa['OA'], aa['kappa']
            # for ii in pa:
            #     ct1.append(ii/100)
            # ct1.append(oa/100)
            # ct1.append(K/100)
            # print(pre_file)
            # find_flag = 1
            print(aa)
            break
        break
                # content.append(ct1)
    write_excel('C:\\Users\\25321\\Desktop\\acc0.xls', content)
