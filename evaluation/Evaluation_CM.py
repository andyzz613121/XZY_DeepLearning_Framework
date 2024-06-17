
import numpy as np
from osgeo import gdal
import configparser
import os
import numpy as np
from collections import Counter

import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from postprocessing.Ensemble import HS_Ensemble
from data_processing.bishe.month2year import *
class Evaluation():
    def __init__(self, class_num, 
    oa=True, aa=True, mF1=True, mIoU=True, Kappa=True, CM=True, precision=True, recall=True,
    bk_value=None, batch_flag=False):

        # 各指标计算标识
        self.oa_flag = oa
        self.aa_flag = aa
        self.mF1_flag = mF1
        self.mIoU_flag = mIoU
        self.Kappa_flag = Kappa
        self.CM_flag = CM
        self.precision = precision
        self.recall = recall
        self.batch_flag = batch_flag   # 只预测一张图还是批量预测一个文件夹里的

        # 类别数
        self.class_num = class_num

        # 忽略的背景值
        self.bk_value = bk_value
        if bk_value != None and bk_value > class_num:
            print('ERROR: bk_value > class_num')
            return

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

    def metrics(self, confu_mat_total, save_path=None, out_warning=True):
        '''
        param confu_mat_total: 总的混淆矩阵
        return: txt写出混淆矩阵, precision,recall,IOU,f-score
        '''
        return_dict = {}
        class_num = confu_mat_total.shape[0]
        # confu_mat = confu_mat_total.astype(np.float32) + 0.0001
        confu_mat = confu_mat_total.astype(np.float32)
        
        # 去除CM中未出现像素的行列(行列都为0)
        tmp_class = 0
        while tmp_class < len(confu_mat):
            max_row = np.max(confu_mat[tmp_class])
            max_col = np.max(confu_mat[:, tmp_class])
            if max_row == 0 and max_col == 0:
                confu_mat = np.delete(confu_mat, tmp_class, axis=0)
                confu_mat = np.delete(confu_mat, tmp_class, axis=1)
                if out_warning:
                    print('Warning: Delete class %d in the confu_mat, because of pixel number == 0' %tmp_class)
                tmp_class -= 1
                class_num -= 1
            tmp_class += 1

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
        if pe == 1:
            if out_warning:
                print('Warning: Kappa---pe==1, Kappa return 1')
            kappa = 1
        else:
            kappa = (oa - pe) / (1 - pe)
    
        # 将混淆矩阵写入list中
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
            if TP[i] * 2 + FP[i] + FN[i] < 1:
                f1 = 0
            else:
                f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
            f1_m.append(f1)
            
            # 写出iou
            if TP[i] + FP[i] + FN[i] < 1:
                iou = 0
            else:
                iou = TP[i] / (TP[i] + FP[i] + FN[i])
            iou_m.append(iou)
    
        f1_m = np.array(f1_m)
        iou_m = np.array(iou_m)
        
        return_dict['OA'] = oa*100
        return_dict['kappa'] = kappa*100
        return_dict['mF1'] = np.mean(f1_m)*100
        return_dict['mIoU'] = np.mean(iou_m)*100
        precision_list = []
        recall_list = []
        F1_list = []
        IoU_list = []
        for i in range(class_num):
            if raw_sum[i] == 0:
                precision_list.append(0)
                if out_warning:
                    print('Warning: class not in Predict but appear in Label, Precision return 0')
            else:
                precision_list.append(float(TP[i]/raw_sum[i])*100)

            if col_sum[i] == 0:
                recall_list.append(0)
                if out_warning:
                    print('Warning: class not in Label but appear in Predict, Recall return 0')
            else:
                recall_list.append(float(TP[i]/col_sum[i])*100)

            F1_list.append(float(f1_m[i])*100)
            IoU_list.append(float(iou_m[i])*100)
        return_dict['precision'] = precision_list
        return_dict['recall'] = recall_list
        return_dict['F1'] = F1_list
        return_dict['IoU'] = IoU_list

        if save_path is not None:
            with open(save_path, 'w') as f:
                if self.oa_flag == True:
                    f.write('OA:\t%.1f\n' % (oa*100))
                if self.Kappa_flag == True:
                    f.write('kappa:\t%.1f\n' % (kappa*100))
                if self.mF1_flag == True:
                    f.write('mf1-score:\t%.1f\n' % (np.mean(f1_m)*100))
                if self.mIoU_flag == True:
                    f.write('mIou:\t%.1f\n' % (np.mean(iou_m)*100))
        
                # 写出precision
                if self.precision == True:
                    f.write('precision:\n')
                    for i in range(class_num):
                        f.write('%.1f\t' % (float(TP[i]/raw_sum[i])*100))
                    f.write('\n\n')
    
                # 写出recall
                if self.recall == True:
                    f.write('recall:\n')
                    for i in range(class_num):
                        f.write('%.1f\t' % (float(TP[i] /col_sum[i])*100))
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

        return return_dict

    def compute_metrics_singleimg(self, PreImg_file, ValImg_file, Out_folder=None, out_warning=True):
        '''
            PreImg_file: 预测图像
            ValImg_file: 验证图像
            Out_folder: 输出精度统计文件路径(如果为None则只通过函数返回值返回精度)
        '''
        # 打开图像, 并检查预测图与验证图大小是否一致
        pre = self.open_img(PreImg_file)
        lab = self.open_img(ValImg_file)
        assert pre.shape == lab.shape
        # pre = month2year(pre, month)
        lab = np.resize(lab, (lab.shape[0]*lab.shape[1]))
        pre = np.resize(pre, (pre.shape[0]*pre.shape[1]))
        
        # # 去除背景值
        # if self.bk_value != None:
        #     pos_index = (self.lab!=0)
        #     lab_evl = self.lab[pos_index]
        #     pre_evl = self.pre[pos_index]
        # else:
        #     lab_evl = self.lab
        #     pre_evl = self.pre
        
        cm = self.cal_confu_matrix(lab, pre, self.class_num)
        # 去除背景值（先统计不去除背景值的CM, 然后把背景值那一列和那一行去掉）
        if self.bk_value != None:
            cm = np.delete(cm, self.bk_value, axis=0)
            cm = np.delete(cm, self.bk_value, axis=1)

        if Out_folder is not None:
            out_acc_filename = Out_folder + 'Accuracy_singleimg.txt'
            return self.metrics(cm, out_acc_filename, out_warning=out_warning)
        else:
            return self.metrics(cm, None, out_warning=out_warning)

    def compute_metrics_folderimg(self, PreImg_folder, ValImg_folder, Out_folder, valimg_hzname='.png'):
        '''
            Input: PreImg_folder (预测图像文件夹)
                   ValImg_folder (label图像文件夹)  必须是和PreImg对应
                   Out_folder (输出文件夹)
        '''
        cm_list = []
        for item in os.listdir(PreImg_folder):
            PreImg_file = PreImg_folder + item
            if '.png' in PreImg_file or '.jpg' in PreImg_file or '.tiff' in PreImg_file or '.tif' in PreImg_file:
                img_num = item.split('.')[0]
                ValImg_file = ValImg_folder + img_num + valimg_hzname
                # ValImg_file = 'E:\\dataset\\ImageBlur\\Fuzzy Data\\Label\\Label.png'
                # 打开图像, 并检查预测图与验证图大小是否一致
                pre = self.open_img(PreImg_file)
                lab = self.open_img(ValImg_file)
                assert pre.shape == lab.shape

                lab = np.resize(lab, (lab.shape[0]*lab.shape[1]))
                pre = np.resize(pre, (pre.shape[0]*pre.shape[1]))
                
                # # 去除背景值
                # if self.bk_value != None:
                #     pos_index = (self.lab!=0)
                #     lab_evl = self.lab[pos_index]
                #     pre_evl = self.pre[pos_index]
                # else:
                #     lab_evl = self.lab
                #     pre_evl = self.pre
                cm_list.append(self.cal_confu_matrix(lab, pre, self.class_num))
            else:
                print('Not image file (pass):  ', PreImg_file)
        cm = np.array(cm_list).sum(0)
        # 去除背景值（先统计不去除背景值的CM, 然后把背景值那一列和那一行去掉）
        if self.bk_value != None:
            cm = np.delete(cm, self.bk_value, axis=0)
            cm = np.delete(cm, self.bk_value, axis=1)
        out_acc_filename = Out_folder + 'Accuracy_folderimg.txt'
        return self.metrics(cm, out_acc_filename)

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

        img_path = folder + '500.tif'
        bk_value = 0
        if dataset == 'Houston2013':
            class_num = 15
            val_path = HS_config_dict['Houston13'][6][1]
            Ensemble.Houston13_ensemble(folder, ['0\\500', '1\\500', '2\\500'])
        elif dataset == 'Houston2018':
            class_num = 20
            val_path = HS_config_dict['Houston18'][6][1]
            Ensemble.Houston18_ensemble(folder, ['0\\500', '1\\500', '2\\500'])
        elif dataset == 'Pavia':
            class_num = 9
            val_path = HS_config_dict['Pavia'][6][1]
            Ensemble.Pavia_ensemble(folder, ['0\\500', '1\\500', '2\\500'])
        elif dataset == 'Salinas':
            class_num = 16
            val_path = HS_config_dict['Salinas'][6][1]
            Ensemble.Salinas_ensemble(folder, ['2\\500'])
        else:
            print('Unknown Dataset for eval')
            return

        # class_num需要加上1，因为有一个背景
        # Eval = Evaluation(img_path, val_path, folder, class_num+1, oa, aa, mF1, mIoU, Kappa, CM, precision, recall, bk_value)
        Eval = Evaluation(class_num+1, oa, aa, mF1, mIoU, Kappa, CM, precision, recall, bk_value)
        self.acc = Eval.compute_metrics_singleimg(img_path, val_path, folder+'\\BatchEval_')

        
if __name__ == '__main__':
    import os
    from data_processing.excel import *
    content = []
    for i in range(100):
    # for net in ['Transformer_decompms_(1_2_3)']:
        net = 'Transformer_decompms_(1_2_3)'+str(i)
        ct1 = []
        find_flag = 0
        base_folder = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\Time_500\\SS\\' +net+'\\'
        aa = Evaluation(13)
        bb = aa.compute_metrics_singleimg(base_folder+'pred_'+net+'_pre.png', 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Imgs\\Spectral_pred\\lab_test.tif', base_folder)
        pa, oa, K, IOU = bb['recall'], bb['OA'], bb['kappa'], bb['mIoU']
        print(net, oa)
        ct1.append(net)
        for ii in pa:
            ct1.append(ii/100)
        ct1.append(oa/100)
        ct1.append(np.mean(np.array(pa))/100)
        ct1.append(IOU/100)
        ct1.append(K/100)
        
        find_flag = 1
        content.append(ct1)
    write_excel('C:\\Users\\25321\\Desktop\\paper_ss.xls', content)


    # content = []
    # for date in ['5_20', '6_29', '7_14', '8_04', '9_12','10_17']:
    #     if date == '6_29':
    #         datet = '6.29'
    #     elif date == '7_14':
    #         datet = '7.14'
    #     elif date == '8_04':
    #         datet = '8.04'
    #     elif date == '9_12':
    #         datet = '9.12'
    #     elif date == '5_20':
    #         datet = '5.20'
    #     elif date == '10_17':
    #         datet = '10.17'
    #     for net in ['pred_hs_pre.png']:
    #         ct1 = []
    #         find_flag = 0
    #         base_folder = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\Spatial_Spectral\\Spectral\\'+date+'\\'
    #         aa = Evaluation(13)         # 'E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\Test\\'+datet+'\\lab_test.tif'
    #         bb = aa.compute_metrics_singleimg(date, base_folder+net, 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Imgs\\Spectral_pred\\lab_test.tif', base_folder)
    #         pa, oa, K, IOU = bb['recall'], bb['OA'], bb['kappa'], bb['mIoU']
    #         print(net, oa)
    #         ct1.append(net)
    #         for ii in pa:
    #             ct1.append(ii/100)
    #         ct1.append(oa/100)
    #         ct1.append(np.mean(np.array(pa))/100)
    #         ct1.append(IOU/100)
    #         ct1.append(K/100)
            
    #         find_flag = 1
    #         content.append(ct1)
    # write_excel('C:\\Users\\25321\\Desktop\\spectral.xls', content)