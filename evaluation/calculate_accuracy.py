import os
import numpy as np
from osgeo import gdal

import os
import numpy as np
from glob import glob
from collections import Counter

def cal_confusion_degree(predict, label):
    confuse_matrix = np.zeros([5,5]).astype(np.float)
    confusion_degree = np.zeros([5,5]).astype(np.float)


    pre_pos_list = [] #predict等于各个类的下标数组
    label_pos_list = [] #label等于各个类的下标数组
    label = label[:,0,:,:]
    for pre_class in range(5):
        pos_index = (predict == pre_class)
        pre_pos_list.append(pos_index)
    for label_class in range(5):
        pos_index = (label == label_class)
        label_pos_list.append(pos_index)
    
    for pre_class in range(5):
        for label_class in range(5):
            if pre_class != label_class:
                pos_index = pre_pos_list[pre_class]*label_pos_list[label_class]
                confuse_matrix[pre_class][label_class] = (pos_index.sum())

    zero_index = (confuse_matrix == 0)
    #confusion_matrix Vaihingen
    #                           
    #                                  真实 
    #        预测：          不透水面      房屋      低矮植被      高植被       车
    #      不透水面             -          DEM        NDVI      NDVI + DEM   DIS
    #      房屋                DEM          -       NDVI+DEM      NDVI    DEM + DIS
    #      低矮植被            NDVI      NDVI+DEM      -           DEM       NDVI
    #      高植被           NDVI + DEM     NDVI       DEM           -     NDVI + DEM
    #      车                  DIS      DEM + DIS    NDVI      NDVI + DEM     - 
    
    #confusion_matrix Vaihingen
    #                           
    #                                  真实 
    #        预测：          不透水面      房屋      低矮植被      高植被       车
    #      不透水面             -          DEM        NDVI      NDVI + DEM   DIS
    #      房屋                DEM          -        'DEM'      NDVI    DEM + DIS
    #      低矮植被            NDVI       'DEM'     -           DEM       NDVI
    #      高植被           NDVI + DEM     NDVI       DEM           -     NDVI + DEM
    #      车                  DIS      DEM + DIS    NDVI      NDVI + DEM     - 

    #confusion_matrix Vaihingen
    #                           
    #                                  真实 
    #        预测：          不透水面      房屋      低矮植被      高植被       车
    #      不透水面             -          DEM        NDVI      NDVI + DEM   DIS
    #      房屋                DEM          -       NDVI+DEM      NDVI      DEM
    #      低矮植被            NDVI      NDVI+DEM      -           DEM       NDVI
    #      高植被           NDVI + DEM     NDVI       DEM           -     NDVI + DEM
    #      车                  DIS         DEM    NDVI      NDVI + DEM     - 

    for pre_class in range(5):
        confusion_degree[pre_class] = confuse_matrix[pre_class]/(sum(confuse_matrix[pre_class])-confuse_matrix[pre_class][pre_class])
    return confusion_degree

def cal_confu_matrix(label, predict, class_num):
    confu_list = []
    for i in range(class_num):
        c = Counter(predict[np.where(label == i)])
        #print(c)
        single_row = []
        for j in range(class_num):
            single_row.append(c[j])
            #print(single_row)
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int32)

def cal_confu_matrix_relax(label, predict, class_num, relax_num):
    confu_mat = []
    for i in range(3):
        confu_l = []
        for j in range(3):
            confu_l.append(0)
        confu_mat.append(confu_l)
    
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            
            # if label[i][j]==predict[i][j]:
            #     confu_mat[label[i][j]][label[i][j]]+=1
            # else:
            #     confu_mat[label[i][j]][predict[i][j]]+=1

            start_i = i-relax_num
            start_j = j-relax_num
            end_i = i+relax_num
            end_j = j+relax_num
            if start_i < 0:
                start_i = i
            if end_i >= label.shape[0]:
                end_i = label.shape[0] - 1
            if start_j < 0:
                start_j = j
            if end_j >= label.shape[1]:
                end_j = label.shape[1] - 1

            temp = 0
            for x in range(start_i, end_i):
                for y in range(start_j, end_j):
                    if predict[x][y] == label[i][j]:
                            temp=1
            if temp==1:
                confu_mat[label[i][j]][label[i][j]]+=1
            else:
                confu_mat[label[i][j]][predict[i][j]]+=1

    return np.array(confu_mat).astype(np.int32)
 
 
def metrics(confu_mat_total, save_path=None):
    '''
    :param confu_mat: 总的混淆矩阵
    backgound：是否干掉背景
    :return: txt写出混淆矩阵, precision，recall，IOU，f-score
    '''

    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float32) + 0.0001

    col_sum = np.sum(confu_mat, axis=1)  # 按行求和
    raw_sum = np.sum(confu_mat, axis=0)  # 每一列的数量
 
    '''计算各类面积比，以求OA值'''
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
    
    class_rate = raw_sum/raw_sum.sum()
    if save_path is not None:
        with open(save_path + 'accuracy.txt', 'w') as f:
            f.write('OA:\t%.1f\n' % (oa*100))
            f.write('kappa:\t%.1f\n' % (kappa*100))
            f.write('mf1-score:\t%.1f\n' % (np.mean(f1_m)*100))
            f.write('mIou:\t%.1f\n' % (np.mean(iou_m)*100))
 
            # 写出precision
            f.write('precision:\n')
            for i in range(class_num):
                f.write('%.1f\t' % (float(TP[i]/raw_sum[i])*100))
                f.write('%.1f\t' % (raw_sum[i]))
            f.write('\n\n')
 
            # 写出recall
            f.write('recall:\n')
            for i in range(class_num):
                f.write('%.1f\t' % (float(TP[i] /col_sum[i])*100))
                f.write('%.1f\t' % (col_sum[i]))
            f.write('\n\n')
 
            # 写出f1-score
            f.write('f1-score:\n')
            for i in range(class_num):
                f.write('%.1f\t' % (float(f1_m[i])*100))
            f.write('\n\n')
 
            # 写出 IOU
            f.write('Iou:\n')
            for i in range(class_num):
                f.write('%.1f\t' % (float(iou_m[i])*100))
            f.write('\n\n')

            f.write('Confuse Matrix 横向相加为Label的数量，纵向相加为Predict的数量:\n')
            f.write('XXXXX  Predict\n')
            f.write('Label\n')
            for i in range(class_num):
                for j in range(class_num):
                    f.write('%.1f\t' % (float(confu_mat[i][j])))
                f.write('\n')

def test_vai(pre_folder, erode=False):
    #test vaihingen
    if erode == False:
        label_folder = 'D:\\Papers\\投稿\\Semantic segmentation of VHR remote sensing images fused with classified boundary information\\论文数据\\label_gray\\'
    else:
        label_folder = 'D:\\Papers\\投稿\\Semantic segmentation of VHR remote sensing images fused with classified boundary information\\论文数据\\label_erode_gray\\'
    label_list = []
    pre_list = []
    for i in os.listdir(pre_folder):
        filename = pre_folder + i
        if 'tif' in i:
            if '5_ensemble' in i:
                pic_num = 5
            elif '7_ensemble' in i:
                pic_num = 7
            elif '23_ensemble' in i:
                pic_num = 23
            elif '30_ensemble' in i:
                pic_num = 30
            else:
                continue
            print(i)
            pre = gdal.Open(filename)
            img_w = pre.RasterXSize
            img_h = pre.RasterYSize
            pre = np.array(pre.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
            label_filename = label_folder + str(pic_num) + '.tif'
            label = gdal.Open(label_filename)
            label = np.array(label.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)

            label = np.resize(label, (label.shape[0]*label.shape[1]))
            pre = np.resize(pre, (pre.shape[0]*pre.shape[1]))

            pos_index = (label!=6)
            label0 = label[pos_index]
            pre0 = pre[pos_index]

            # CD = cal_confusion_degree(pre0, label0)
            # print(CD)

            if 'erode' in label_folder:
                eroded_flag = 'erode'
            else:
                eroded_flag = 'original'

            for class_number in [5, 6]:
                a = cal_confu_matrix(label0, pre0, class_number)
                out_acc_filename = pre_folder + str(pic_num) + '_' + eroded_flag + '_' + str(class_number)
                metrics(a, out_acc_filename)

            label_list.append(label0)
            pre_list.append(pre0)


    label_all = np.hstack((label_list[0],label_list[1],label_list[2],label_list[3]))
    pre_all = np.hstack((pre_list[0],pre_list[1],pre_list[2],pre_list[3]))
    for class_number in [5, 6]:
        a = cal_confu_matrix(label_all, pre_all, class_number)
        out_acc_filename = pre_folder + 'total_' + eroded_flag + '_' + str(class_number)
        metrics(a, out_acc_filename)

def test_pos():
    import os
    import numpy as np
    from osgeo import gdal
    pre_folder = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\Skeleton\\ResU_Net_Pos_ske\\新建文件夹\\'
    #label_folder = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_pos\\label_gray_erode\\'
    label_folder = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_pos\\label_gray\\'
    label_list = []
    pre_list = []
    for i in os.listdir(pre_folder):
        filename = pre_folder + i
        if '2_11' in i:
            pic_num = '2_11'
        elif '4_10' in i:
            pic_num = '4_10'
        elif '5_11' in i:
            pic_num = '5_11'
        elif '7_08' in i:
            pic_num = '7_08'
        else:
            continue
        if 'tif' not in i:
            continue
        print(i)
        pre = gdal.Open(filename)
        img_w = pre.RasterXSize
        img_h = pre.RasterYSize
        pre = np.array(pre.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
        label_filename = label_folder + 'label' + pic_num + '_gray.tif'
        label = gdal.Open(label_filename)
        label = np.array(label.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)

        label = np.resize(label, (label.shape[0]*label.shape[1]))
        pre = np.resize(pre, (pre.shape[0]*pre.shape[1]))

        if 'erode' in label_folder:
            eroded_flag = 'erode'
        else:
            eroded_flag = 'original'

        pos_index = (label!=6)
        label0 = label[pos_index]
        pre0 = pre[pos_index]

        for class_number in [5, 6]:
            a = cal_confu_matrix(label0, pre0, class_number)
            out_acc_filename = pre_folder + pic_num + '_' + eroded_flag + '_' + str(class_number)
            metrics(a, out_acc_filename)
        
        label_list.append(label0)
        pre_list.append(pre0)


    label_all = np.hstack((label_list[0],label_list[1],label_list[2],label_list[3]))
    pre_all = np.hstack((pre_list[0],pre_list[1],pre_list[2],pre_list[3]))
    for class_number in [5, 6]:
        a = cal_confu_matrix(label_all, pre_all, class_number)
        out_acc_filename = pre_folder + eroded_flag + '_' + str(class_number)
        metrics(a, out_acc_filename)

def test_rb():
    import os
    import numpy as np
    from osgeo import gdal
    pre_folder = 'C:\\Users\\ASUS\\Desktop\\RB\\文章\\RB_SegNet\\'
    # label_folder = 'D:\\dataset\\Road_Building_Datasets\\Valid\\merge_Label\\'
    label_folder = 'D:\\dataset\\Road_Building_Datasets\\Test\\Label_merge\\'
    #label_folder = 'D:\\dataset\\Road_Building_Datasets\\Test\\Label_merge_dilate\\'
    label_list = []
    pre_list = []
    relax_num = 3
    for i in os.listdir(pre_folder):
        filename = pre_folder + i
        if '.tif' in i:
            pic_num = i.split('-')[0]
            pic_num = pic_num.split('pre')[1]
            print(pic_num)
            
            pre = gdal.Open(filename)
            img_w = pre.RasterXSize
            img_h = pre.RasterYSize
            pre = np.array(pre.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
            label_filename = label_folder + pic_num + '.tif'
            label = gdal.Open(label_filename)
            label = np.array(label.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
            
            # label_0_index = (label==0)
            # pre_0_index = (pre==0)
            # label[label_0_index] = 3
            # pre[pre_0_index] = 3
            # label = label-1
            # pre = pre-1

            label_list.append(label)
            pre_list.append(pre)

            
            relaxed = cal_confu_matrix_relax(label, pre, 3, 3)
            non_relaxed = cal_confu_matrix(label, pre, 3)
            relax_filename_metrics = pre_folder + 'relax' + str(relax_num) + '_' + pic_num
            non_relax_filename_metrics = pre_folder + 'non_relax_' + pic_num
            metrics(relaxed, relax_filename_metrics)
            metrics(non_relaxed, non_relax_filename_metrics)

    label_all = np.hstack((label_list[0],label_list[1],label_list[2],label_list[3]))
    pre_all = np.hstack((pre_list[0],pre_list[1],pre_list[2],pre_list[3]))

    relaxed = cal_confu_matrix_relax(label_all, pre_all, 3, 3)
    non_relaxed = cal_confu_matrix(label_all, pre_all, 3)
    relax_filename_metrics = pre_folder + 'all_relax'+ str(relax_num) + '_' + pic_num
    non_relax_filename_metrics = pre_folder + 'all_non_relax_' + pic_num
    metrics(relaxed, relax_filename_metrics)
    metrics(non_relaxed, non_relax_filename_metrics)

def test_GID(pre_folder, label_folder):
    #test GID
    label_list = []
    pre_list = []
    for item in os.listdir(pre_folder):
        if '.tif' in item:
            filename = pre_folder + item
            img_name = item.split('_0')[0]
            img_name = img_name.split('pre')[1]

            pre = gdal.Open(filename)
            img_w = pre.RasterXSize
            img_h = pre.RasterYSize
            pre = np.array(pre.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
            label_filename = label_folder + str(img_name) + '_label.tif'
            label = gdal.Open(label_filename)
            label = np.array(label.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)

            label = np.resize(label, (label.shape[0]*label.shape[1]))
            pre = np.resize(pre, (pre.shape[0]*pre.shape[1]))

            class_number = 6
            a = cal_confu_matrix(label, pre, class_number)
            out_acc_filename = pre_folder + str(img_name) + '_' + str(class_number)
            metrics(a, out_acc_filename)

            label_list.append(label)
            pre_list.append(pre)

    label_all = np.hstack(label for label in label_list)
    pre_all = np.hstack(pre for pre in pre_list)
    a = cal_confu_matrix(label_all, pre_all, class_number)
    out_acc_filename = pre_folder + 'total_' + str(class_number)
    metrics(a, out_acc_filename)

if __name__ == '__main__':
    # test_GID('D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\GID_Base\\', 'E:\\dataset\\GID数据集\\test\\big_label\\')
    test_vai('D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\20-15\\')