import numpy as np
import cv2
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
class DIS_MAP():
    def Make_One_Hot(self, label, class_num):
        array = []
        for i in range(class_num):
            pos_index = (label==i)
            array.append(pos_index.astype(np.uint8))
        array = np.array(array)
        return array

    def Gen_Dis_Map(self, label, max_flag=False):
        one_hot = self.Make_One_Hot(label,6)
        dis_final = np.zeros([label.shape[0], label.shape[1]]).astype(np.float32)
        for i in range(one_hot.shape[0]):
            pos_index = (label == i)
            dis = cv2.distanceTransform(one_hot[i], cv2.DIST_L2, 0)
            dis = cv2.normalize(dis, dis, 0, 1.0, cv2.NORM_MINMAX)
            dis_final[pos_index] = dis[pos_index]
        return dis_final
    def Gen_Dis_Map_onehot(self, label, max_flag=False):
        one_hot = self.Make_One_Hot(label,6)
        dis_final = np.zeros([one_hot.shape[0], label.shape[0], label.shape[1]]).astype(np.float32)
        for i in range(one_hot.shape[0]):
            pos_index = (label == i)
            dis = cv2.distanceTransform(one_hot[i], cv2.DIST_L2, 0)
            dis = cv2.normalize(dis, dis, 0, 1.0, cv2.NORM_MINMAX)
            dis_final[i][pos_index] = dis[pos_index]
        return dis_final
            

class pri_knowlegde(nn.Module):
    def __init__(self):
        super(pri_knowlegde,self).__init__()
        self.confuse_matrix = torch.zeros([5,5]).float()
        self.weight_matrix = torch.zeros([5,5]).float()

    def cal_pri_knowledge_weight(self, predict, label):
        
        dem_weight = torch.zeros([5]).float()
        ndvi_weight = torch.zeros([5]).float()
        dis_weight = torch.zeros([5]).float()
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
                    self.confuse_matrix[pre_class][label_class] = (pos_index.sum())

        zero_index = (self.confuse_matrix == 0)
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
        #      不透水面             -          DEM        NDVI      NDVI + DEM   (DIS) + DEM
        #      房屋                DEM          -       NDVI+DEM      NDVI      DEM
        #      低矮植被            NDVI      NDVI+DEM      -           DEM       NDVI
        #      高植被           NDVI + DEM     NDVI       DEM           -     NDVI + DEM
        #      车                  (DIS) + DEM         DEM    NDVI      NDVI + DEM     - 

        for pre_class in range(5):
            self.weight_matrix[pre_class] = self.confuse_matrix[pre_class]/(sum(self.confuse_matrix[pre_class])-self.confuse_matrix[pre_class][pre_class])
        self.weight_matrix[zero_index] = 0
        
        dem_weight[0] = self.weight_matrix[0][1] + self.weight_matrix[0][3] + self.weight_matrix[0][4] 
        # dem_weight[0] = self.weight_matrix[0][1] + self.weight_matrix[0][3]
        dis_weight[0] = self.weight_matrix[0][4]
        #dis_weight[0] = 0
        ndvi_weight[0] = self.weight_matrix[0][2] + self.weight_matrix[0][3]

        dem_weight[1] = self.weight_matrix[1][0] + self.weight_matrix[1][2] + self.weight_matrix[1][4]
        # #dis_weight[1] = self.weight_matrix[1][4]
        ndvi_weight[1] = self.weight_matrix[1][2] + self.weight_matrix[1][3]


        dem_weight[2] = self.weight_matrix[2][1] + self.weight_matrix[2][3]
        ndvi_weight[2] = self.weight_matrix[2][0] + self.weight_matrix[2][1] + self.weight_matrix[2][4]


        dem_weight[3] = self.weight_matrix[3][0] + self.weight_matrix[3][2] + self.weight_matrix[3][4]
        ndvi_weight[3] = self.weight_matrix[3][0] + self.weight_matrix[3][1] + self.weight_matrix[3][4]

        dem_weight[4] = self.weight_matrix[4][0] + self.weight_matrix[4][1] + self.weight_matrix[4][3]
        # #dis_weight[4] = self.weight_matrix[4][0] + self.weight_matrix[4][1]
        dis_weight[4] = self.weight_matrix[4][0]
        #dis_weight[4] = 0
        ndvi_weight[4] = self.weight_matrix[4][2] + self.weight_matrix[4][3]

        return dem_weight, dis_weight, ndvi_weight

if __name__ == "__main__":
    img = cv2.imread('C:\\Users\\ASUS\\Desktop\\320.png',-1)
    label = cv2.imread('C:\\Users\\ASUS\\Desktop\\32lab.png',-1)
    
    img = np.resize(img, (img.shape[0]*img.shape[1])).astype(np.uint8)
    label = np.resize(label, (label.shape[0]*label.shape[1])).astype(np.uint8)
        
    matrix = confusion_matrix(label, img)
    print(matrix)
    # kappa_score = cohen_kappa_score(target, predect)
    for i in range(3):
        a = pri_knowlegde()
        a.cal_pri_knowledge_weight(torch.from_numpy(img),torch.from_numpy(label))