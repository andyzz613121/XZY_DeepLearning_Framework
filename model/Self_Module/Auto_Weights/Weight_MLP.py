from ast import If
import torch
import numpy as np
from torch import nn
def cal_confuse_matrix_onlyfalse(predict, label):
        pre_pos_list = [] #predict等于各个类的下标数组
        label_pos_list = [] #label等于各个类的下标数组
        confuse_matrix = torch.zeros([6,6]).float().cuda()
        # label = label[:,0,:,:]
        for pre_class in range(6):
            pos_index = (predict == pre_class)
            pre_pos_list.append(pos_index)
        for label_class in range(6):
            pos_index = (label == label_class)
            label_pos_list.append(pos_index)
        
        for pre_class in range(6):
            for label_class in range(6):
                if pre_class != label_class:
                    pos_index = pre_pos_list[pre_class]*label_pos_list[label_class]
                    confuse_matrix[pre_class][label_class] = (pos_index.sum())
        return  confuse_matrix/torch.max(confuse_matrix)

def cal_confuse_matrix(predict, label, class_num):
    pre_pos_list = [] #predict等于各个类的下标数组
    label_pos_list = [] #label等于各个类的下标数组
    confuse_matrix = torch.zeros([class_num,class_num]).float().cuda()
    # label = label[:,0,:,:]
    for pre_class in range(class_num):
        pos_index = (predict == pre_class)
        pre_pos_list.append(pos_index)
    for label_class in range(class_num):
        pos_index = (label == label_class)
        label_pos_list.append(pos_index)
    
    for pre_class in range(class_num):
        for label_class in range(class_num):
            if pre_class != label_class:
                pos_index = pre_pos_list[pre_class]*label_pos_list[label_class]
                confuse_matrix[pre_class][label_class] = (pos_index.sum())
    return  confuse_matrix

class Auto_Weights():
    def __init__(self, MLP_Layernum, MLP_nodelist):
        
        self.MLP = Weight_MLP(MLP_Layernum, MLP_nodelist)

    def cal_confuse_matrix(self, predict, label):
        pre_pos_list = [] #predict等于各个类的下标数组
        label_pos_list = [] #label等于各个类的下标数组
        confuse_matrix = np.zeros([5,5]).astype(np.float32)
        # label = label[:,0,:,:]
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
        return  confuse_matrix/np.max(confuse_matrix)

    def cal_weights(self, predict, label):
        confuse_matrix = self.cal_confuse_matrix(predict, label)
        confuse_matrix = torch.from_numpy(confuse_matrix)
        return self.MLP(confuse_matrix)

class Weight_MLP(nn.Module):
    def __init__(self, layer_num, node_list):
        super().__init__()
        assert layer_num == len(node_list), "ERROR at Weight_MLP: Layer_num != len(node_list)"
        self.MLP = self.get_mlp(layer_num, node_list)
        print(self.MLP)

    def get_mlp(self, layer_num, node_list, drop_rate=0.2):
        layers = []
        for layer in range(layer_num-1):
            layers.append(nn.Linear(node_list[layer], node_list[layer+1]))
            if layer+1 != (layer_num-1):  #Last layer
                layers.append(nn.Dropout(drop_rate))
                layers.append(nn.ReLU())
        mlp = nn.Sequential(*layers)
        for m in mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        return mlp

    def forward(self, confuse_matrix):
        # print((confuse_matrix))
        confuse_matrix_flatten = torch.reshape(confuse_matrix, (1, -1))
        # print((confuse_matrix_flatten.shape))
        return self.MLP(confuse_matrix_flatten)

if __name__ == "__main__":
    # from sklearn.metrics import confusion_matrix
    from osgeo import gdal
    img_path = 'C:\\Users\\admin\\Desktop\\Laplace\\result\\pre5_0.8769957009852107.tif'
    img_raw = gdal.Open(img_path)
    img_w = img_raw.RasterXSize
    img_h = img_raw.RasterYSize
    label_path = 'C:\\Users\\admin\\Desktop\\RS_image_paper_vai\\label_gray\\label5_gray.tif'
    label_raw = gdal.Open(label_path)
    
    img = np.array(img_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('uint8')
    label = np.array(label_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('uint8')
    # print(img.shape, label.shape)
    # img1 = np.reshape(img, [-1])
    # label1 = np.reshape(label, [-1])
    # print(img1.shape, label1.shape)
    # print(confusion_matrix(label1, img1))
    img = torch.from_numpy(img)
    label = torch.from_numpy(label)
    A_W = Auto_Weights(4, [25, 50, 25, 6])
    w = A_W.cal_weights(img, label)
    print(w)