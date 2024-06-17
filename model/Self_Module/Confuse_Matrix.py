import torch

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