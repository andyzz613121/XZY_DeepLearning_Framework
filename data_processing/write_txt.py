import random

def split_sample(img_folder, img_num, train_rate=1):
    '''
    划分数据集为训练集和测试集
    Input: 
        img_folder: 输入图像的文件夹，子文件夹应该包含'img'以及'lab'
        img_num: 图像的数量
        train_rate: 训练数据比例
    '''
    random_trainfile = img_folder + 'train_random.csv'
    random_testfile = img_folder + 'test_random.csv'

    #图像编号从0到3156
    num_list = list(range(1,img_num))
    #训练集比例1
    train_list = random.sample(num_list, int(len(num_list)*train_rate))
    test_list = [x for x in num_list if x not in train_list]

    with open(random_trainfile,'w') as train_csv:
        for item in train_list:
            # item = '%06d'%item
            str1 = img_folder + 'img\\' + str(item) + '.tif,' + img_folder + 'lab\\' + str(item) + '.tif\n'
            train_csv.write(str1)
    
    with open(random_testfile,'w') as test_csv:
        for item in test_list:
            # item = '%06d'%item
            str1 = img_folder + 'img\\' + str(item) + '.tif,' + img_folder + 'lab\\' + str(item) + '.tif\n'
            test_csv.write(str1)


if __name__ == '__main__':
    # # 图像数量
    # img_num = 3011
    # # 训练集比例1
    # train_rate = 0.6
    # # 文件夹名称
    # img_folder = 'E:\\dataset\\连云港GF2数据\\1_RPC+全色融合\\GF2_PMS1_E119.1_N34.2_20210730_L1A0005787958-pansharp1\\train_all_label\\'
    # # 样本划分
    # split_sample(img_folder, img_num, train_rate)
    import os
    img_folder = 'E:\\dataset\\ImageBlur\\Data\\train\\无模糊图像（复杂度+模糊半径）\\selected\\r_5\\train\\'
    with open(img_folder + '\\trainblur_r5.csv','w') as csv:
        for item in os.listdir(img_folder+'img\\'):
            str1 = img_folder + 'img\\' + item + ',' + img_folder + 'minspam\\' + item + ',' + 'E:\\dataset\\ImageBlur\\Data\\train\\无模糊图像（复杂度+模糊半径）\\selected\\r_5\\train\\lab\\lab.png\n'
            csv.write(str1)