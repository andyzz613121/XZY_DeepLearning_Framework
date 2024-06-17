import os
def writecsv_number(img_num, data_folder, csv_name='', img_name_list=[]):
    '''
        Input: img_num（文件夹中的图像总数）
               data_folder（存数据的文件夹，包含img和label文件夹）
        Output:
               在data_folder文件夹下面输出csv
    '''
    train_list = list(range(img_num))
    with open(data_folder + csv_name + '.csv','w') as train_csv:
        for item in train_list:
            str1 = ''
            for name_idx in range(len(img_name_list)):
                str1 += data_folder + img_name_list[name_idx] + '\\' + str(item)
                if name_idx == len(img_name_list) - 1:
                    str1 += '.tif\n'
                else:
                    str1 += '.tif,'
            train_csv.write(str1)

def writecsv_name(data_folder, csv_name='', img_name_list=[]):
    '''
        Input: img_num（文件夹中的图像总数）
               data_folder（存数据的文件夹，包含img和label文件夹）
        Output:
               在data_folder文件夹下面输出csv
    '''

    with open(data_folder + csv_name + '.csv','w') as train_csv:
        for item in os.listdir(data_folder+'img'):
            img_name = item.split('.')[0]
            str1 = ''
            for name_idx in range(len(img_name_list)): 
                str1 += data_folder + img_name_list[name_idx] + '\\' + item
                if name_idx == len(img_name_list) - 1:
                    str1 += '\n'
                else:
                    str1 += ','
            train_csv.write(str1)

if __name__ == '__main__':
    # writecsv_number(1333, 'E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\Train\\9.12\\', 'train_edge', ['img', 'edge', 'lab'])
    # writecsv_number(1333, 'E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\Train\\9.12\\', 'train_edge_dilate', ['img', 'edge_dilate', 'lab'])
    writecsv_name('E:\\dataset\\ImageBlur\\Data\\train\\训练图像\\test\\', 'new_testblur_rgb', ['img', 'minspam_gray', 'label'])
