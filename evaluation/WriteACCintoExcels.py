import os 
import shutil
import xlwt

def read_acc(txt_file):
    with open(txt_file) as f:
        lines = f.readlines()
        acc_line = lines[len(lines)-1]
        assert 'all' in acc_line
        acc = float(acc_line.split(':')[1])
    return acc

source_basefolder = 'result\\HyperSpectral\\Mlp_Auto\\conv2D_multiply\\抽样\\增加样本量\\ratio16，scale1_2_4_8，samesize=True\\' 
# source_basefolder = 'result\\HyperSpectral\\Mlp_Auto\\conv2D_multiply\\pt_img(增加样本量)\\gy_第二次（loss仅分类）\\' 
excel_path = 'C:\\Users\\25321\Desktop\\acc.xls'
workbook = xlwt.Workbook(encoding = 'utf-8')

# Read By Epoch
run_times = 4
# for dataset in ['Houston13','Houston18','Pavia','Salinas']:
for dataset in ['Pavia', 'Salinas']:
    name = dataset + '_Epoch'
    worksheet = workbook.add_sheet(name)
    for net in range(1,10,1):
        for times in range(1, run_times):
            # source_folder = source_basefolder + 'Net' + str(net) + '\\' + dataset + '\\' + str(times) + '\\'
            source_folder = source_basefolder + dataset + '\\' + str(times) + '\\'
            txt500_filename = source_folder + '500.txt'
            acc = read_acc(txt500_filename)
            worksheet.write(net, times, label = acc)
# workbook.save(excel_path)

# Read By Max
# for dataset in ['Houston13','Houston18','Pavia','Salinas']:
for dataset in ['Pavia', 'Salinas']:
    name = dataset + '_Max'
    worksheet = workbook.add_sheet(name)
    for net in range(1,10,1):
        for times in range(1, run_times):
            # source_folder = source_basefolder + 'Net' + str(net) + '\\' + dataset + '\\' + str(times) + '\\'
            source_folder = source_basefolder + dataset + '\\' + str(times) + '\\'
            acc_list = []
            for epoch in range(300,550,50):
                txt500_filename = source_folder + str(epoch) + '.txt'
                acc = read_acc(txt500_filename)
                acc_list.append(acc)
            acc_list.sort()
            worksheet.write(net, times, label = acc_list[-1])
workbook.save(excel_path)


'''
BaseLine 
'''
# source_basefolder = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\HyperSpectral\\BaseLine\\SA3D2D\\'
# excel_path = 'C:\\Users\\25321\Desktop\\acc.xls'
# workbook = xlwt.Workbook(encoding = 'utf-8')

# # Read By Epoch
# for dataset in ['Houston13','Houston18','Pavia']:
#     worksheet = workbook.add_sheet(dataset)
#     # for net in range(1, 10):
#     for times in range(1,6):
#         source_folder = source_basefolder + dataset + '\\' + str(times) + '\\'
#         txt500_filename = source_folder + '500.txt'
#         acc = read_acc(txt500_filename)
#         worksheet.write(1, times, label = acc)
# workbook.save(excel_path)


# # Read By Max
# for dataset in ['Houston13','Houston18','Pavia']:
#     worksheet = workbook.add_sheet(dataset)
#     for times in range(1, 6):
#         source_folder = source_basefolder + dataset + '\\' + str(times) + '\\'
#         acc_list = []
#         for epoch in range(200,550,50):
#             txt500_filename = source_folder + str(epoch) + '.txt'
#             acc = read_acc(txt500_filename)
#             acc_list.append(acc)
#         acc_list.sort()
#         worksheet.write(1, times, label = acc_list[-1])
# workbook.save(excel_path)