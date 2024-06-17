from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
lab_folder = 'C:\\Users\\25321\\Desktop\\数据\\label\\'
# img_folder = 'C:\\Users\\25321\\Desktop\\数据\\pre_XPL\\'
img_folder = 'C:\\Users\\25321\\Desktop\\数据\\XPL_SC\\'
def read_labinfolder(folder):
  img_list = []
  for item in range(32):
    # print(item)
    img_path = folder + str(item) + 'lab.png'
    img = np.array(Image.open(img_path))
    # print(img.shape)
    img = np.reshape(img, [img.shape[0]*img.shape[1]])
    # print(img.shape)
    img_list = img_list + list(img)
  return img_list
def read_imginfolder(folder):
  img_list = []
  for item in range(32):
    # print(item)
    img_path = folder +'SC_' + str(item+1) + '.png'
    # img_path = folder +str(item) + 'pre.png'
    img = np.array(Image.open(img_path))
    # print(img.shape)
    img = np.reshape(img, [img.shape[0]*img.shape[1]])
    # print(img.shape)
    img_list = img_list + list(img)
  return img_list
label_list = read_labinfolder(lab_folder)
img_list = read_imginfolder(img_folder)
print(len(img_list))

y_pred = img_list
y_true = label_list
labelname = ['其它', '石英', '斜长石', '碱性长石', '角闪石', '黑云母', '绿泥石', '不透明矿物', '孔洞']

tick_marks = np.array(range(len(labelname))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # config = {
    #     "font.family": 'serif', # 衬线字体
    #     "font.size": 12, # 相当于小四大小
    #     "font.serif": ['SimSun'], # 宋体
    #     "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    #     'axes.unicode_minus': False # 处理负号，即-号
    # }
    # rcParams.update(config)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labelname)))
    plt.xticks(xlocations, labelname, rotation=90)
    plt.yticks(xlocations, labelname)
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print cm_normalized
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labelname))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.20)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix ')
# plt.text(4.5, 11.8, 'kappa = ' + str(kp), fontsize=11, ha='center', va='center')
# show confusion matrix
# plt.savefig('D:/confusion_matrix.png', format='png')
plt.show()