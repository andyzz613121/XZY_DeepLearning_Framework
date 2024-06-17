import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from visual.Draw import *

def read_lossfile(path):
    with open(path, 'rb') as text:
        f=text.read().decode('utf-8').split()
        epoch_list, loss_list = [], []
        for line in f:
            line = line.strip()
            epoch = int(line.split('poch')[1].split(',')[0])
            loss = float(line.split('loss')[1])
            epoch_list.append(epoch)
            loss_list.append(loss)
        return epoch_list, loss_list

if __name__ == '__main__':
    txt_path = 'C:\\Users\\25321\\Desktop\\111.csv'
    out_path = 'C:\\Users\\25321\\Desktop\\222.tif'
    epoch_list, loss_list = read_lossfile(txt_path)
    epoch_list, loss_list = epoch_list[:100], loss_list[:100]
    draw_curve(epoch_list, loss_list, out_path, xticks=[0, 25, 50, 75, 100], y_ticknum=5)
