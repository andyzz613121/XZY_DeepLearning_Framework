import torch
import torch.utils.data.dataset as Dataset
import os
import numpy as np


class ImgSeg_dataset(Dataset.Dataset):
    def __init__(self, csv_dir, gpu=True):
        self.csv_dir = csv_dir          
        self.names_list = []
        self.size = 0
        self.gpu = gpu
        if not os.path.isfile(self.csv_dir):
            print(self.csv_dir + ':txt file does not exist!')

        file = open(self.csv_dir)
        
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        fv = []
        fv_read = self.names_list[idx].split(',')
        for i in range(len(fv_read)-1):
            fv.append(float(fv_read[i]))
        lab = int(fv_read[-1])
        fv = torch.from_numpy(np.array(fv)).float()
        lab = torch.from_numpy(np.array(lab)).long()

        if self.gpu == True:
            fv = fv.cuda()
            lab = lab.cuda()

        sample = {'fv':fv, 'lab': lab}
        return sample

if __name__ == '__main__':
    from torch.utils import data
    train_dst = ImgSeg_dataset('F:\\新建文件夹\\test\\feature.csv')
    train_loader = data.DataLoader(
        train_dst, batch_size = 1, shuffle = False)
    for i, sample in enumerate(train_loader, 0):
        print(sample)