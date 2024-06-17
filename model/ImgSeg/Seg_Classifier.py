import sys
import numpy as np
import torch
from torch import nn
from torch.utils import data


base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from data_processing.Raster import gdal_read_tif, gdal_write_tif
from dataset.ImgSeg_dataset import ImgSeg_dataset
class MLP(nn.Module):
    def __init__(self, channel):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, 3, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(3, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

batch_size = 256
train_dst = ImgSeg_dataset('F:\\新建文件夹\\test\\feature.csv')
train_loader = data.DataLoader(train_dst, batch_size = batch_size, shuffle = True)
model = MLP(4).cuda()
# model = torch.load('F:\\新建文件夹\\test\\model_CE.pkl').cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer = torch.optim.SGD(params=[
        {'params': model.parameters(), 'lr': 0.01}
    ], lr=0.01, momentum=0.9, weight_decay=1e-4)
loss_function = nn.MSELoss()
# loss_function = nn.CrossEntropyLoss()
total_epoch = 20

cur_itrs = 0
cur_epochs = 0
while True: 
    optimizer.zero_grad()
    loss_epoch = 0
    for i, sample in enumerate(train_loader, 0):
        fv = sample['fv']
        lab = sample['lab']
        pred = model(fv)
        loss = loss_function(pred, lab.float())
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
        
    print('cur_epoch: %d, total_loss: %f'%(cur_epochs, loss_epoch))
    cur_epochs += 1
    print(pred, fv, lab)
    if cur_epochs > total_epoch:
        break
torch.save(model, 'F:\\新建文件夹\\test\\model.pkl')


# cur_step = 0
# class_len = [len(seg_dict[0]), len(seg_dict[1]), len(seg_dict[2]), len(seg_dict[3])]
# print(class_len)
# loss_sum = 0
# while(cur_step < iter_num):
#     optimizer.zero_grad()
#     cur_step+=1
#     train_segidx = cur_step%3==0
#     sample_idx1 = np.random.randint(0, class_len[train_segidx])
#     sample_idx2 = np.random.randint(0, class_len[train_segidx])
#     sample1 = torch.from_numpy(np.array(seg_dict[train_segidx][sample_idx1])).unsqueeze(0).float().cuda()
#     sample2 = torch.from_numpy(np.array(seg_dict[train_segidx][sample_idx2])).unsqueeze(0).float().cuda()
#     sample = torch.cat([sample1, sample2], 1)
#     pred = model(sample)
#     label = torch.ones_like(pred).cuda()
#     loss = loss_function(pred, label)
#     loss_sum+=loss.item()
#     loss.backward()
#     optimizer.step()
#     if(cur_step%10000==0):
#         print(cur_step, loss_sum)
#         loss_sum = 0
# torch.save(model, 'F:\\新建文件夹\\test\\model.pkl')
