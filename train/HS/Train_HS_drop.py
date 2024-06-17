
import sys
import time
import numpy as np

from torch.utils import data
import torchvision.transforms as transforms
import torch
import torch.nn as nn

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from dataset.HS_dataset import HS_dataset
from model.HyperSpectral.SpectralImage_drop import HS_SI_drop
from testing.HS.test_HS import testHS

def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 32
    init_lr = 0.001

    # dataset = 'Pavia'
    # input_channels = 103
    # output_channels = 9
    # train_dst = HS_dataset('E:\\dataset\\高光谱数据集\\Pavia\\Train\\data\\label.csv')

    dataset = 'Houston'
    input_channels = 144
    output_channels = 15
    train_dst = HS_dataset('E:\\dataset\\高光谱数据集\\2013_DFTC\\2013_DFTC\\Train\\data\\label.csv')

    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    model_img = HS_SI_drop(input_channels, output_channels).cuda()

    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': 1*init_lr}
    ], lr=init_lr, momentum=0.9, weight_decay=1e-4)
    
    loss_function = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    # Restore
    total_epoch = 400
    cur_itrs = 0
    cur_epochs = 0

    
    #==========   Train Loop   ==========#
    while True: #cur_itrs < opts.total_itrs:
        cur_epochs += 1
        interval_loss = 0

        true = 0
        total = 0
        model_img.train()
        for i, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            cur_itrs += 1

            img=sample['img']
            label=sample['label']
            out_list = model_img(img)
            out_x, out_drop = out_list[0], out_list[1]

            loss_x = loss_function(out_x, label.long())
            loss_drop = loss_function(out_drop, label.long())
            if loss_x > loss_drop:
                loss = loss_drop
                model_img.update_w()
            else:
                loss = loss_x
            loss.backward()
            interval_loss += loss.item()
            optimizer.step()

            total += img.shape[0]
            pre = torch.argmax(out_x, 1)
            true_index = (pre == label)
            true += true_index.sum()
            
        scheduler.step()
        print('--------epoch %d done --------'%cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        print('model_img.w', model_img.w)
        print('model_img.softmax(model_img.w)', model_img.softmax(model_img.w)*input_channels)
        print('acc:', true/total)

        if cur_epochs in [200, 250, 300, 350, 400]:
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
            model_img.eval()
            path = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\' + str(cur_epochs)
            testHS(dataset, model_img, path)

        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()
