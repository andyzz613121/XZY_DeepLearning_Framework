
import sys
import time
import numpy as np

from torch.utils import data
import torchvision.transforms as transforms
import torch
import torch.nn as nn

base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from dataset.HS_dataset_new import HS_dataset
from model.HyperSpectral.CMAttention.SpectralImage_CM import HS_SI_CM
from model.Self_Module.CAM import Grad_CAM, overlay_mask, CAM
from testing.HS.test_HS import testHS
from PIL import Image
from visual.Draw import draw_curve


def draw_curve_cam(pt_img, path):
    pt_imgnp = pt_img.view(1, pt_img.shape[0]*pt_img.shape[1]).cpu().detach().numpy()[0]
    pt_flat = pt_imgnp
    x = np.array([x for x in range(pt_flat.shape[0])])
    draw_curve(x, pt_flat, path)

def main():
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    batch_size = 32
    init_lr = 0.001
    PCA = False
    norm = True

    dataset = 'Pavia'

    if dataset == 'Pavia':
        input_channels = 103
        output_channels = 9
    elif dataset == 'Houston13':
        input_channels = 144
        output_channels = 15
    elif dataset == 'Houston18':
        input_channels = 48
        output_channels = 20
    elif dataset == 'Salinas':
        input_channels = 204
        output_channels = 16

    if PCA == True:
        input_channels = 30

    train_dst = HS_dataset(dataset, PCA, norm)
    train_loader = data.DataLoader(
        train_dst, batch_size = batch_size, shuffle = True)

    model_img = HS_SI_CM(input_channels, output_channels).cuda()
    # model_img = torch.load('D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\HyperSpectral\\Pavia\\pt_sample\\Conv2D+3D卷积711+fc2层\\image_model500.pkl').cuda()
    model_CAM = CAM()
    
    optimizer = torch.optim.SGD(params=[
        {'params': model_img.parameters(), 'lr': 1*init_lr}
    ], lr=init_lr, momentum=0.9, weight_decay=1e-4)

    loss_function = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    # Restore
    total_epoch = 500
    cur_itrs = 0
    cur_epochs = 0
    class_CAM_num = np.zeros([output_channels])  # 预测的时候每个类的CAM有几个
    #==========   Train Loop   ==========#
    while True: #cur_itrs < opts.total_itrs:
        cur_epochs += 1
        interval_loss = 0
        model_img.train()

        for _, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            cur_itrs += 1
            
            handle1 = model_img.SI_conv_3.register_forward_hook(model_CAM.forward_hook)
            handle2 = model_img.SI_conv_3.register_backward_hook(model_CAM.backward_hook)

            img=sample['img']
            label=sample['label']

            out_list, pt = model_img(img)
            
            loss = 0
            for outs in out_list:
                loss += loss_function(outs, label.long())

            loss.backward()

            cams = model_CAM.compute_CAM_batch()
            handle1.remove()
            handle2.remove()
            # print(len(model_CAM.feats_list), len(model_CAM.grads_list))
            interval_loss += loss.item()
            optimizer.step()

            out = out_list[0]
            pre = torch.argmax(out, 1)
            for b in range(model_img.SI.shape[0]):
                pre_b = pre[b].cpu().detach().numpy()
                lab_b = label[b].cpu().detach().numpy()
                out_PIL = model_img.SI[b].cpu().detach().numpy()*255
                out_PIL = Image.fromarray(out_PIL.astype(np.uint8))
                cams_PIL = Image.fromarray(cams[b])
                
                cam_path = 'result\\新建文件夹\\' + str(lab_b) + '\\SI\\' + 'cam_pre' + str(pre_b) + '_No_' + str(int(class_CAM_num[lab_b])) +'.tif'
                cams_PIL.save(cam_path)
                
                pt_b = pt[b][0]
                curve_path = 'result\\新建文件夹\\' + str(lab_b) + '\\SP\\' + 'curve_pre' + str(pre_b) + '_No_' + str(int(class_CAM_num[lab_b])) +'.tif'
                draw_curve_cam(pt_b, curve_path)
                class_CAM_num[lab_b] += 1

        scheduler.step()
    
        print('--------epoch %d done --------'%cur_epochs)
        print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        print('loss: %f' %(interval_loss))
        print('lr: %f', optimizer.param_groups[0]['lr'])
        
        if cur_epochs in [1, 200, 250, 300, 350, 400, 450, 500]:
            image_model_name = 'result\\image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
            model_img.eval()

            out = out_list[0]
            pre = torch.argmax(out, 1)
    
            for b in range(model_img.SI.shape[0]):
                pre_b = pre[b].cpu().detach().numpy()
                out_PIL = model_img.SI[b].cpu().detach().numpy()*255
                out_PIL = Image.fromarray(out_PIL.astype(np.uint8))
                cams_PIL = Image.fromarray(cams[b])
                # cams_mask = overlay_mask(out_PIL, cams_PIL)
                cam_path = 'result\\cam_class' + str(pre_b) + '_epoch_' + str(cur_epochs) +'.tif'
                # cam_path = 'result\\cam' + '_Class' + str(pre_b) + 'num' + str(class_CAM_num[pre_b]) + '.tif'
                # cam_path_mask = 'result\\cam_mask' + str(b) + '.tif'
                cams_PIL.save(cam_path)
                class_CAM_num[pre_b] += 1
                # cams_mask.save(cam_path_mask)

            path = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\' + str(cur_epochs)
            testHS(dataset, model_img, path, norm, PCA)

        if cur_epochs >= total_epoch:
            break

        
if __name__ == '__main__':
    main()
