import torch
import cv2
import os
import numpy as np
import torch.nn.functional as F
import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)
from PIL import Image
from osgeo import gdal
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from dataset.XZY_dataset import ISPRS_dataset
from model.PSPNet.PSPNet_Basic import PSPNet_AW
from model.SegNet.SegNet_dis import SegNet
from model.SegNet import SegNet_skeleton

class ISPRS_test_dataset():
    def __init__(self, dataset_name, image_path, dsm_path, ndsm_path, label_path):
        self.image_path = image_path
        self.dsm_path = dsm_path
        self.ndsm_path = ndsm_path
        self.label_path = label_path
        self.dataset_name = dataset_name
        assert self.dataset_name == 'Vaihingen' or self.dataset_name == 'Potsdam', "ERROR:: test_image//dataset"

        self.dataset = ISPRS_dataset(None, dataset_name, train_flag=False)
        self.sample = self.open_test_data(self.image_path, self.dsm_path, self.ndsm_path, self.label_path)
        self.img_h = self.sample['img'].shape[2]
        self.img_w = self.sample['img'].shape[3]
    def open_test_data(self, image_path, dsm_path, ndsm_path, label_path):
        return self.dataset.open_and_procress_data(self.dataset_name, image_path, dsm_path, ndsm_path, label_path, label_path)

class test_model():
    def __init__(self, dataset_name, class_num=6):
        self.dataset_name = dataset_name
        self.class_num = 6

    def test_image(self, model_img, model_addition, patch_size=256, run_time=3000):
        if self.dataset_name == 'Vaihingen':
            self.test_Vaihingen(model_img, model_addition, patch_size, run_time)
        elif self.dataset_name == 'Potsdam':
            self.test_Potsdam(model_img, model_addition, patch_size, run_time)
        else:
            print("Unknow dataset_name")

    def test_Vaihingen(self, model_img, model_addition, patch_size, run_time):
        print("---testing model in Vaihingen---")
        self.total_pixel = 0
        self.pos_pixel = 0
        for img_NO in (30,5,7,23):
        
            image_path = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai\\test\\big_image\\image' + str(img_NO) + '.tif'
            dsm_path = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai\\test\\big_dsm\\big_dsm' + str(img_NO) + '.tif'
            ndsm_path = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai\\test\\big_ndsm\\big_ndsm' + str(img_NO) + '.tif'
            label_path = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_vai\\label_gray\\label' + str(img_NO) + '_gray.tif'
            predict_path = 'result\\pre' + str(img_NO) + '_'

            test_dataset = ISPRS_test_dataset('Vaihingen', image_path,  dsm_path, ndsm_path, label_path)
            self.image_probability = np.zeros((test_dataset.img_h, test_dataset.img_w)).astype(np.float32)
            self.predect_count_list = []
            for classes in range(self.class_num):
                self.predect_count_list.append(np.zeros((test_dataset.img_h, test_dataset.img_w))) #记录每个类的个数
            #########################################################################
            total_num, pos_num = self.test_patches(test_dataset, img_NO, model_img, model_addition, predict_path, patch_size, run_time)
            # total_num, pos_num = self.test_total_image(test_dataset, img_NO, model_img, model_addition, predict_path)
            self.total_pixel += total_num
            self.pos_pixel += pos_num
        acc = self.pos_pixel/self.total_pixel
        print('AC is %f'%acc)

    def test_Potsdam(self, model_img, model_addition, patch_size, run_time):
        print("---testing model in Potsdam---")
        self.total_pixel = 0
        self.pos_pixel = 0
        image_folder = 'D:\\dataset\\Postdam\\4_Ortho_RGBIR\\'
        dsm_folder = 'D:\\dataset\\Postdam\\1_DSM\\'
        ndsm_folder = 'D:\\dataset\\Postdam\\1_DSM_normalisation\\'
        label_folder = 'D:\\Code\\LULC\\Hed_Seg\\data\\RS_image_paper_pos\\label_gray\\'
        for img_NO in ['2_11', '5_11', '4_10', '7_08']:
        # for img_NO in ['2_11']:
            image_path = image_folder + 'top_potsdam_' + img_NO + '_RGBIR.tif'
            dsm_path = dsm_folder + 'dsm_potsdam_0' + img_NO + '.tif'
            ndsm_path = ndsm_folder + 'dsm_potsdam_0' + img_NO + '_normalized_lastools.jpg'
            label_path = label_folder + 'label' + img_NO + '_gray.tif'
            predict_path = 'result\\pre' + str(img_NO) + '_'

            test_dataset = ISPRS_test_dataset('Potsdam', image_path,  dsm_path, ndsm_path, label_path)
            self.image_probability = np.zeros((test_dataset.img_h, test_dataset.img_w)).astype(np.float32)
            self.predect_count_list = []
            for classes in range(self.class_num):
                self.predect_count_list.append(np.zeros((test_dataset.img_h, test_dataset.img_w))) #记录每个类的个数
            #########################################################################
            total_num, pos_num = self.test_patches(test_dataset, img_NO, model_img, model_addition, predict_path, patch_size, run_time)
            self.total_pixel += total_num
            self.pos_pixel += pos_num
        acc = self.pos_pixel/self.total_pixel
        print('AC is %f'%acc)
    
    def test_patches(self, test_dataset, img_NO, model_img, model_addition, predict_path, patch_size, run_time):
        model_dis = SegNet(4, 1).cuda()
        model_dis.load_state_dict(torch.load('pretrained\\Dis\\Dis_Pos.pth'))
        # model_dis.load_state_dict(torch.load('pretrained\\Dis\\Dis_Vai.pth'))
        # model_dis.load_state_dict(torch.load('D:\\Code\\LULC\\Laplace\\result\\Dis_sigmoid\\image_model20.pkl'))
        model_dis.eval()
        img_model = model_img
        addition_model = model_addition
        img_model.eval()
        addition_model.eval()

        
        img_w = test_dataset.img_w
        img_h = test_dataset.img_h

        cur_h = 0
        while cur_h <= img_h: 
            start_h = cur_h 
            end_h = cur_h + patch_size
            if end_h >= img_h:
                end_h = img_h - 1
                start_h = end_h - patch_size
            cur_w = 0
            while cur_w <= img_w:
                start_w = cur_w
                end_w = cur_w + patch_size
                if end_w >= img_w:
                    end_w = img_w - 1
                    start_w = end_w - patch_size
                self.test_single_patch(test_dataset, start_h, end_h, start_w, end_w, img_model, addition_model, model_dis)

                cur_w+=int(patch_size/2)
            cur_h+=int(patch_size/2)

        for i in range(1000):
            start_h = np.random.randint(0, img_h-1)
            end_h = start_h + patch_size
            if end_h >= img_h:
                end_h = img_h - 1
                start_h = end_h - patch_size
            start_w = np.random.randint(0, img_w-1)
            end_w = start_w + patch_size
            if end_w >= img_w:
                end_w = img_w - 1
                start_w = end_w - patch_size
            self.test_single_patch(test_dataset, start_h, end_h, start_w, end_w, img_model, addition_model, model_dis)

        image_predect = np.array([self.predect_count_list[0], self.predect_count_list[1], self.predect_count_list[2], 
            self.predect_count_list[3], self.predect_count_list[4], self.predect_count_list[5]])
        image_predect1 = np.array([self.predect_count_list[i] for i in range(self.class_num)])    
        image_predect = np.argmax(image_predect, 0).astype(np.uint8)
        
        label = test_dataset.sample['label'].cpu().detach().numpy()
        pos_index = (label==image_predect)
        total_num = img_h * img_w
        self.total_pixel += total_num
        pos_num = pos_index.sum()
        self.pos_pixel += pos_num
        true_rate = pos_num/total_num
        print('image%s , acc is %f'%(str(img_NO), true_rate))
        # self.image_probability = Image.fromarray(self.image_probability)
        image_predect = Image.fromarray(image_predect)
        predict_path = predict_path + str(true_rate) + '.tif'
        image_predect.save(predict_path)
        return total_num, pos_num
        
    def test_total_image(self, test_dataset, img_NO, model_img, model_addition, predict_path):
        img_model = model_img
        addition_model = model_addition
        img_model.eval()
        addition_model.eval()
        
        model_dis = SegNet(3, 1).cuda()
        # model_dis = torch.load('D:\\Code\\LULC\\Laplace\\result\\Dis_Pos\\image_model50.pkl').cuda()
        model_dis.load_state_dict(torch.load('pretrained\\Dis\\Dis_Vai.pth'))
        model_dis.eval()

        img_w = test_dataset.img_w
        img_h = test_dataset.img_h
        w = None
        # AW_mlp = Weight_MLP.Auto_Weights(4, [25, 50, 25, 6])

        with torch.no_grad():
            sample = test_dataset.sample
            self.img = sample['img']
            self.dsm = sample['dsm']
            self.ndvi = sample['ndvi']
            self.label = sample['label']

            img = self.img
            dsm = self.dsm      
            ndvi = self.ndvi
            label = self.label
            dis = model_dis(img)
            if torch.max(dis)==torch.min(dis):
                if torch.max(dis)!=0:
                    dis = dis/torch.max(dis)
            else:
                dis = (dis-torch.min(dis))/(torch.max(dis)-torch.min(dis))
            dis = (dis - 0.5)/0.5

            inputs = torch.cat([img, dsm, dis, ndvi], 1)
            # outimg_list, outCM_list = img_model(inputs, inputs)
            # output = outimg_list[0][0].cpu().detach().numpy()
            x = inputs
            #encoder
            conv_1 = img_model.conv_1(x)
            conv_1_copy = conv_1
            conv_1, index_1 = img_model.pool(conv_1)

            conv_2 = img_model.conv_2(conv_1)
            conv_2_copy = conv_2
            conv_2, index_2 = img_model.pool(conv_2)

            conv_3 = img_model.conv_3(conv_2)
            conv_3_copy = conv_3
            conv_3, index_3 = img_model.pool(conv_3)

            conv_4 = img_model.conv_4(conv_3)
            conv_4_copy = conv_4
            conv_4, index_4 = img_model.pool(conv_4)

            conv_5 = img_model.conv_5(conv_4)
            conv_5_copy = conv_5
            conv_5, index_5 = img_model.pool(conv_5)

            #decoder
            deconv_5 = img_model.MSA5(conv_5)
            deconv_5 = img_model.unpool(deconv_5,index_5,output_size=conv_5_copy.shape)
            deconv_5 = img_model.deconv_5(deconv_5)
            deconv_5_sideout = img_model.Sideout5(deconv_5)
            deconv_5_sideout = F.interpolate(deconv_5_sideout, size=(x.shape[2], x.shape[3]), mode='bilinear')
            pre_5 = torch.argmax(deconv_5_sideout, 1)

            CM2 = torch.tensor([[1.4851e+05, 7.2789e+02, 1.4851e+03, 1.2285e+03, 4.1411e+02, 4.7202e+02],
        [8.2692e+02, 1.3880e+05, 5.5759e+02, 2.0508e+02, 4.6006e+00, 1.7920e+02],
        [1.6558e+03, 5.1929e+02, 1.1679e+05, 2.9521e+03, 1.6707e+01, 5.0403e+02],
        [1.5676e+03, 2.4722e+02, 3.3711e+03, 6.8248e+04, 8.8354e+01, 1.4968e+02],
        [6.5245e+02, 7.3725e+00, 2.4830e+01, 1.0110e+02, 8.4022e+03, 1.9985e+01],
        [6.0371e+02, 2.2208e+02, 5.8673e+02, 1.6281e+02, 1.9369e+01, 2.3920e+04]]).cuda()
            CM3 = torch.tensor([[1.4847e+05, 7.3441e+02, 1.4957e+03, 1.2205e+03, 4.1862e+02, 4.8229e+02],
        [8.3023e+02, 1.3879e+05, 5.6839e+02, 2.0471e+02, 4.9723e+00, 1.8098e+02],
        [1.6791e+03, 5.2134e+02, 1.1676e+05, 2.9441e+03, 1.7308e+01, 5.0867e+02],
        [1.5781e+03, 2.4829e+02, 3.3794e+03, 6.8272e+04, 8.8480e+01, 1.5459e+02],
        [6.5940e+02, 6.7812e+00, 2.3697e+01, 9.9752e+01, 8.3956e+03, 1.8988e+01],
        [5.9645e+02, 2.1781e+02, 5.8464e+02, 1.5611e+02, 2.0327e+01, 2.3900e+04]]).cuda()
            CM4 = torch.tensor([[1.4789e+05, 8.3742e+02, 1.7313e+03, 1.2692e+03, 4.9410e+02, 5.7954e+02],
        [9.4837e+02, 1.3858e+05, 6.5336e+02, 2.1990e+02, 5.3038e+00, 2.0675e+02],
        [1.9238e+03, 6.0405e+02, 1.1628e+05, 3.0416e+03, 2.0419e+01, 6.0422e+02],
        [1.6229e+03, 2.6332e+02, 3.4641e+03, 6.8103e+04, 8.9521e+01, 1.6538e+02],
        [7.4977e+02, 7.0464e+00, 2.5553e+01, 1.0350e+02, 8.3146e+03, 1.9776e+01],
        [6.7312e+02, 2.3427e+02, 6.5886e+02, 1.5983e+02, 2.1376e+01, 2.3670e+04]]).cuda()
            CM5 = torch.tensor([[1.4639e+05, 1.0989e+03, 2.3408e+03, 1.4122e+03, 7.0035e+02, 8.6360e+02],
        [1.2486e+03, 1.3803e+05, 8.2580e+02, 2.6855e+02, 8.6352e+00, 2.8818e+02],
        [2.5352e+03, 7.8224e+02, 1.1494e+05, 3.5178e+03, 3.2322e+01, 8.7239e+02],
        [1.7837e+03, 3.0526e+02, 3.8552e+03, 6.7420e+04, 1.0443e+02, 2.0713e+02],
        [9.8580e+02, 9.6172e+00, 3.4577e+01, 1.0800e+02, 8.0685e+03, 2.9055e+01],
        [8.6627e+02, 2.9466e+02, 8.1930e+02, 1.7027e+02, 3.1160e+01, 2.2985e+04]]).cuda()

            CMA_4 = img_model.CWA5(deconv_5, CM5)
            MSA_4 = img_model.MSA4(conv_4)
            deconv_4 = torch.cat([CMA_4, MSA_4], 1)
            deconv_4 = img_model.MSA_CWA_Fuse4(deconv_4)
            deconv_4 = img_model.unpool(deconv_4,index_4,output_size=conv_4_copy.shape)
            deconv_4 = img_model.deconv_4(deconv_4)
            deconv_4_sideout = F.interpolate(deconv_4, size=(x.shape[2], x.shape[3]), mode='bilinear')
            pre_4 = torch.argmax(deconv_4_sideout, 1)
            

            CMA_3 = img_model.CWA4(deconv_4, CM4)
            MSA_3 = img_model.MSA3(conv_3)
            deconv_3 = torch.cat([CMA_3, MSA_3], 1)
            deconv_3 = img_model.MSA_CWA_Fuse3(deconv_3)
            deconv_3 = img_model.unpool(deconv_3,index_3,output_size=conv_3_copy.shape)
            deconv_3 = img_model.deconv_3(deconv_3)
            deconv_3_sideout = F.interpolate(deconv_3, size=(x.shape[2], x.shape[3]), mode='bilinear')
            pre_3 = torch.argmax(deconv_3_sideout, 1)
            

            CMA_2 = img_model.CWA3(deconv_3, CM3)
            MSA_2 = img_model.MSA2(conv_2)
            deconv_2 = torch.cat([CMA_2, MSA_2], 1)
            deconv_2 = img_model.MSA_CWA_Fuse2(deconv_2)
            deconv_2 = img_model.unpool(deconv_2,index_2,output_size=conv_2_copy.shape)
            deconv_2 = img_model.deconv_2(deconv_2)
            deconv_2_sideout = F.interpolate(deconv_2, size=(x.shape[2], x.shape[3]), mode='bilinear')
            pre_2 = torch.argmax(deconv_2_sideout, 1)


            CMA_1 = img_model.CWA2(deconv_2, CM2)
            MSA_1 = img_model.MSA1(conv_1)
            deconv_1 = torch.cat([CMA_1, MSA_1], 1)
            deconv_1 = img_model.MSA_CWA_Fuse1(deconv_1)
            deconv_1 = img_model.unpool(deconv_1,index_1,output_size=conv_1_copy.shape)
            deconv_1 = img_model.deconv_1(deconv_1)
            print((pre_5.shape))
            output = deconv_1[0].cpu().detach().numpy()

        image_predect = np.argmax(output, 0).astype(np.uint8)
        label = label.cpu().detach().numpy()
        pos_index = (label==image_predect)
        total_num = img_h * img_w
        self.total_pixel += total_num
        pos_num = pos_index.sum()
        self.pos_pixel += pos_num
        true_rate = pos_num/total_num
        print('image%s , acc is %f'%(str(img_NO), true_rate))
        image_predect = Image.fromarray(image_predect)
        predict_path = predict_path + str(true_rate) + '.tif'
        image_predect.save(predict_path)
        return total_num, pos_num

    def test_single_patch(self, test_dataset, start_h, end_h, start_w, end_w, model_img, model_addition, model_dis):
        with torch.no_grad():
            sample = test_dataset.sample
            self.img = sample['img']
            self.dsm = sample['dsm']
            self.ndvi = sample['ndvi']
            self.label = sample['label']
            
            img_patch = self.img[:, :, start_h:end_h, start_w:end_w]
            dsm_patch = self.dsm[:, :, start_h:end_h, start_w:end_w]
            ndvi_patch = self.ndvi[:, :, start_h:end_h, start_w:end_w]

            # dis_patch = model_dis(img_patch)

            # # driver = gdal.GetDriverByName('GTiff')
            # # dataset = driver.Create("C:\\Users\\ASUS\\Desktop\\2.tif", 256, 256, 1, gdal.GDT_Float32)
            # # dataset.GetRasterBand(1).WriteArray(dis_patch[0][0].cpu().numpy())

            # if torch.max(dis_patch)==torch.min(dis_patch):
            #     if torch.max(dis_patch)!=0:
            #         dis_patch = dis_patch/torch.max(dis_patch)
            # else:
            #     dis_patch = (dis_patch-torch.min(dis_patch))/(torch.max(dis_patch)-torch.min(dis_patch))
            
            # dis_patch = (dis_patch - 0.5)/0.5
        #     CM = torch.tensor([[0.0000e+00, 2.7676e-01, 4.2650e-01, 4.1396e-01, 2.3732e-01, 9.2217e-03],
        # [3.4178e-01, 0.0000e+00, 3.0709e-01, 1.3014e-01, 5.3354e-03, 3.9408e-03],
        # [4.6157e-01, 3.0308e-01, 0.0000e+00, 9.7779e-01, 5.9302e-03, 9.7456e-04],
        # [4.9370e-01, 1.3768e-01, 1.0000e+00, 0.0000e+00, 1.0768e-02, 1.2189e-02],
        # [3.3726e-01, 5.3631e-03, 4.3315e-03, 4.6560e-03, 0.0000e+00, 5.3284e-03],
        # [8.5881e-03, 1.7178e-03, 7.9324e-04, 1.3374e-02, 1.6254e-03, 0.0000e+00]]).cuda()
            inputs = torch.cat([img_patch, dsm_patch, ndvi_patch], 1)
            output = model_img(inputs)
            # print(model_img.T1, model_img.a, model_img.b)
        #     # outimg_list, outCM_list = img_model(inputs, inputs)
        #     # output = outimg_list[0][0].cpu().detach().numpy()
        #     x = inputs
        #     #encoder

        #     CM1 = torch.tensor([[8.3836e-01, 4.9593e-03, 7.6171e-03, 8.6747e-03, 3.3146e-03, 2.3399e-04],
        # [5.4679e-03, 7.3266e-01, 5.3234e-03, 2.4674e-03, 5.5246e-05, 3.1886e-05],
        # [8.3037e-03, 5.4453e-03, 6.3980e-01, 1.8213e-02, 1.6021e-04, 2.6755e-05],
        # [9.7973e-03, 2.5309e-03, 1.8544e-02, 6.9105e-01, 1.3463e-04, 2.5853e-04],
        # [4.6658e-03, 9.4976e-05, 1.3042e-04, 1.4449e-04, 3.4076e-02, 1.9074e-05],
        # [2.0176e-04, 2.8454e-05, 2.2081e-05, 2.9788e-04, 2.4031e-06, 2.7478e-02]],).cuda()
        #     CM2 = torch.tensor([[8.3887e-01, 5.0799e-03, 7.8510e-03, 8.7563e-03, 3.4463e-03, 2.0639e-04],
        # [5.4347e-03, 7.3260e-01, 5.3900e-03, 2.5178e-03, 6.1048e-05, 2.1248e-05],
        # [7.9751e-03, 5.4446e-03, 6.3950e-01, 1.7990e-02, 1.5460e-04, 2.1989e-05],
        # [9.7113e-03, 2.5577e-03, 1.8685e-02, 6.9127e-01, 1.4677e-04, 2.5219e-04],
        # [4.6302e-03, 8.3680e-05, 1.4563e-04, 1.3815e-04, 3.3933e-02, 1.3519e-05],
        # [2.2051e-04, 2.3290e-05, 3.0254e-05, 2.8678e-04, 1.4348e-06, 2.7532e-02]]).cuda()
        #     CM3 = torch.tensor([[8.3847e-01, 5.6330e-03, 8.5812e-03, 9.3219e-03, 3.9665e-03, 2.2800e-04],
        # [6.1626e-03, 7.3325e-01, 6.0707e-03, 2.8599e-03, 7.2454e-05, 2.8579e-05],
        # [8.9474e-03, 6.1792e-03, 6.3958e-01, 1.8785e-02, 1.9459e-04, 3.4048e-05],
        # [1.0092e-02, 2.8568e-03, 1.9081e-02, 6.9162e-01, 1.6773e-04, 2.6932e-04],
        # [5.7605e-03, 9.5373e-05, 1.8205e-04, 1.6911e-04, 3.3476e-02, 2.1724e-05],
        # [2.4367e-04, 2.2676e-05, 3.8629e-05, 2.9639e-04, 7.2212e-07, 2.7544e-02]]).cuda()
        #     CM4 = torch.tensor([[8.3741e-01, 7.5512e-03, 1.1655e-02, 1.2112e-02, 6.1838e-03, 4.4195e-04],
        # [8.5412e-03, 7.3645e-01, 8.3828e-03, 3.9153e-03, 1.2204e-04, 4.7831e-05],
        # [1.2022e-02, 8.4542e-03, 6.3774e-01, 2.3210e-02, 3.2399e-04, 6.0757e-05],
        # [1.3070e-02, 3.9732e-03, 2.3316e-02, 6.9158e-01, 2.6608e-04, 5.2122e-04],
        # [8.7795e-03, 1.4386e-04, 3.1534e-04, 2.4076e-04, 3.1451e-02, 1.7919e-05],
        # [3.5208e-04, 2.6598e-05, 5.1682e-05, 3.9880e-04, 7.5653e-07, 2.7323e-02]]).cuda()

        #     x = model_img.resnet.conv1(x)
        #     x = model_img.resnet.bn1(x)
        #     x = model_img.resnet.relu(x)
        #     x_unpool = x
        #     x = model_img.resnet.maxpool(x)      #64

        #     e1 = model_img.encoder1(x)           #256
        #     e2 = model_img.encoder2(e1)          #512
        #     e3 = model_img.encoder3(e2)          #1024
        #     e4 = model_img.encoder4(e3)          #2048

        #     d4_in = model_img.MSA4(e4)
        #     d4_out = model_img.decoder4(d4_in)                       
        #     d4 = model_img.CWA4(d4_out, CM4) + model_img.MSA3(e3)

        #     d3_in = d4
        #     d3_out = model_img.decoder3(d3_in)                       
        #     d3 = model_img.CWA3(d3_out, CM3) + model_img.MSA2(e2)

        #     d2_in = d3
        #     d2_out = model_img.decoder2(d2_in)                       
        #     d2 = model_img.CWA2(d2_out, CM2) + model_img.MSA1(e1)

        #     d1_in = d2
        #     d1_out = model_img.decoder1(d1_in)                       
        #     d1 = model_img.CWA1(d1_out, CM1) + model_img.MSA0(x_unpool)
            
        #     out = model_img.classifier(d1)
        #     out0 = model_img.Pre_0(out)
        #     out1 = model_img.Pre_1(out)  
        #     out2 = model_img.Pre_2(out)  
        #     out3 = model_img.Pre_3(out)  
        #     out4 = model_img.Pre_4(out)  
        #     out5 = model_img.Pre_5(out)  
        #     output = torch.cat([out0, out1, out2, out3, out4, out5], 1)

            max_probability, max_index = torch.max(output, 1)
            max_probability = max_probability.cpu().detach().numpy()
            max_index = max_index.cpu().detach().numpy()
            max_probability = max_probability.reshape((max_probability.shape[1], max_probability.shape[2]))
            max_index = max_index.reshape((max_index.shape[1], max_index.shape[2]))

            big_pb_index = (self.image_probability[start_h:end_h, start_w:end_w] < max_probability)
            self.image_probability[start_h:end_h, start_w:end_w][big_pb_index] = max_probability[big_pb_index]

            for classes in range(6):
                class_index = (max_index == classes)
                self.predect_count_list[classes][start_h:end_h, start_w:end_w][class_index] += 1


if __name__ == "__main__":
    
    for i in range(200, 195, -1):
        mlp_path= '..\\XZY_DeepLearning_Framework\\result\\image_model' +str(i)+'.pkl'
        img_path = '..\\XZY_DeepLearning_Framework\\result\\image_model' +str(i)+'.pkl'
        img_model = torch.load(img_path).cuda()
        print(img_model.T1, img_model.spectral_list)
        mlp_model = torch.load(mlp_path).cuda()
        test = test_model('Vaihingen', 6)
        # test = test_model('Potsdam', 6)
        test.test_image(img_model, mlp_model, run_time=0)

