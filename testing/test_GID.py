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


from dataset.XZY_dataset import ISPRS_dataset, RS_dataset
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
    def __init__(self, dataset_name, class_num):
        self.dataset_name = dataset_name
        self.class_num = class_num

    def test_image(self, model_imgs, patch_size=256, run_time=3000):
        if self.dataset_name == 'Vaihingen':
            self.test_Vaihingen(model_imgs[0], model_imgs[1], patch_size, run_time)
        elif self.dataset_name == 'Potsdam':
            self.test_Potsdam(model_imgs[0], model_imgs[1], patch_size, run_time)
        elif self.dataset_name == 'GID':
            self.test_GID(model_imgs, patch_size, run_time)
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
    
    def test_GID(self, model_img, patch_size, run_time):
        print("---testing model in GID---")
        self.total_pixel = np.zeros([1])
        self.pos_pixel = np.zeros([1])
        base_folder = 'E:\\dataset\\GID数据集\\test\\'
        img_folder = base_folder + 'big_img\\'
        lab_folder = base_folder + 'big_label\\'

        test_dataset = RS_dataset(train_flag = False)
        for item in os.listdir(img_folder):
            img_name = item.split('.')[0]
            image_path = img_folder + item
            label_path = lab_folder + img_name + '_label.tif'
            
            
            sample = test_dataset.open_and_procress_data(image_path, label_path)
            img_h = sample['img'].shape[2]
            img_w = sample['img'].shape[3]
            self.image_probability = np.zeros((img_h, img_w)).astype(np.float32)
            self.predect_count_list = []
            for classes in range(self.class_num):
                self.predect_count_list.append(np.zeros((img_h, img_w))) #记录每个类的个数
            #########################################################################
            predict_img = self.test_patches(test_dataset, img_name, model_img, img_h, img_w, patch_size, run_time)
            
            total_num, pos_num = self.save_with_acc(img_name, sample['label'], predict_img)
            self.total_pixel += total_num
            self.pos_pixel += pos_num
        acc = self.pos_pixel/self.total_pixel
        print('AC is %f'%acc)

    def test_patches(self, test_dataset, img_NO, models, img_h, img_w, patch_size, run_time):
        img_model = models
        img_model.eval()

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
                self.test_single_patch(test_dataset, start_h, end_h, start_w, end_w, img_model)
                
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
            self.test_single_patch(test_dataset, start_h, end_h, start_w, end_w, img_model)

        image_predect = np.array([self.predect_count_list[0], self.predect_count_list[1], self.predect_count_list[2], 
            self.predect_count_list[3], self.predect_count_list[4], self.predect_count_list[5]])
        
        image_predect = np.argmax(image_predect, 0).astype(np.uint8)
        return image_predect
        
        
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

    def test_single_patch(self, test_dataset, start_h, end_h, start_w, end_w, model_img):
        with torch.no_grad():
            sample = test_dataset.sample
            self.img = sample['img']
            self.label = sample['label']
            inputs = self.img[:, :, start_h:end_h, start_w:end_w]

            inputs = inputs.permute(0, 2, 3, 1).contiguous()
            inputs_0 = torch.mul(inputs, model_img.auto_weight0).permute(0, 3, 1, 2).contiguous()
            inputs_1 = torch.mul(inputs, model_img.auto_weight1).permute(0, 3, 1, 2).contiguous()
            inputs_2 = torch.mul(inputs, model_img.auto_weight2).permute(0, 3, 1, 2).contiguous()
            inputs_3 = torch.mul(inputs, model_img.auto_weight3).permute(0, 3, 1, 2).contiguous()
            inputs_4 = torch.mul(inputs, model_img.auto_weight4).permute(0, 3, 1, 2).contiguous()
            inputs_5 = torch.mul(inputs, model_img.auto_weight5).permute(0, 3, 1, 2).contiguous()

            output_0 = model_img.CARB0(model_img(inputs_0)[0])
            output_1 = model_img.CARB1(model_img(inputs_1)[0])
            output_2 = model_img.CARB2(model_img(inputs_2)[0])
            output_3 = model_img.CARB3(model_img(inputs_3)[0])
            output_4 = model_img.CARB4(model_img(inputs_4)[0])
            output_5 = model_img.CARB5(model_img(inputs_5)[0])
            output = torch.cat([output_0, output_1, output_2, output_3, output_4, output_5], 1)

            max_probability, max_index = torch.max(output, 1)
            max_probability = max_probability.cpu().detach().numpy()
            max_index = max_index.cpu().detach().numpy()
            max_probability = max_probability.reshape((max_probability.shape[1], max_probability.shape[2]))
            max_index = max_index.reshape((max_index.shape[1], max_index.shape[2]))

            
            big_pb_index = (self.image_probability[start_h:end_h, start_w:end_w] < max_probability)
            self.image_probability[start_h:end_h, start_w:end_w][big_pb_index] = max_probability[big_pb_index]

            for classes in range(self.class_num):
                class_index = (max_index == classes)
                self.predect_count_list[classes][start_h:end_h, start_w:end_w][class_index] += 1


    def save_with_acc(self, img_name, label, image_predect):
        label = label.cpu().detach().numpy()
        img_h = label.shape[0]
        img_w = label.shape[1]

        pos_index = (label==image_predect)
        total_num = img_h*img_w
        self.total_pixel += total_num
        pos_num = pos_index.sum()
        self.pos_pixel += pos_num
        true_rate = pos_num/total_num
        print('image %s , acc is %f'%(str(img_name), true_rate))

        image_predect = Image.fromarray(image_predect)
        predict_path = 'result\\pre' + str(img_name) + '_' + str(true_rate) + '.tif'
        image_predect.save(predict_path)
        return total_num, pos_num

if __name__ == "__main__":
    
    for i in range(10, 0, -1):
        img_path = '..\\XZY_DeepLearning_Framework\\result\\image_model' +str(i)+'.pkl'
        img_model = torch.load(img_path).cuda()
        test = test_model('GID', 6)
        test.test_image(img_model, 512, 0)
        break

