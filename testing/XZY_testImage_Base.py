import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

import os
import torch
import numpy as np
from PIL import Image

# from dataset.XZY_dataset_20221227 import XZY_test_dataset
from dataset.XZY_dataset_new import XZY_test_dataset
from data_processing.Raster import *
from utils.wrapper import *
class XZY_testIMG_classification_Base():
    def __init__(self, model_path_list, class_num, TestImgfile_list, Pre_Folder = 'result\\', pre_imgtag = '', norm_list=[True], gpu=True, patchsize=256, randomtime=0):
        '''
            model_path_list: 所有需要输入的模型路径 e.g. [seg_model_path, hed_model_path]
        '''
        print("---XZY_testIMG_classification_Base---")
        test_dataset = XZY_test_dataset(gpu=gpu, norm_list=norm_list)
        test_samples = test_dataset.get_test_samples(TestImgfile_list, norm_list)
        self.class_num = class_num
        self.img_h = test_samples['rawimg_0'].shape[2]
        self.img_w = test_samples['rawimg_0'].shape[3]
        self.predect_count_list = []
        for classes in range(self.class_num):
            self.predect_count_list.append(np.zeros((self.img_h, self.img_w)).astype(np.uint8)) #记录每个类的个数
        
        predict_path = Pre_Folder + pre_imgtag + '_pre.png'
        model_list = []
        for model_path in model_path_list:
            model = torch.load(model_path).eval()
            model_list.append(model)

        self.test_patches(test_samples, model_list, patchsize, randomtime)
        self.save_prediction(predict_path)

    def test_patches(self, test_samples, model_list, patch_size, run_time):

        img_w = self.img_w
        img_h = self.img_h

        cur_h = 0
        while cur_h < img_h: 
            start_h = cur_h 
            end_h = cur_h + patch_size
            if end_h >= img_h:
                end_h = img_h
                start_h = end_h - patch_size

            cur_w = 0
            while cur_w < img_w:
                start_w = cur_w
                end_w = cur_w + patch_size
                if end_w >= img_w:
                    end_w = img_w
                    start_w = end_w - patch_size

                self.test_single_patch(test_samples, start_h, end_h, start_w, end_w, model_list)

                cur_w+=int(patch_size/2)
            cur_h+=int(patch_size/2)

        for i in range(run_time):
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
            
            self.test_single_patch(test_samples, start_h, end_h, start_w, end_w, model_list)
        return True
    
    def save_prediction(self, predict_path):
        image_predect = np.array([self.predect_count_list[i] for i in range(self.class_num)])    
        image_predect = np.argmax(image_predect, 0).astype(np.uint8)
        image_predect = Image.fromarray(image_predect)
        image_predect.save(predict_path)

    def test_single_patch(self, test_samples, start_h, end_h, start_w, end_w, model_list):
        with torch.no_grad():

            # img = test_samples['img_0']
            # img_patch = img[:, :, start_h:end_h, start_w:end_w]
            
            rawimg1 = test_samples['rawimg_0'][:, :, start_h:end_h, start_w:end_w]
            rawimg2 = test_samples['rawimg_1'][:, :, start_h:end_h, start_w:end_w]
            rawimg = torch.cat([rawimg1, rawimg2], 1)
            # rawimg = test_samples['rawimg_0'][:, :, start_h:end_h, start_w:end_w]
            # rawimg1, rawimg2 = rawimg[:,4,:,:].unsqueeze(1), rawimg[:,4+6,:,:].unsqueeze(1)
            # rawimg = torch.cat([rawimg1, rawimg2], 1)
            # rawimg = rawimg.transpose(0, 1).contiguous().view(1, -1, 1)
            try:
                # img_outputs = model_list[0](rawimg)

                # output = model_list[0](img).view(1, 128, 128, 13)
                # img_outputs = output.transpose(2, 3).transpose(1, 2)
                # # output = output.view(128, 128, 13)
                # # img_outputs = output.transpose(0, 1).view(1, -1, 128, 128)

                rawimg = rawimg.contiguous().view(rawimg.shape[0], rawimg.shape[1], -1)      # B, C, H*W
                rawimg = rawimg.transpose(1, 2).contiguous().view(-1, 12)  # B*H*W, C, 1
                enc_input = rawimg.long()
                dec_input = torch.zeros_like(enc_input)[:,0].long().unsqueeze(1)+13
                output = model_list[0](enc_input, dec_input)
                img_outputs = output.transpose(0, 1).view(1, -1, 128, 128)
            except:
                # img_outputs = model_list[0]([rawimg1], 'base')[0].unsqueeze(0)
                start = 4
                img_patch = torch.cat([img_patch[:,start,:,:].unsqueeze(1), img_patch[:,start,:,:].unsqueeze(1), img_patch[:,start+6,:,:].unsqueeze(1)], 1)
                img_outputs = model_list[0](img_patch)
            self.agg_vote(img_outputs, start_h, end_h, start_w, end_w)

    def agg_vote(self, outputs, start_h, end_h, start_w, end_w):
        _, max_index = torch.max(outputs, 1)
        max_index = max_index.cpu().detach().numpy()
        max_index = max_index.reshape((max_index.shape[1], max_index.shape[2]))
        
        for classes in range(self.class_num):
            class_index = (max_index == classes)
            self.predect_count_list[classes][start_h:end_h, start_w:end_w][class_index] += 1

class XZY_testIMG_regression_Base(XZY_testIMG_classification_Base):
    def __init__(self, model_path_list, class_num, TestImgfile_list, Pre_Folder='result\\', pre_imgtag='', norm=True, gpu=True, patchsize=256, randomtime=0):
        print("---XZY_testIMG_regression_Base---")
        test_dataset = XZY_test_dataset(gpu=True, norm=True)
        test_samples = test_dataset.get_test_samples(TestImgfile_list)
        self.class_num = class_num

        self.img_h = test_samples['raw_image'].shape[2]
        self.img_w = test_samples['raw_image'].shape[3]
        self.predict_img = np.zeros((class_num, self.img_h, self.img_w)) 
        
        predict_path = Pre_Folder + pre_imgtag + '_pre.png'
        model_list = []
        for model_path in model_path_list:
            model = torch.load(model_path).eval()
            model_list.append(model)

        self.test_patches(test_samples, model_list, patchsize, randomtime)
        self.save_prediction(predict_path)

    def test_single_patch(self, test_samples, start_h, end_h, start_w, end_w, model_list):
        with torch.no_grad():

            img = test_samples['img_0']
            img_patch = img[:, :, start_h:end_h, start_w:end_w]
            img_patch = torch.cat([img_patch[:,1:4,:,:], img_patch[:,7,:,:].unsqueeze(1)], 1)

            edge_outputs = model_list[0](img_patch)[1][5]
            # edge_sum = edge_outputs[0].sum(0)
            # edge_neg = (edge_sum < 0.5)

            # edge_outputs[:, :, edge_neg] = 0
            self.predict_img[:, start_h:end_h, start_w:end_w] = edge_outputs.cpu().detach().numpy()

    def save_prediction(self, predict_path):
        image_predect = self.predict_img
        gdal_write_tif(predict_path, image_predect, image_predect.shape[1], image_predect.shape[2], image_predect.shape[0], datatype=2)

class XZY_testIMG_HS_Base(XZY_testIMG_classification_Base):
    # 增加了对高光谱图像的预测效率
    def __init__(self, model_list, class_num, TestImgfile_list, Pre_Folder='result\\', pre_imgtag='', norm_list=[True], gpu=True, patchsize=256, stride=128, randomtime=1000):
        print("---XZY_testIMG_HS_Base---")
        test_dataset = XZY_test_dataset(gpu=True, norm_list=norm_list)
        test_samples = test_dataset.get_test_samples(TestImgfile_list, norm_list)
        self.class_num = class_num
        self.img_h = test_samples['rawimg_0'].shape[2]
        self.img_w = test_samples['rawimg_0'].shape[3]

        self.predict_img = np.zeros([self.img_h, self.img_w])
        predict_path = Pre_Folder + pre_imgtag + '_pre.png'

        for i in range(len(model_list)):
            model_list[i] = model_list[i].eval()
        
        self.predect_count_list = []
        for classes in range(self.class_num):
            self.predect_count_list.append(np.zeros((self.img_h, self.img_w)).astype(np.uint8)) #记录每个类的个数

        self.test_patches(test_samples, model_list, patchsize, stride, randomtime)
        self.save_prediction(predict_path)

    @timer
    def test_patches(self, test_samples, model_list, patch_size, stride, run_time):

        img_w = self.img_w
        img_h = self.img_h

        cur_h = 0
        while cur_h < img_h: 
            start_h = cur_h 
            end_h = cur_h + patch_size
            if end_h >= img_h:
                end_h = img_h
                start_h = end_h - patch_size

            cur_w = 0
            while cur_w < img_w:
                start_w = cur_w
                end_w = cur_w + patch_size
                if end_w >= img_w:
                    end_w = img_w
                    start_w = end_w - patch_size
                
                self.test_single_patch(test_samples, start_h, end_h, start_w, end_w, model_list)

                cur_w+=stride
            cur_h+=stride

        for i in range(run_time):
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
            
            self.test_single_patch(test_samples, start_h, end_h, start_w, end_w, model_list)
        return True

    def test_single_patch(self, test_samples, start_h, end_h, start_w, end_w, model_list):
        with torch.no_grad():

            raw_imgs = test_samples['rawimg_0']
            img = test_samples['img_0']
            
            img_patch = raw_imgs[:, :, start_h:end_h, start_w:end_w]
            img_outputs = model_list[0](img_patch)[0]
            self.agg_vote(img_outputs, start_h, end_h, start_w, end_w)

    def save_prediction(self, predict_path):
        image_predect = np.array([self.predect_count_list[i] for i in range(self.class_num)])    
        image_predect = np.argmax(image_predect, 0).astype(np.uint8)
        image_predect = Image.fromarray(image_predect)
        image_predect.save(predict_path)

class XZY_testIMG_SS_Base(XZY_testIMG_HS_Base):
    # 预测空-谱双模型
    def __init__(self, model_list, class_num, TestImgfile_list, Pre_Folder='result\\', pre_imgtag='', norm_list=[True], gpu=True, patchsize=256, stride=128, randomtime=1000):
        print("---XZY_testIMG_SS_Base---")
        test_dataset = XZY_test_dataset(gpu=True, norm_list=norm_list)
        test_samples = test_dataset.get_test_samples(TestImgfile_list, norm_list)
        self.class_num = class_num
        self.img_h = test_samples['rawimg_0'].shape[2]
        self.img_w = test_samples['rawimg_0'].shape[3]

        self.predict_img_s1 = np.zeros([self.img_h, self.img_w])
        self.predict_img_s2 = np.zeros([self.img_h, self.img_w])
        self.predict_img_ss = np.zeros([self.img_h, self.img_w])
        predict_path_s1 = Pre_Folder + pre_imgtag + '_pre_s1.png'
        predict_path_s2 = Pre_Folder + pre_imgtag + '_pre_s2.png'
        predict_path_ss = Pre_Folder + pre_imgtag + '_pre_ss.png'
        for i in range(len(model_list)):
            model_list[i] = model_list[i].eval()
        
        self.predect_count_list_s1 = []
        self.predect_count_list_s2 = []
        self.predect_count_list_ss = []
        for classes in range(self.class_num):
            self.predect_count_list_s1.append(np.zeros((self.img_h, self.img_w)).astype(np.uint8)) #记录每个类的个数
            self.predect_count_list_s2.append(np.zeros((self.img_h, self.img_w)).astype(np.uint8)) #记录每个类的个数
            self.predect_count_list_ss.append(np.zeros((self.img_h, self.img_w)).astype(np.uint8)) #记录每个类的个数

        self.test_patches(test_samples, model_list, patchsize, stride, randomtime)

        self.save_prediction(predict_path_s1, self.predect_count_list_s1, self.class_num)
        self.save_prediction(predict_path_s2, self.predect_count_list_s2, self.class_num)
        self.save_prediction(predict_path_ss, self.predect_count_list_ss, self.class_num)

    def test_single_patch(self, test_samples, start_h, end_h, start_w, end_w, model_list):
        with torch.no_grad():

            img = test_samples['img_0']
            
            img_patch = img[:, :, start_h:end_h, start_w:end_w]
            hs_outputs = model_list[0](img_patch, 'test')

            img_patch4c = torch.cat([img_patch[:,1:4,:,:], img_patch[:,7,:,:].unsqueeze(1)], 1)
            cb_outputs = model_list[1](img_patch4c)[1][5]
            img_cb = torch.cat([img_patch, cb_outputs], 1)
            cb_inputs = [img_cb, cb_outputs]
            cb_outputs = model_list[2](cb_inputs, 'CB')
            
            # hs_outputs, cb_outputs = hs_outputs[0], cb_outputs[0]
            # hs_outputs = torch.softmax(hs_outputs, 0)
            # cb_outputs = torch.softmax(cb_outputs, 0)
            # ss_outputs = hs_outputs+cb_outputs
            # hs_outputs, cb_outputs, ss_outputs = hs_outputs.unsqueeze(0), cb_outputs.unsqueeze(0), ss_outputs.unsqueeze(0)

            hs_outputs, cb_outputs = hs_outputs[0], cb_outputs[0]
            hs_max, _ = torch.max(torch.softmax(hs_outputs, 0), 0)
            cb_max, _ = torch.max(torch.softmax(cb_outputs, 0), 0)
            max_dif = hs_max - cb_max
            hs_pos = max_dif > 0 # 256, 256
            ss_outputs = cb_outputs.clone()
            ss_outputs[:, hs_pos] = hs_outputs[:, hs_pos]
            hs_outputs, cb_outputs, ss_outputs = hs_outputs.unsqueeze(0), cb_outputs.unsqueeze(0), ss_outputs.unsqueeze(0)

            # hs_outputs, cb_outputs = hs_outputs[0], cb_outputs[0]
            # hs_outputs, cb_outputs = torch.softmax(hs_outputs, 0), torch.softmax(cb_outputs, 0)
            # hs_argmax, cb_argmax = torch.argmax(hs_outputs, 0), torch.argmax(cb_outputs, 0)
            # neg_idx = (hs_argmax != cb_argmax)
            
            # hs_max, _ = torch.max(hs_outputs, 0)
            # cb_max, _ = torch.max(cb_outputs, 0)
            # hsnew = hs_outputs.clone()
            # max_dif = hs_max - cb_max
            # cb_pos = max_dif < 0 # 256, 256
            # hsnew[:, cb_pos] = cb_outputs[:, cb_pos]
            # ss_outputs = cb_outputs.clone()
            # ss_outputs[:, neg_idx] = hsnew[:, neg_idx]
            # hs_outputs, cb_outputs, ss_outputs = hs_outputs.unsqueeze(0), cb_outputs.unsqueeze(0), ss_outputs.unsqueeze(0)

            self.agg_vote(self.predect_count_list_s1, hs_outputs, start_h, end_h, start_w, end_w)
            self.agg_vote(self.predect_count_list_s2, cb_outputs, start_h, end_h, start_w, end_w)
            self.agg_vote(self.predect_count_list_ss, ss_outputs, start_h, end_h, start_w, end_w)

    def agg_vote(self, predect_count_list, outputs, start_h, end_h, start_w, end_w):
        _, max_index = torch.max(outputs, 1)
        max_index = max_index.cpu().detach().numpy()
        max_index = max_index.reshape((max_index.shape[1], max_index.shape[2]))
        
        for classes in range(self.class_num):
            class_index = (max_index == classes)
            predect_count_list[classes][start_h:end_h, start_w:end_w][class_index] += 1

    def save_prediction(self, predict_path, predect_count_list, class_num):
        image_predect = np.array([predect_count_list[i] for i in range(class_num)])    
        image_predect = np.argmax(image_predect, 0).astype(np.uint8)
        image_predect = Image.fromarray(image_predect)
        image_predect.save(predict_path)

if __name__ == "__main__":
    base_folder = 'result\\Time_500\\SS\\'
    for i in range(10):
        net = 'Transformer_decompms_(1_2_3)' + str(i)
        img_path1 = 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Imgs\\Spectral_pred\\img_test.tif'
        img_path2 = 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Imgs\\Spectral_pred\\img_test.tif'
        model_path = base_folder + net + '\\' + net + '_model10.pkl'
        test = XZY_testIMG_classification_Base([model_path], 10, [img_path1, img_path2], norm_list=[True, True], Pre_Folder = base_folder+net+'\\', pre_imgtag = 'pred_'+net, patchsize=128, randomtime=3000)

    
    # for date in ['9_12', '8_04', '7_14', '6_29', '5_20', '10_17']:
    #     if date == '6_29':
    #         output_channels = 8
    #         datet = '6.29'
    #     elif date == '7_14':
    #         output_channels = 10
    #         datet = '7.14'
    #     elif date == '7_24':
    #         output_channels = 11
    #         datet = '7.24'
    #     elif date == '8_04':
    #         output_channels = 11
    #         datet = '8.04'
    #     elif date == '9_12':
    #         output_channels = 10
    #         datet = '9.12'
    #     elif date == '4_10':
    #         output_channels = 7
    #         datet = '4.10'
    #     elif date == '5_20':
    #         output_channels = 7
    #         datet = '5.20'
    #     elif date == '10_17':
    #         output_channels = 6
    #         datet = '10.17'

    #     img_path = 'E:\\dataset\\毕设数据\\new\\2. MS\\Segment\\Test\\'+datet+'\\img_test.tif'
    #     hsmodel_path = 'result\\HS500\\Sentinel'+date+'\\LGSF\\image_model_xzy500.pkl'
    #     cbmodel_path = 'result\\Spatial_paper\\Edge\\'+datet+'\\CB\\Hed_image_model200.pkl'
    #     sgmodel_path = 'result\\Spatial_paper\\Segment\\'+datet+'\\allchannel_train\\CB\\deeplabv3_plus_image_model300.pkl'
    #     Pre_Folder = 'result\\Spatial_Spectral\\'+date+'\\'
    #     if not os.path.exists(Pre_Folder):
    #         os.makedirs(Pre_Folder)

    #     hsmodel = torch.load(hsmodel_path).cuda()
    #     cbmodel = torch.load(cbmodel_path).cuda()
    #     sgmodel = torch.load(sgmodel_path).cuda()
    #     test = XZY_testIMG_SS_Base([hsmodel, cbmodel, sgmodel], output_channels, [img_path], Pre_Folder=Pre_Folder, pre_imgtag = 'pred_seg100', patchsize=256, stride=100, randomtime=0)
        # img_path = 'E:\\dataset\\毕设数据\\new\\2. MS\\SS_month\\img_test.tif'
        # sgmodel_path = 'result\\SS\\'+datet+'\\SegNet_model60.pkl'
        # Pre_Folder = 'result\\SS\\'+datet+'\\'
        # if not os.path.exists(Pre_Folder):
        #     os.makedirs(Pre_Folder)
        # test = XZY_testIMG_classification_Base([sgmodel_path], output_channels, [img_path], Pre_Folder=Pre_Folder, pre_imgtag = 'pred_seg100', patchsize=256, randomtime=0)

    # for date in ['9_12', '8_04', '7_14', '6_29', '5_20']:
    #     img_path = 'E:\\dataset\\毕设数据\\new\\2. MS\\Imgs\\'+date + '.tif'
    #     seg_path = 'result\\Segment\\'+date+'\\allchannel_train\\CB\\deeplabv3_plus_image_model200.pkl'
    #     hed_path = 'result\\Edge\\'+date+'\\CB\\Hed_image_model200.pkl'
    #     model_path_list = [seg_path, hed_path]
    #     if date == '6_29':
    #         output_channels = 8
    #     elif date == '7_14':
    #         output_channels = 10
    #     elif date == '7_24':
    #         output_channels = 11
    #     elif date == '8_04':
    #         output_channels = 11
    #     elif date == '9_12':
    #         output_channels = 10
    #     elif date == '4_10':
    #         output_channels = 7
    #     elif date == '5_20':
    #         output_channels = 7
    #     elif date == '10_17':
    #         output_channels = 6
    #     test = XZY_testIMG_classification_Base(model_path_list, output_channels, [img_path], Pre_Folder = 'E:\\dataset\\毕设数据\\new\\2. MS\\Time_Imgs\\', pre_imgtag = date, patchsize=512, randomtime=3000)
    
    