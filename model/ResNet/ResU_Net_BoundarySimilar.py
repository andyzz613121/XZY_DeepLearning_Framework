import re
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from model.ResNet.ResU_Net import ResU_Net

class ResU_Net_BS(ResU_Net):
    def __init__(self, arch, input_channels, num_classes, pre_train=True):
        super(ResU_Net_BS, self).__init__(arch, input_channels, num_classes, pre_train)

        self.spectral_list = torch.zeros([num_classes, input_channels]).cuda() #记录各类像素光谱值的总和
        self.class_list = torch.zeros([num_classes]).cuda()                    #记录各类像素个数的总和
        
        self.T1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.a = nn.Parameter(torch.ones(1, requires_grad=True))
        self.w1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.w2 = nn.Parameter(torch.ones(1, requires_grad=True))

    def extract_edge(self, lulc_map, size):
        '''
        input: LULC MAPS: b * h * w
        out:   EDGE MAPS: b * h * w
        '''
        batch_size, h, w = lulc_map.size()

        mean_filter = torch.ones([1, 1, size, size], requires_grad=False).cuda()
        lulc_map = lulc_map.view(batch_size, 1, h, w)

        means_map = F.conv2d(lulc_map.float(), mean_filter, padding=int(size/2))
        # means_map_out = means_map.cpu().detach().numpy()[0][0]
        # gdal_write_tif('C:\\Users\\25321\\Desktop\\means_map_out.tif', means_map_out, 256, 256, 1)

        diff = means_map - lulc_map * size * size
        edge_index = (diff != 0)
        diff[edge_index] = 1
        edge = diff
        return edge

    def compute_class_spectral(self, img, lulc_map):
        '''
        input: img: b * c * h * w
               lulc_map: b * h * w
        out:   EDGE MAPS: b * h * w
        '''
        class_num = self.spectral_list.shape[0]
        channel_num = self.spectral_list.shape[1]
        batch_size, channel_num, h, w = img.size()

        img = img.permute(1, 0, 2, 3).contiguous().view(channel_num, batch_size*h*w)
        lulc_map = lulc_map.view(batch_size*h*w)

        for c in range(class_num):
            class_index = (lulc_map == c)
            self.spectral_list[c] += img[:, class_index].sum(1)
            self.class_list[c] += class_index.sum()
        
    def similar_match(self, img, edge, predict_old):
        '''
        input: img: b * c * h * w
               edge: b * h * w
               predict_old: b * class * h * w

        out:   predict_new: b * class * h * w, 其中需要更改的像素，通道的值是与每个类的相似度。
        '''
        class_num = self.spectral_list.shape[0]
        channel_num = self.spectral_list.shape[1]
        batch_size, channel_num, h, w = img.size()

        img = img.permute(0, 2, 3, 1).contiguous().view(batch_size, h*w, channel_num)
        edge = edge.view(batch_size, h*w)

        mean_spectral_list = self.spectral_list.permute(1, 0)/self.class_list
        nor_mean_spectral_list = mean_spectral_list.permute(1, 0)/torch.max(torch.abs(mean_spectral_list.permute(1, 0)), 0)[0]  #class, channel
        # for c in range(class_num):
        #     spectral_class = self.spectral_list[c]/self.class_list[c] 
        #     print('spectral_class', spectral_class)
        # print('nor_mean_spectral_list', nor_mean_spectral_list, nor_mean_spectral_list.shape)
        # print('mean_spectral_list', mean_spectral_list)
        similar_list = []     #相似性图像class*b*h*w
        for c in range(class_num):
            mean_vector = nor_mean_spectral_list[c].view(1, channel_num, 1).repeat(batch_size, 1, 1)
            # print(mean_vector, img)
            similar_c = torch.bmm(img, mean_vector)                               #img：c * (b*h*w) ;  mean_spectral: c
            # print('similar_c', similar_c)
            #是否要归一化
            similar_mask = similar_c.view(batch_size, h*w) * edge                 #非边界的地方,相似性为0;边界的地方,相似性有一个值
            similar_list.append(similar_mask)
        
        similar_list = torch.cat([x.view(batch_size, 1, h, w) for x in similar_list], 1)  # class*b*h*w
        similar_max, index_max = torch.max(similar_list, 1)
        similar_min, index_min = torch.min(similar_list, 1)

        # similar_mask_out = similar_list[0].cpu().detach().numpy()
        # gdal_write_tif('C:\\Users\\25321\\Desktop\\similar_mask_out.tif', similar_mask_out, 256, 256, 6)
        # index_max_out = index_max[0].cpu().detach().numpy()
        # gdal_write_tif('C:\\Users\\25321\\Desktop\\index_max_out.tif', index_max_out, 256, 256, 1)
        # index_min_out = index_min[0].cpu().detach().numpy()
        # gdal_write_tif('C:\\Users\\25321\\Desktop\\index_min_out.tif', index_min_out, 256, 256, 1)

        T2 = self.T1 + self.a * self.a
        p = F.relu(similar_max-T2) * F.relu(self.T1-similar_min)
        # p_negindex = (p == 0)

        predict_old = predict_old.view(class_num, batch_size, h, w)
        similar_list = similar_list.view(class_num, batch_size, h, w)
        predict_old = self.w1*(predict_old*p + predict_old) + self.w2*similar_list
        # predict_old[:,p_negindex] = similar_list[:,p_negindex]
        predict_old = predict_old.view(batch_size, class_num, h, w)
        return predict_old

    def forward(self, x):
        batch_size, channel_size, height, width = x.size()
        input_img = x
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_unpool = x
        x = self.resnet.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2) + x_unpool

        out = self.classifier(d1)
        lulc_map = torch.argmax(out, 1)  #b, h, w

        self.compute_class_spectral(input_img, lulc_map)
        # edge = self.extract_edge(lulc_map, 9)
        # predict_new = self.similar_match(input_img, edge, out)
        # return predict_new
        
        # lulc_map_out = lulc_map.cpu().detach().numpy()[0]
        # gdal_write_tif('C:\\Users\\25321\\Desktop\\lcmap.tif', lulc_map_out, 256, 256, 1)
        # edge_out = edge.cpu().detach().numpy()[0,0,:,:]
        # gdal_write_tif('C:\\Users\\25321\\Desktop\\edge.tif', edge_out, 256, 256, 1)

        
        return out

from osgeo import gdal
def gdal_write_tif(filename, img, img_w, img_h, bands=1, GeoTransform=None, Spatial_Ref=None, datatype=gdal.GDT_Float64):#img:[c,h,w]
        driver = gdal.GetDriverByName("GTiff") 
        dataset = driver.Create(filename, img_w, img_h, bands, datatype)
        if GeoTransform is not None:
            dataset.SetGeoTransform(GeoTransform)
        if Spatial_Ref is not None:
            dataset.SetSpatialRef(Spatial_Ref)
        
        if len(img.shape) == 2:
            dataset.GetRasterBand(1).WriteArray(img)
        elif len(img.shape) == 3:
            for b in range(bands):
                dataset.GetRasterBand(1+b).WriteArray(img[b])
        else:
            print('Error: gdal_write_tif//unknow img shape')

        del dataset

def add_ResUNet(model, premodel):
    model_dict = model.state_dict()
    premodel_dict = premodel.state_dict()
    premodel_list = []
    for key, value in premodel_dict.items():
        temp_dict = {'key':key,'value':value}
        premodel_list.append(temp_dict)

    for key in model_dict:
        for layer in premodel_list:
            pre_k = layer['key']
            pre_v = layer['value']
            if key == pre_k:
                assert model_dict[key].shape == pre_v.shape
                model_dict[key] = pre_v
    model.load_state_dict(model_dict)

    model.spectral_list = premodel.spectral_list
    model.class_list = premodel.class_list
    return model
