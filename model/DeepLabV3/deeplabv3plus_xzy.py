import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.ResNet.ResNet import resnet18, resnet34, resnet50, resnet101, resnet152
from model.ResNet.ResNet import BasicBlock
from model.DeepLabV3.ASPP_xzy import ASPP
from model.Self_Module.GetLayer import LayerGetter

backbone_dict = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50, 
                 'resnet101':resnet101, 'resnet152':resnet152}
featnumb_dict = {'resnet18':512, 'resnet34':512, 'resnet50':2048, 
                 'resnet101':2048, 'resnet152':2048}

class deeplabv3plus(nn.Module):
    def __init__(self, input_channels, out_channels, backbone='resnet101', aspp_rate=[6, 12, 18], pretrained=True):
        super(deeplabv3plus,self).__init__()
        # 基于ResNet的backbone
        self.backbone = backbone_dict[backbone](input_channels=input_channels, pretrained=pretrained)
        
        self.return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        self.highfeat_num = featnumb_dict[backbone]
        self.aspprate = aspp_rate

        self.aspp = ASPP(self.highfeat_num, self.aspprate)
        proj_num = int(self.highfeat_num/8)
        self.project = nn.Sequential( 
            nn.Conv2d(proj_num, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1)
        )
        
        self.layergetter = LayerGetter(self.backbone, self.return_layers)

        self.edgefuse_resblock = BasicBlock(out_channels, out_channels)
        self.CBfuse_conv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1), 
            nn.BatchNorm2d(out_channels)           
        )
        # self.CBfuse_conv_list = []
        # for i in range(out_channels):
        #     self.CBfuse_conv_list.append(nn.Sequential(
        #     nn.Conv2d(2, 1, kernel_size=1), 
        #     nn.BatchNorm2d(1)           
        # ).cuda())
            
        self.GBfuse_conv = nn.Sequential(
            nn.Conv2d(out_channels+1, out_channels, kernel_size=1), 
            nn.BatchNorm2d(out_channels)           
        )

    def forward(self, input_list, modelstyle='base'):
        if modelstyle == 'base':
            return self.forward_base(input_list)
        elif modelstyle == 'CB':
            return self.forward_CB(input_list)
        elif modelstyle == 'GB':
            return self.forward_GB(input_list)

    def forward_base(self, input_list):
        img = input_list[0]
        input_shape = img.shape[-2:]

        # 获取ResNet Backbone的第一层和最后一层输出
        backbone_out = self.layergetter(img)
        backbone_lowfeature = backbone_out['low_level']
        backbone_highfeature = backbone_out['out']
        
        aspp_out = self.aspp(backbone_highfeature)
        low_proj = self.project(backbone_lowfeature)

        output_feature = F.interpolate(aspp_out, size=low_proj.shape[2:], mode='bilinear', align_corners=False)
        out = self.classifier( torch.cat( [ low_proj, output_feature ], dim=1 ) )
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out

    def forward_CB(self, input_list):
        img = input_list[0]
        edge = input_list[1]
        input_shape = img.shape[-2:]

        # 获取ResNet Backbone的第一层和最后一层输出
        backbone_out = self.layergetter(img)
        backbone_lowfeature = backbone_out['low_level']
        backbone_highfeature = backbone_out['out']

        aspp_out = self.aspp(backbone_highfeature)
        low_proj = self.project(backbone_lowfeature)

        output_feature = F.interpolate(aspp_out, size=low_proj.shape[2:], mode='bilinear', align_corners=False)
        out = self.classifier( torch.cat( [ low_proj, output_feature ], dim=1 ) )
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)

        # out_convfuse = []
        # for i in range(out.shape[1]):
        #     out_tmp, edge_tmp = out[:,i,:,:].unsqueeze(1), edge[:,i,:,:].unsqueeze(1)
        #     fuse_tmp = torch.cat([out_tmp, edge_tmp], 1)
        #     out_convfuse.append(self.CBfuse_conv_list[i](fuse_tmp))
        # out_convfuse = torch.cat([x for x in out_convfuse], 1)
        out_convfuse = self.CBfuse_conv(torch.cat([out, edge], 1))
        out_resfuse = out_convfuse + self.edgefuse_resblock(out_convfuse)
        return out_resfuse
    
    def forward_GB(self, input_list):
        img = input_list[0]
        edge = input_list[1]
        input_shape = img.shape[-2:]

        # 获取ResNet Backbone的第一层和最后一层输出
        backbone_out = self.layergetter(img)
        backbone_lowfeature = backbone_out['low_level']
        backbone_highfeature = backbone_out['out']

        aspp_out = self.aspp(backbone_highfeature)
        low_proj = self.project(backbone_lowfeature)

        output_feature = F.interpolate(aspp_out, size=low_proj.shape[2:], mode='bilinear', align_corners=False)
        out = self.classifier( torch.cat( [ low_proj, output_feature ], dim=1 ) )
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)

        out_convfuse = self.GBfuse_conv(torch.cat([out, edge], 1))
        out_resfuse = out_convfuse + self.edgefuse_resblock(out_convfuse)
        return out_convfuse