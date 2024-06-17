import torch
from torch import nn
from torch.nn import functional as F
from osgeo import gdal
from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass
class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, imgdsm, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.imgdsm = imgdsm
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)
        self.dsm_aspp_feature = None
        self.dsm_low_feature = None
        # self.hed_feature = None
        # self.SE = SELayer(126)
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

        self.imgfusedsm_aspp = nn.Sequential(
            nn.Conv2d(256*2, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.imgfusedsm_low = nn.Sequential(
            nn.Conv2d(48*2, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

    def forward_lulc(self, feature):
        if self.imgdsm == 'dsm':
            low_level_feature = self.project( feature['low_level'] )
            self.dsm_low_feature = low_level_feature
            
            output_feature = self.aspp(feature['out'])
            self.dsm_aspp_feature = output_feature
            output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
            return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
        elif self.imgdsm == 'img':
            low_level_feature = self.project( feature['low_level'] )
            h, w = self.dsm_low_feature.shape[2], self.dsm_low_feature.shape[3]
            # self.hed_feature = F.interpolate(self.hed_feature, size=(h, w), mode='bilinear')
            # low_level_feature = torch.cat([low_level_feature, self.dsm_low_feature, self.hed_feature], 1)
            low_level_feature = torch.cat([low_level_feature, self.dsm_low_feature], 1)
            
            # low_level_feature = self.SE(low_level_feature)
            low_level_feature = self.imgfusedsm_low(low_level_feature)##
            # print(feature['out'].shape)
            output_feature = self.aspp(feature['out'])
            # image_predect = low_level_feature.cpu().detach().numpy()
            # predict_path = 'C:\\Users\\ASUS\\Desktop\\论文数据\\HED_2313\\aspp\\img_low.tif'
            # driver = gdal.GetDriverByName("GTiff") 
            # dataset = driver.Create(predict_path, 64, 64, 6, gdal.GDT_Byte)
            # for i in range(6):
            #     print(image_predect[:,i].shape)
            #     dataset.GetRasterBand(i+1).WriteArray(image_predect[0,i])

            output_feature = self.imgfusedsm_aspp( torch.cat([output_feature, self.dsm_aspp_feature], 1) )##
            output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
            return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
        # elif self.imgdsm == 'img':
        #     low_level_feature = self.project( feature['low_level'] )
        #     output_feature = self.aspp(feature['out'])
        #     output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        #     return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
        # elif self.imgdsm == 'img':
        #     low_level_feature = self.project( feature['low_level'] )
        #     low_level_feature = self.imgfusedsm_low( torch.cat([low_level_feature, self.dsm_low_feature], 1) )##
        #     # print(feature['out'].shape)
        #     output_feature = self.aspp(feature['out'])
            
        #     output_feature = self.imgfusedsm_aspp( torch.cat([output_feature, self.dsm_aspp_feature], 1) )##
        #     output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        #     return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
    def forward(self, feature):
        if self.imgdsm == 'dsm':
            low_level_feature = self.project( feature['low_level'] )
            self.dsm_low_feature = low_level_feature
            output_feature = self.aspp(feature['out'])
            self.dsm_aspp_feature = output_feature
            output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
            return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
        elif self.imgdsm == 'img':
            print(feature['low_level'].shape)
            low_level_feature = self.project( feature['low_level'] )
            print(low_level_feature.shape)
            output_feature = self.aspp(feature['out'])
            
            output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
            return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class DeepLabHeadV3Plus_img(nn.Module):
    def __init__(self, dsm_feature, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus_img, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.dsm_feature = dsm_feature
        self.aspp = ASPP(in_channels, aspp_dilate)
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()
        
        self.imgdsm = nn.Sequential(
            nn.Conv2d(256*2, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    def forward(self, dsm_feature, feature):
        low_level_feature = self.project( feature['low_level'] )
        print(feature['out'].shape)
        output_feature = self.aspp(feature['out'])
        output_feature = self.imgdsm( torch.cat([output_feature, self.dsm_feature], 1) )
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        # print(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            # print(x.shape,conv(x).shape)
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module