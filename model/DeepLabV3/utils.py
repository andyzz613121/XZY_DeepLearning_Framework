import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

from model.DeepLabV3.backbone.resnet import conv1x1, conv3x3

class Deeplab_Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Deeplab_Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        width = 64
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        self.Bottleneck_Sequential = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
            self.relu,
            self.conv3,
            self.bn3,
        )
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class _SimpleSegmentationModel_lulc(nn.Module):
    def __init__(self, imgdsm, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.edgeconv = torch.nn.Conv2d(11, 10, kernel_size=1)#12
        # self.BasicBlock = BasicBlock(6,6)
        self.Bottleneck = Deeplab_Bottleneck(10,10)#6,6
        self.Bottleneck.expansion = 1
        self.imgdsmconv = nn.Conv2d(12, 6, 1)
        self.aspp_dsmfeature = None
        self.dsm_low_feature = None

    def give_img_dsm_aspp_feature(self, aspp_dsmfeature):
        self.classifier.dsm_aspp_feature = aspp_dsmfeature
    def give_img_dsm_low_feature(self, low_dsmfeature):
        self.classifier.dsm_low_feature = low_dsmfeature
    def give_img_hed_feature(self, hed_feature):
        self.classifier.hed_feature = hed_feature

    def get_dsm_low_feature(self):
        return self.classifier.dsm_low_feature
    def get_dsm_aspp_feature(self):
        return self.classifier.dsm_aspp_feature

    def forward(self, x, EDGE):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        self.dsm_low_feature = features
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        # x = torch.nn.functional.relu(x)
        # x_copy = x
        x = torch.cat([x, EDGE], 1)
        # x = self.SE(x)
        x = self.edgeconv(x)
        # x = nn.ReLU(inplace=True)(self.edgeconv(x))
        # print(x.shape, EDGE.shape)
        x_res = self.Bottleneck.Bottleneck_Sequential(x)
        x = x + x_res
        # return x, x_res, EDGE, x_copy
        return x, features
    def forward_EDGE(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        self.dsm_low_feature = features
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x, features

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, imgdsm, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.edgeconv = nn.Sequential(
            nn.Conv2d(4, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        # self.BasicBlock = BasicBlock(6,6)
        self.Bottleneck0 = Deeplab_Bottleneck(6,6)
        self.Bottleneck0.expansion = 1
        self.Bottleneck1 = Deeplab_Bottleneck(6,6)
        self.Bottleneck1.expansion = 1
        self.imgdsmconv = nn.Conv2d(6, 3, 1)

        self.aspp_dsmfeature = None
        self.dsm_low_feature = None

    def give_img_dsm_aspp_feature(self, aspp_dsmfeature):
        self.classifier.dsm_aspp_feature = aspp_dsmfeature
    def give_img_dsm_low_feature(self, low_dsmfeature):
        self.classifier.dsm_low_feature = low_dsmfeature
    
    def get_dsm_low_feature(self):
        return self.classifier.dsm_low_feature
    def get_dsm_aspp_feature(self):
        return self.classifier.dsm_aspp_feature
    # # def forward(self, x, EDGE):
    # def forward(self, x):
    #     input_shape = x.shape[-2:]
    #     features = self.backbone(x)
    #     self.dsm_low_feature = features
    #     x = self.classifier(features)
    #     x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
    #     # # x = torch.nn.functional.relu(x)
    #     # # x_copy = x
    #     # x = torch.cat([x, EDGE], 1)
    #     # x = self.edgeconv(x)
    #     # # x = nn.ReLU(inplace=True)(self.edgeconv(x))
    #     # # print(x.shape, EDGE.shape)
    #     # x_res = self.Bottleneck0.Bottleneck_Sequential(x)
    #     # x = x + x_res
    #     # # #x_res = self.Bottleneck1.Bottleneck_Sequential(x)
    #     # # #x = x + x_res
    #     # # return x, x_res, EDGE, x_copy
    #     return x
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        # x = torch.cat([x, EDGE], 1)
        # x = self.edgeconv(x)
        # print(x.shape)
        # x = self.edgeconv(x)
        return x
class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        # print('out',out)
        return out
