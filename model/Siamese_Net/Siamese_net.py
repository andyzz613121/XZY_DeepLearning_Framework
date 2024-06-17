import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Bottleneck(nn.Module):
    expansion = 1
    def conv3x3(self, in_planes, out_planes, stride=1, groups=1, dilation=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # width = 64
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = self.conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = self.conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = self.conv1x1(width, planes * self.expansion)
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


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            # nn.ReflectionPad2d(6),
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )
        self.cnn2 = nn.Sequential(
            # nn.ReflectionPad2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.cnn3 = nn.Sequential(
            # nn.ReflectionPad2d(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )
        self.cnn4 = nn.Sequential(
            # nn.ReflectionPad2d(32),
            nn.Conv2d(32, 6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(6),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(6*256*256, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

        #bottleneck
        self.bottleneck1 = Bottleneck(32, 32)
        self.bottleneck2 = Bottleneck(64, 64)
        self.bottleneck3 = Bottleneck(32, 32)
        self.bottleneck4 = Bottleneck(6, 6)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward_once(self, x):
        x1 = self.cnn1(x)
        x1 = self.bottleneck1(x1)
        x1 = self.pool(x1)

        x2 = self.cnn2(x1)
        x2 = self.bottleneck2(x2)
        x2 = self.pool(x2)

        x3 = self.cnn3(x2)
        x3 = self.bottleneck3(x3)
        x3 = self.pool(x3)
        
        x4 = self.cnn4(x3)
        x4 = self.bottleneck4(x4)
        
        output = x4.view(x4.size()[0], -1)
        # output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
 
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
 
    def forward(self, output1, output2, label):
        print(output1.shape)
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
