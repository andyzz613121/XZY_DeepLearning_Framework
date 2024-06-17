import torch
from torch import nn
import numpy as np
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        #shortcut
        self.shortcut = nn.Sequential()
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    def forward(self, x):
        return nn.ReLU(inplace=False)(self.residual_function(x) + self.shortcut(x))

class CARB_Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CARB_Block, self).__init__()
        self.sideout_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels)
        )
        self.bottlenect = BasicBlock(output_channels, output_channels)
        
    def forward(self, x):
        x_1 = self.sideout_conv(x)
        return self.bottlenect(x_1)

