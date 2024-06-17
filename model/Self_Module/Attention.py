import torch
import torch.nn as nn
import torch.nn.functional as F
class Spatial_ATT(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Spatial_ATT, self).__init__()
                
        self.spa1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm2d(in_channels),)
        
        self.spa2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
                
        out = self.spa1(input)
        out = self.bn2(self.spa2(out))
        
        return F.relu(out + input)

class Channel_ATT(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_ATT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


        