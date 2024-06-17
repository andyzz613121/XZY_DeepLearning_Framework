import torch
import torch.nn as nn
import torch.nn.functional as F
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()

        modules = []
        
        # 增加1*1卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))
        # 增加3*3空洞卷积
        
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        # 增加可变卷积
        # modules.append(DeformableConv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.convs = nn.ModuleList(modules)

        # 空洞卷积结果降维（1个1*1，3个3*3，1个池化，一个可变卷积）
        self.project = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),)
            # nn.Dropout(0.1))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)