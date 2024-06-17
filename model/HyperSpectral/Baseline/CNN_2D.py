import torch
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self, input_channels, out_channels):
        print('Using CNN2D model')
        super(CNN2D,self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_channels),
        )

        self.global_conv = nn.Conv2d(input_channels, input_channels, kernel_size=11)

        self.softmax = nn.Softmax(dim=-1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)

        x3_GAP = torch.nn.AdaptiveAvgPool2d(1)(x3)
        x3_GAP = x3_GAP.view(b, -1)
        out = self.fc(x3_GAP)

        return [out]

