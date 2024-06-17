from turtle import pen
import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN3D(nn.Module):
    """
    Based on paper:3-D Deep Learning Approach for Remote Sensing Image Classification  IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 56, NO. 8, AUGUST 2018
    """
    def __init__(self, classes):
        super(CNN3D, self).__init__()
        print('Using CNN3D model')
        self.conv1 = nn.Sequential(
            # Notice:cause input shape is [N,C,D,H,W]，kernel_size here should be (D,H,W)
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(),
        )
        self.FC = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(3456, classes)
        )
    def forward(self, x):
        b, c, h, w = x.size()
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt = x[:,:,pth-1:pth+2,ptw-1:ptw+2]

        x = torch.unsqueeze(pt, 1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3 = conv3.view(b, -1)
        out = self.FC(conv3)

        return [out]
