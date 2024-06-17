import torch
import torch.nn as nn
import torch.nn.functional as F
class HybridSN(nn.Module):
    """
    Based on paper:HybridSN: Exploring 3-D-2-D CNN Feature Hierarchy for Hyperspectral Image Classification. IEEE Geoscience and Remote Sensing Letters
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, input_channels, bandnum, classes):
        super(HybridSN, self).__init__()
        print('Using HybridSN model')
        # self.FE = nn.Sequential(
        #     nn.Conv2d(in_channels=input_channels, out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
        #     nn.BatchNorm2d(30),
        # )
        self.conv1 = nn.Sequential(
            # Notice:cause input shape is [N,C,D,H,W]，kernel_size here should be (D,H,W)
            nn.Conv3d(in_channels=input_channels, out_channels=8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            # nn.Conv2d(in_channels=576, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(in_channels=32*(bandnum-12), out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.FC1 = nn.Sequential(
            nn.Linear(7744, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.classifier = nn.Linear(128, classes)

    def forward(self, x):
        if len(x.size()) < 5:
            x = torch.unsqueeze(x, 1)
        
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3 = torch.reshape(conv3, (conv3.shape[0], -1, conv3.shape[3], conv3.shape[4]))
        conv4 = self.conv4(conv3)
        conv4 = torch.reshape(conv4, (conv4.shape[0], -1))
        fc1 = self.FC1(conv4)
        fc2 = self.FC2(fc1)
        out = self.classifier(fc2)
        return out
