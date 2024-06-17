import torch
import torch.nn as nn
import torch.nn.functional as F
class FDSSC(nn.Module):
    """
    Based on paper:A Fast Dense Spectral–Spatial Convolution Network Framework for Hyperspectral Images Classification
    """
    def __init__(self, input_channels, classes):
        super(FDSSC, self).__init__()
        print('Using FDSSC model')
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(7, 1, 1), stride=(2, 1, 1), padding=(int((7-1)/2), 0, 0)),
            nn.BatchNorm3d(24),
            nn.PReLU(),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=12, kernel_size=(7, 1, 1), padding=(int((7-1)/2), 0, 0)),
            nn.BatchNorm3d(12),
            nn.PReLU(),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv3d(in_channels=36, out_channels=12, kernel_size=(7, 1, 1), padding=(int((7-1)/2), 0, 0)),
            nn.BatchNorm3d(12),
            nn.PReLU(),
        )
        self.conv1_4 = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=12, kernel_size=(7, 1, 1), padding=(int((7-1)/2), 0, 0)),
            nn.BatchNorm3d(12),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=60, out_channels=200, kernel_size=(int(input_channels/2), 1, 1)),
            nn.BatchNorm3d(200),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(200, 3, 3)),
            nn.BatchNorm3d(24),
            nn.PReLU(),
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=12, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(12),
            nn.PReLU(),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv3d(in_channels=36, out_channels=12, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(12),
            nn.PReLU(),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=12, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(12),
            nn.PReLU(),
        )
        self.GAP = torch.nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(60, classes),
            nn.PReLU(),
        )
    def forward(self, x):
        b, c, h, w = x.size()
        x = torch.unsqueeze(x, 1)

        x1_1 = self.conv1_1(x)      # 24
        x1_2 = self.conv1_2(x1_1)   # 12
        x1_3 = self.conv1_3(torch.cat([x1_1, x1_2], 1)) #36->12
        x1_4 = self.conv1_4(torch.cat([x1_1, x1_2, x1_3], 1)) #45->12
        x2 = torch.cat([x1_1, x1_2, x1_3, x1_4], 1)

        x3 = self.conv2(x2)
        x3 = torch.reshape(x3, [b, 1, -1, h, w])
        x3 = self.conv3(x3)      # 24
        x3_1 = self.conv3_1(x3)  # 12
        x3_2 = self.conv3_2(torch.cat([x3, x3_1], 1))
        x3_3 = self.conv3_3(torch.cat([x3, x3_1, x3_2], 1))
        x4 = torch.cat([x3, x3_1, x3_2, x3_3], 1)
        
        x5 = self.GAP(x4).view(b, -1)
        out = self.fc(x5)
        return [out]
