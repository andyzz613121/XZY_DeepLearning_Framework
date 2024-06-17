import torch
import torch.nn as nn
import torch.nn.functional as F
class FDSSC(nn.Module):
    """
    Based on paper:3-D Deep Learning Approach for Remote Sensing Image Classification  IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 56, NO. 8, AUGUST 2018
    """
    def __init__(self, input_channels, classes):
        super(FDSSC, self).__init__()
        print('Using FDSSC model')
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(in_channels=input_channels, out_channels=24, kernel_size=(7, 1, 1)),
            nn.PReLU(inplace=True),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=12, kernel_size=(7, 1, 1)),
            nn.PReLU(inplace=True),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv3d(in_channels=36, out_channels=12, kernel_size=(7, 1, 1)),
            nn.PReLU(inplace=True),
        )
        self.conv1_4 = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=12, kernel_size=(7, 1, 1)),
            nn.PReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=60, out_channels=200, kernel_size=(1, 1, 1)),
            nn.PReLU(inplace=True),
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x1_1 = self.conv1_1(x)      # 24
        print(x1_1.shape)
        x1_2 = self.conv1_2(x1_1)   #12
        x1_3 = self.conv1_3(torch.cat([x1_1, x1_2], 1)) #36->12
        x1_4 = self.conv1_4(torch.cat([x1_1, x1_2, x1_3], 1)) #45->12
        x2 = torch.cat([x1_1, x1_2, x1_3, x1_4], 1)
        
        

        return [out]
