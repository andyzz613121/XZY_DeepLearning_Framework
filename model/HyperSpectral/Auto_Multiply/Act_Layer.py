import torch.nn as nn
from model.Self_Module.ASPP import ASPP
class Conv1D_Act(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.conv_act = nn.Sequential(
            nn.Conv1d(input_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm1d(out_channels),
        )
    def forward(self, x):
        return self.conv_act(x)

class Conv2D_Act(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.conv_act = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(out_channels),
            
        )
    def forward(self, x):
        return self.conv_act(x)

class BandMLP(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.band_mlp = nn.Sequential(
            nn.Linear(input_channels, 1),
            # nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.band_mlp(x)