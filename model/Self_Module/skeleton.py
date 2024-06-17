import torch
from torch import nn
class skeleton(nn.Module):
    def __init__(self):
        super(skeleton, self).__init__()
        self.skeleton_deconv_5 = nn.Sequential(
            nn.Linear(36, , bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )