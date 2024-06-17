from torch import nn

class mlp(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=16):
        super(mlp, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel, 2*in_channel, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(2*in_channel, out_channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)