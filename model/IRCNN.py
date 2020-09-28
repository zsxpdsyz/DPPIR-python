import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch

class IRCNN(nn.Module):
    def __init__(self, inchannel):
        super(IRCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x


class Mean_Squared_Error(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(Mean_Squared_Error, self).__init__(size_average, reduce, reduction)
    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def _xavier_init_(m:nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)