import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.main(x) + self.shortcut(x)
        return F.relu(out, inplace=True)

class ResNet(nn.Module):
    def __init__(self, channel_1, channel_2, num_classes):
        super().__init__()

        self.resnet = nn.Sequential(
                nn.Conv2d(3, channel_1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channel_1),
                nn.ReLU(inplace=True),

                ResidualBlock(channel_1, channel_1, stride=1),
                ResidualBlock(channel_1, channel_2, stride=2),
                ResidualBlock(channel_2, channel_2, stride=2),

                nn.Flatten(),
                nn.Linear(channel_2 * 8 * 8, num_classes),
            )
    
    def forward(self, x):
        return self.resnet(x)
