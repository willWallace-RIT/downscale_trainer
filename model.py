import torch
import torch.nn as nn

class ContourGuidedNet(nn.Module):
    def __init__(self, in_channels=4, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, lr, contour):
        return self.net(torch.cat([lr, contour], dim=1))
