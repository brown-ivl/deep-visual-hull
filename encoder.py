import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, Sequential

# VGG-16 architecture, derived from SegNet encoder (no skip connections)
class DeepVisualHullEncoder(nn.Module):
    def __init__(self, in_channels = 3):
        self.architecture = Sequential(
            # SEGNET ARCHITECTURE
            Conv2d(in_channels, 64, 3, 1, 1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 64, 3, 1, 1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(2, 2),

            Conv2d(64, 128, 3, 1, 1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 128, 3, 1, 1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(2, 2),

            Conv2d(128, 256, 3, 1, 1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, 3, 1, 1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, 3, 1, 1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(2, 2),

            Conv2d(256, 512, 3, 1, 1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, 3, 1, 1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, 3, 1, 1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(2, 2),

            Conv2d(512, 512, 3, 1, 1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, 3, 1, 1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, 3, 1, 1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(2, 2),

            Linear(512, 256),
            ReLU(inplace=True),
            Linear(512, 256)
        )

