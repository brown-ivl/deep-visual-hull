import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, Sequential


# VGG-16 architecture, derived from SegNet encoder (no skip connections)
class DeepVisualHullEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.architecture = Sequential(
            # SegNet ARCHITECTURE
            Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(2, 2),

            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(2, 2),

            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(2, 2),

            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(2, 2),

            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(2, 2),

            Linear(512, 256),
            ReLU(inplace=True),
            Linear(512, 256)
        )