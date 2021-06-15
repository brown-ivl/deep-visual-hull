import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, Sequential, Flatten
from torchinfo import summary

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
            MaxPool2d(2, 2), # batch_size, 512, 7, 10

            Flatten(),
            Linear(512*7*7, 256), # TODO: 512 * w/(2^5) * h/(2^5)
            ReLU(inplace=True),
            Linear(256, 256),
            ReLU(inplace=True)
        )

    def forward(self, images):
        out = self.architecture(images)
        return out

if __name__ == "__main__":
    encoder = DeepVisualHullEncoder()
    summary(encoder, (1,3,224,224))