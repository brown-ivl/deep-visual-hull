# Borrowed from https://github.com/meetshah1995/pytorch-semseg
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, Sequential
import torchvision.models as models
import numpy as np
import os, sys

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath))
from encoder import DeepVisualHullEncoder
from decoder import DeepVisualHullDecoder

class dvhNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DeepVisualHullEncoder() # Seg Net Encoder
        self.decoder = DeepVisualHullDecoder() # Occupancy Network Decoder

    def forward(self, images, points):
        '''
        args:
        images: observations of the object （batch_size, c, w, h)
        points: a batch of 3d points to be passed into the decoder (batch_size, 3, T) or (batch_size, T, 3)
        '''
        B, C, W, H = images.size()
        # TODO: will we use the pre-trained weights?
        # for b in range(B):
        #     images[b] = normalizeInput(images[b], format='imagenet')  # assuming input is the range 0-1

        raw_c = self.encoder(images) # (batch_sizez, ?)
        if points.size(1) != 3:
            points = points.transpose(1, 2) # (batch_size, 3, T)
        out = self.decoder(points, raw_c)
        return out


def normalizeInput(Image, format='imagenet'):
    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    ImageN = Image # Assuming that input is in 0-1 range already
    if 'imagenet' in format:
        # Apply ImageNet batch normalization for input
        # https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560
        ImageN[0] = (ImageN[0] - 0.485 ) / 0.229
        ImageN[1] = (ImageN[1] - 0.456 ) / 0.224
        ImageN[2] = (ImageN[2] - 0.406 ) / 0.225
    else:
        print('[ WARN ]: Input normalization implemented only for ImageNet.')

    return ImageN


# summary(dvhNet(), [(1, 3, 224, 224), (1, 3, 4)])
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# dvhNet                                   --                        --
# ├─DeepVisualHullEncoder: 1-1             [1, 256]                  --
# │    └─Sequential: 2-1                   [1, 256]                  --
# │    │    └─Conv2d: 3-1                  [1, 64, 224, 224]         1,792
# │    │    └─BatchNorm2d: 3-2             [1, 64, 224, 224]         128
# │    │    └─ReLU: 3-3                    [1, 64, 224, 224]         --
# │    │    └─Conv2d: 3-4                  [1, 64, 224, 224]         36,928
# │    │    └─BatchNorm2d: 3-5             [1, 64, 224, 224]         128
# │    │    └─ReLU: 3-6                    [1, 64, 224, 224]         --
# │    │    └─MaxPool2d: 3-7               [1, 64, 112, 112]         --
# │    │    └─Conv2d: 3-8                  [1, 128, 112, 112]        73,856
# │    │    └─BatchNorm2d: 3-9             [1, 128, 112, 112]        256
# │    │    └─ReLU: 3-10                   [1, 128, 112, 112]        --
# │    │    └─Conv2d: 3-11                 [1, 128, 112, 112]        147,584
# │    │    └─BatchNorm2d: 3-12            [1, 128, 112, 112]        256
# │    │    └─ReLU: 3-13                   [1, 128, 112, 112]        --
# │    │    └─MaxPool2d: 3-14              [1, 128, 56, 56]          --
# │    │    └─Conv2d: 3-15                 [1, 256, 56, 56]          295,168
# │    │    └─BatchNorm2d: 3-16            [1, 256, 56, 56]          512
# │    │    └─ReLU: 3-17                   [1, 256, 56, 56]          --
# │    │    └─Conv2d: 3-18                 [1, 256, 56, 56]          590,080
# │    │    └─BatchNorm2d: 3-19            [1, 256, 56, 56]          512
# │    │    └─ReLU: 3-20                   [1, 256, 56, 56]          --
# │    │    └─Conv2d: 3-21                 [1, 256, 56, 56]          590,080
# │    │    └─BatchNorm2d: 3-22            [1, 256, 56, 56]          512
# │    │    └─ReLU: 3-23                   [1, 256, 56, 56]          --
# │    │    └─MaxPool2d: 3-24              [1, 256, 28, 28]          --
# │    │    └─Conv2d: 3-25                 [1, 512, 28, 28]          1,180,160
# │    │    └─BatchNorm2d: 3-26            [1, 512, 28, 28]          1,024
# │    │    └─ReLU: 3-27                   [1, 512, 28, 28]          --
# │    │    └─Conv2d: 3-28                 [1, 512, 28, 28]          2,359,808
# │    │    └─BatchNorm2d: 3-29            [1, 512, 28, 28]          1,024
# │    │    └─ReLU: 3-30                   [1, 512, 28, 28]          --
# │    │    └─Conv2d: 3-31                 [1, 512, 28, 28]          2,359,808
# │    │    └─BatchNorm2d: 3-32            [1, 512, 28, 28]          1,024
# │    │    └─ReLU: 3-33                   [1, 512, 28, 28]          --
# │    │    └─MaxPool2d: 3-34              [1, 512, 14, 14]          --
# │    │    └─Conv2d: 3-35                 [1, 512, 14, 14]          2,359,808
# │    │    └─BatchNorm2d: 3-36            [1, 512, 14, 14]          1,024
# │    │    └─ReLU: 3-37                   [1, 512, 14, 14]          --
# │    │    └─Conv2d: 3-38                 [1, 512, 14, 14]          2,359,808
# │    │    └─BatchNorm2d: 3-39            [1, 512, 14, 14]          1,024
# │    │    └─ReLU: 3-40                   [1, 512, 14, 14]          --
# │    │    └─Conv2d: 3-41                 [1, 512, 14, 14]          2,359,808
# │    │    └─BatchNorm2d: 3-42            [1, 512, 14, 14]          1,024
# │    │    └─ReLU: 3-43                   [1, 512, 14, 14]          --
# │    │    └─MaxPool2d: 3-44              [1, 512, 7, 7]            --
# │    │    └─Flatten: 3-45                [1, 25088]                --
# │    │    └─Linear: 3-46                 [1, 256]                  6,422,784
# │    │    └─ReLU: 3-47                   [1, 256]                  --
# │    │    └─Linear: 3-48                 [1, 256]                  65,792
# │    │    └─ReLU: 3-49                   [1, 256]                  --
# ├─DeepVisualHullDecoder: 1-2             [1, 1, 4]                 --
# │    └─Conv1d: 2-2                       [1, 256, 4]               1,024
# │    └─CondResnetBlock: 2-3              [1, 256, 4]               --
# │    │    └─CondBatchNorm: 3-50          [1, 256, 4]               131,584
# │    │    └─ReLU: 3-51                   [1, 256, 4]               --
# │    │    └─Conv1d: 3-52                 [1, 256, 4]               65,792
# │    │    └─CondBatchNorm: 3-53          [1, 256, 4]               131,584
# │    │    └─ReLU: 3-54                   [1, 256, 4]               --
# │    │    └─Conv1d: 3-55                 [1, 256, 4]               65,792
# │    └─CondResnetBlock: 2-4              [1, 256, 4]               --
# │    │    └─CondBatchNorm: 3-56          [1, 256, 4]               131,584
# │    │    └─ReLU: 3-57                   [1, 256, 4]               --
# │    │    └─Conv1d: 3-58                 [1, 256, 4]               65,792
# │    │    └─CondBatchNorm: 3-59          [1, 256, 4]               131,584
# │    │    └─ReLU: 3-60                   [1, 256, 4]               --
# │    │    └─Conv1d: 3-61                 [1, 256, 4]               65,792
# │    └─CondResnetBlock: 2-5              [1, 256, 4]               --
# │    │    └─CondBatchNorm: 3-62          [1, 256, 4]               131,584
# │    │    └─ReLU: 3-63                   [1, 256, 4]               --
# │    │    └─Conv1d: 3-64                 [1, 256, 4]               65,792
# │    │    └─CondBatchNorm: 3-65          [1, 256, 4]               131,584
# │    │    └─ReLU: 3-66                   [1, 256, 4]               --
# │    │    └─Conv1d: 3-67                 [1, 256, 4]               65,792
# │    └─CondResnetBlock: 2-6              [1, 256, 4]               --
# │    │    └─CondBatchNorm: 3-68          [1, 256, 4]               131,584
# │    │    └─ReLU: 3-69                   [1, 256, 4]               --
# │    │    └─Conv1d: 3-70                 [1, 256, 4]               65,792
# │    │    └─CondBatchNorm: 3-71          [1, 256, 4]               131,584
# │    │    └─ReLU: 3-72                   [1, 256, 4]               --
# │    │    └─Conv1d: 3-73                 [1, 256, 4]               65,792
# │    └─CondResnetBlock: 2-7              [1, 256, 4]               --
# │    │    └─CondBatchNorm: 3-74          [1, 256, 4]               131,584
# │    │    └─ReLU: 3-75                   [1, 256, 4]               --
# │    │    └─Conv1d: 3-76                 [1, 256, 4]               65,792
# │    │    └─CondBatchNorm: 3-77          [1, 256, 4]               131,584
# │    │    └─ReLU: 3-78                   [1, 256, 4]               --
# │    │    └─Conv1d: 3-79                 [1, 256, 4]               65,792
# │    └─CondBatchNorm: 2-8                [1, 256, 4]               --
# │    │    └─Linear: 3-80                 [1, 256]                  65,792
# │    │    └─Linear: 3-81                 [1, 256]                  65,792
# │    │    └─BatchNorm1d: 3-82            [1, 256, 4]               --
# │    └─ReLU: 2-9                         [1, 256, 4]               --
# │    └─Conv1d: 2-10                      [1, 1, 4]                 257
# ==========================================================================================
# Total params: 23,318,337
# Trainable params: 23,318,337
# Non-trainable params: 0
# Total mult-adds (G): 15.37
# ==========================================================================================
# Input size (MB): 0.60
# Forward/backward pass size (MB): 216.90
# Params size (MB): 93.27
# Estimated Total Size (MB): 310.78
# ==========================================================================================