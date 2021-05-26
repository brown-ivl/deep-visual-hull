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
    def __init__(self, n_classes=4, in_channels=3, is_unpooling=True, Args=None):
        super().__init__(Args)
        self.encoder = DeepVisualHullEncoder() # Seg Net Encoder
        self.decoder = DeepVisualHullDecoder() # Occupancy Network Decoder

    def forward(self, images, points):
        '''
        args:
        images: observations of the object （batch_size, c, w, h)
        points: a batch of 3d points to be passed into the decoder (batch_size, 3, T) or (batch_size, T, 3)
        '''
        B, C, W, H = images.size()
        for b in range(B):
            images[b] = normalizeInput(images[b], format='imagenet')  # assuming input is the range 0-1

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