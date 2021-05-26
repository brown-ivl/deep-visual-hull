# Borrowed from https://github.com/meetshah1995/pytorch-semseg
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, Sequential
import torch.nn.functional as F
import torchvision.models as models
import os, sys
from tk3dv.ptTools import ptUtils
from tk3dv.ptTools import ptNets
import numpy as np


class dvhNet(nn.Module):
    def __init__(self, n_classes=4, in_channels=3, is_unpooling=True, Args=None):
        super().__init__(Args)

        # Seg Net Encoder
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        self.encoder = self.get_segnet_encoder() # TODO

        ## Connection between Encoder and Decoder TODO
        # flatten encoder_out=(batch, 512, 7, 10) to c=(batch, c_dim)

        ## Occupancy Network Decoder
        c_dim = 256 # conv_gamma = conv1d(c_dim, f_dim=hidden_size=256)
        # c_dim: conv_gamma and conv_beta shapes change corresopndingly through construtcor
        hidden_size = 256
        dim = 3
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, hidden_size, hidden_size)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, hidden_size, hidden_size)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, hidden_size, hidden_size)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, hidden_size, hidden_size)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, hidden_size, hidden_size)
        self.bn = CBatchNorm1d(c_dim, f_dim=hidden_size)  # or CBatchNorm1d_legacy
        self.fc_out = nn.Conv1d(hidden_size, 1, 1)
        self.actvn = F.relu

    def forward(self, images, points):
        '''
        args:
        images: observations of the object
        points: a batch of 3d points to be passed into the decoder
        '''
        B, C, W, H = images.size()
        if C == 3:
            # Apply ImageNet batch normalization for input
            for b in range(B):
                images[b] = ptUtils.normalizeInput(images[b], format='imagenet')  # assuming input is the range 0-1

        ## SegNet Encoder TODO
        encoder_out = self.encoder(images)

        ## Connection between Encoder and Decoder TODO
        # flatten encoder_out=(batch, 512, 7, 10) to c=(batch, c_dim)
        # c only used as input to CBatchNorm1d to calculate gamma and beta

        ## Occupancy Network Decoder
        points = points.transpose(1, 2) # batch_size, D, T = points.size()
        # Compute feature vector for each point
        f_in = self.fc_p(points)  # Tx3 -> Tx256
        # Pass feture vector through 5 ResNet blocks (CBN+ReLU+FC+CBN+ReLU+FC)
        net = self.block0(f_in, c)  # return net(f_in) + f_in
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)
        # A last CBN + ReLU + FC that projects down to 1d
        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)
        # p_r = dist.Bernoulli(logits=logits)
        # QUESTION: Bernoulli or sigmoid?

        return out

class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''

    def __init__(self, c_dim, size_in, size_h, size_out):
        super().__init__()
        # Attributes
        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        # QUESTION: conv1d for gamma and beta (kernel-size=1 -> same as linear?
        self.bn_0 = CBatchNorm1d(c_dim, size_in)
        self.bn_1 = CBatchNorm1d(c_dim, size_h)
        self.fc_0 = nn.Conv1d(size_in, size_h, 1)  # 256 -> 256
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)  # 256 -> 256
        self.actvn = nn.ReLU()

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        # x = f_In
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))  # still 256-d
        x_s = x
        return x_s + dx  # x + dx


class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        self.bn = nn.BatchNorm1d(f_dim, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        '''
        Args:
            x: f_in in supplementary paper
        '''
        assert (x.size(0) == c.size(0))  # batch sie
        assert (c.size(1) == self.c_dim)  # c_dim
        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)  # 3-d
        # Affine mapping -> batch_size x f_dim x T?
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)
        # Batchnorm
        net = self.bn(x)  # f_out
        out = gamma * net + beta

        return out
