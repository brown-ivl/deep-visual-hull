from os import fchdir
import torch
import torch.nn as nn
from torchinfo import summary

class DeepVisualHullDecoder(nn.Module):
    def __init__(self, c_dim=256, f_dim=256): 
        super().__init__()
        self.fc_p = nn.Conv1d(3, f_dim, kernel_size=1)
        self.block1 = CondResnetBlock(c_dim, f_dim)
        self.block2 = CondResnetBlock(c_dim, f_dim)
        self.block3 = CondResnetBlock(c_dim, f_dim)
        self.block4 = CondResnetBlock(c_dim, f_dim)
        self.block5 = CondResnetBlock(c_dim, f_dim)
        self.cbn = CondBatchNorm(c_dim, f_dim)
        self.relu = nn.ReLU()
        self.fc_final = nn.Conv1d(f_dim, 1, kernel_size=1)

    def forward(self, p, c):
        '''
        args:
        c: output of encoder (batch_size, 256)
        p: a batch of T 3D coordinates (batch_size, 3, T=resolution**3)
        '''
        f_in = self.fc_p(p) # (batch_size, 3, T) -> (batch_size, f_dim, T)
        x = self.block1(f_in, c)
        x = self.block2(x, c)
        x = self.block3(x, c)
        x = self.block4(x, c)
        x = self.block5(x, c) # (batch_size, f_dim, T)
        x = self.cbn(x, c) # (batch_size, f_dim, T)
        x = self.relu(x) # (batch_size, f_dim, T)
        out = self.fc_final(x) # (batch_size, 1, T)
        return torch.sigmoid(out) # (batch_size, 1, T)


class CondResnetBlock(nn.Module):
    def __init__(self, c_dim, f_dim=256):
        '''
        args:
        c_dim: dimension of c (flattened encoder output)
        '''
        super().__init__()
        self.relu = nn.ReLU()
        self.cbn1 = CondBatchNorm(c_dim, f_dim)
        self.fc1 = nn.Conv1d(f_dim, f_dim, kernel_size=1)
        self.cbn2 = CondBatchNorm(c_dim, f_dim)
        self.fc2 = nn.Conv1d(f_dim, f_dim, kernel_size=1)
    
    def forward(self, f_in, c):
        '''
        args:
        f_in: feature vectors of T 3D coordinates (batch_size, f_dim, T)
        c: flattened 1-dimensional output embedding of encoder (batch_size, c_dim, 1)
        '''
        x = self.cbn1(f_in, c) # (batch_size, f_dim, T)?
        x = self.relu(x)
        x = self.fc1(x)
        x = self.cbn2(x, c)
        x = self.relu(x)
        x = self.fc2(x) # (batch_size, f_dim, T)?
        return x + f_in


class CondBatchNorm(nn.Module):
    def __init__(self, c_dim, f_dim=256):
        '''
        args:
        c_dim: dimension of c (flattened encoder output)
        f_dim: dimension of 1 feature vector for 1 3D coordinate
        '''
        super().__init__()
        self.fc_gamma = nn.Linear(c_dim, f_dim)
        self.fc_beta = nn.Linear(c_dim, f_dim)
        self.bn = nn.BatchNorm1d(f_dim, affine=False, eps=1e-5, momentum=0.1)  # set beta(0) and gamma(1) to be not learnable
    
    def forward(self, f_in, c):
        '''
        args:
        f_in: feature vectors of T 3D coordinates (batch_size, f_dim, T)
        c: flattened 1-dimensional output embedding of encoder (batch_size, c_dim, 1)
        '''
        batch_size = c.size(0) # also equivalent to f_in.size(0)
        gamma_c = self.fc_gamma(c) # (batch_size x f_dim x 1)
        gamma_c = torch.unsqueeze(gamma_c, 2)
        beta_c = self.fc_beta(c) # (batch_size x f_dim x 1)
        beta_c = torch.unsqueeze(beta_c, 2)
        normalized_f_in = self.bn(f_in) # (batch_size, f_dim, T)
        return normalized_f_in * gamma_c + beta_c # (batch_size, f_dim, T)


# decoder = DeepVisualHullDecoder()
# summary(decoder, [(1,3,4), (1,256)])