from os import fchdir
import torch
import torch.nn as nn

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
        self.fc_final = nn.Linear(f_dim, 1)

    def forward(self, p, raw_c):
        '''
        args:
        raw_c: direct output of encoder (batch_size, ?)
        p: a batch of T 3D coordinates (batch_size, 3, T)
        '''   
        c = torch.flatten(raw_c, start_dim = 1) # from encoder_out=(batch_sizez, ?) to (batch_size, c_dim)
        c = torch.unsqueeze(c, 2) # (batch_size, c_dim, 1)
        # TODO: what is this c_dim?
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
        self.relu = nn.Relu()
        self.cbn1 = CondBatchNorm(c_dim, f_dim)
        self.fc1 = nn.Linear(f_dim, f_dim)
        self.cbn2 = CondBatchNorm(c_dim, f_dim)
        self.fc2 = nn.Linear(f_dim, f_dim)
    
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
        # NOTE: f_dim: C from input of size (N,C,L) or L from input of size (N,L)
    
    def forward(self, f_in, c):
        '''
        args:
        f_in: feature vectors of T 3D coordinates (batch_size, f_dim, T)
        c: flattened 1-dimensional output embedding of encoder (batch_size, c_dim, 1)
        '''
        batch_size = c.size(0) # also equivalent to f_in.size(0)
        gamma_c = self.fc_gamma(c) # (batch_size x f_dim x 1)
        beta_c = self.fc_beta(c) # (batch_size f_dim x 1)
        # gamma_c = gamma_c.view(batch_size, self.f_dim, 1)
        # beta_c = beta_c.view(batch_size, self.f_dim, 1) 
        normalized_f_in = self.bn(f_in) # (batch_size, f_dim, T)?
        return normalized_f_in * gamma_c + beta_c # (batch_size, f_dim, T)?


































class ref_CResnetBlockConv1d(nn.Module):
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


class ref_CBatchNorm1d(nn.Module):
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
        # Affine mapping -> (batch_size, f_dim, T)?
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)
        # Batchnorm
        net = self.bn(x)  # f_out
        out = gamma * net + beta

        return out