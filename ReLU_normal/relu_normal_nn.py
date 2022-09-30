import torch
import torch.nn as nn
import torch.nn.functional as nnfun


class ReLUnormal(nn.Module):
    def __init__(self, m, n, d):
        super(ReLUnormal, self).__init__()
        self.m = m
        self.n = n
        self.d = d

        self.w = nn.Linear(self.d, self.m, bias=False)
        self.alpha = nn.Linear(self.m, 1, bias=False)
        self.gamma = nn.Parameter(torch.ones(self.m))

    def forward(self, X):
        Xu = nnfun.relu(self.w(X))
        y1 = self.gamma.mul(nnfun.normalize(Xu,dim=0))
        y = self.alpha(y1)
        return y

    def name(self):
        return "ReLU_network_with_normalization_layer"
