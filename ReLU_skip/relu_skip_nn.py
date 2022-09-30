import torch.nn as nn

class ReLUskip(nn.Module):
    def __init__(self, m, n, d):
        super(ReLUskip, self).__init__()
        self.m = m
        self.n = n
        self.d = d

        self.w = nn.Linear(self.d,self.m,bias=False)
        self.alpha = nn.Linear(self.m,1,bias=False)
        self.w0 = nn.Linear(self.d,1,bias=False)
        self.alpha0 = nn.Linear(1, 1, bias=False)


    def forward(self, X):
        rl = nn.ReLU()
        Xu = rl(self.w(X))
        y = self.alpha(Xu) + self.alpha0(self.w0(X))
        return y

    def name(self):
        return "ReLU_network_with_skip_connection"