from typing import Dict, List, Optional, NamedTuple
import numpy as np
import torch
import torch.nn as nn

class NetworkOutput(NamedTuple):
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor
    hidden_state: torch.Tensor


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, expansion = 4, cardinality = 1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channels*expansion, channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups = cardinality, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.Conv2d(channels, channels*expansion, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels*expansion))
        self.relu = nn.ReLU(inplace = True)
        
        
    def forward(self, x):
        res = x
        out = self.block(x)
        out = self.relu(out+res)
        return out


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.config = config
        self.device = config.device
        self.h = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace = True),
                                        ResidualBlock(4),
                                        ResidualBlock(4))
        
        self.f = ResidualBlock(4)
        self.fcc = nn.Linear(3*3*16, 10)
        self.tanh = nn.Tanh()

        self.g = nn.Sequential(nn.Conv2d(16+1, 16, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace = True),
                                        ResidualBlock(4),
                                        ResidualBlock(4))
        self.s = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.r = nn.Linear(3*3*16, 1)


    def f_head(self, s):
        f = self.f(s).view(-1, 3*3*16)
        f = self.fcc(f)
        pred = f[:, 0:9]
        v = self.tanh(f[:, -1])
        return v, pred

    def initial_inference(self, image):
        # representation + prediction function
        s = self.h(image)
        R, C, H, W = s.size()
        v, pred = self.f_head(s)
        return NetworkOutput(v, torch.zeros(R, device=self.device), pred, s)


    def recurrent_inference(self, hidden_state, actions): #actions is list of action (integers or tensor i think also works)
        # dynamics + prediction function
        R, C, H, W = hidden_state.size()
        a = torch.zeros((R, 1, 3, 3), device=self.device)
        for i, action in enumerate(actions):
            a[i, 0, action//3, action%3] = 1

        s = self.g( torch.cat([hidden_state, a], dim=1) )
        r = self.r( s.view(-1, 16*3*3) )
        r = self.tanh(r[:, -1])
        s = self.s(s)
        v, pred = self.f_head(s)

        return NetworkOutput(v, r, pred, s)
