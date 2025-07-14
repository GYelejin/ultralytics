import math
import torch
import torch.nn as nn
import torchkan
from .conv import autopad
    
class ConvWithKAN(nn.Module):
    default_act = nn.SiLU() 
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, kan_type="KANConv2DLayer"):
        super().__init__()
        if kan_type == "FastKANConv2DLayer":
            self.conv = torchkan.FastKANLayer(
                c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d), groups=g, dilation=d
            )
        else:
            self.conv = torchkan.KANConv2DLayer(
                c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d), groups=g, dilation=d
            )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.conv(x)

class DWConvWithKAN(ConvWithKAN):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)