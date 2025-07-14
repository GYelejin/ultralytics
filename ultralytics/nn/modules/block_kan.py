import torch
import torch.nn as nn
from .conv_kan import ConvWithKAN

class BottleneckWithKAN(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, kan_type="KANConv2DLayer"):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvWithKAN(c1, c_, k[0], 1, kan_type=kan_type)
        self.cv2 = ConvWithKAN(c_, c2, k[1], 1, g=g, kan_type=kan_type)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2fKAN(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvWithKAN(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvWithKAN((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(BottleneckWithKAN(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPFWithKAN(nn.Module):
    def __init__(self, c1, c2, k=5, kan_type="KANConv2DLayer"):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvWithKAN(c1, c_, 1, 1, kan_type=kan_type)
        self.cv2 = ConvWithKAN(c_ * 4, c2, 1, 1, kan_type=kan_type)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class C3kKAN(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, kan_type="KANConv2DLayer"):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvWithKAN(c1, c_, 1, 1, kan_type=kan_type)
        self.cv2 = ConvWithKAN(c_, c2, 3, 1, g=g, kan_type=kan_type)
        self.m = nn.Sequential(*[BottleneckWithKAN(c_, c_, shortcut, g, kan_type=kan_type) for _ in range(n)])
    def forward(self, x):
        return self.cv2(self.m(self.cv1(x)))


class AAttnKAN(nn.Module):
    def __init__(self, dim: int, num_heads: int, area: int = 1, kan_type="KANConv2DLayer"):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = ConvWithKAN(dim, all_head_dim * 3, 1, 1, act=False, kan_type=kan_type)
        self.proj = ConvWithKAN(all_head_dim, dim, 1, 1, act=False, kan_type=kan_type)
        self.pe = ConvWithKAN(all_head_dim, dim, 7, 1, p=3, g=dim, act=False, kan_type=kan_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)

class ABlockKAN(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1, kan_type="KANConv2DLayer"):
        super().__init__()
        self.attn = AAttnKAN(dim, num_heads=num_heads, area=area, kan_type=kan_type)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            ConvWithKAN(dim, mlp_hidden_dim, 1, 1, kan_type=kan_type),
            ConvWithKAN(mlp_hidden_dim, dim, 1, 1, act=False, kan_type=kan_type)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)

class A2C2fKAN(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        kan_type: str = "KANConv2DLayer"
    ):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."
        self.cv1 = ConvWithKAN(c1, c_, 1, 1, kan_type=kan_type)
        self.cv2 = ConvWithKAN((1 + n) * c_, c2, 1, 1, kan_type=kan_type)
        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlockKAN(c_, c_ // 32, mlp_ratio, area, kan_type=kan_type) for _ in range(2)))
            if a2
            else C3kKAN(c_, c_, 2, shortcut, g, kan_type=kan_type)
            for _ in range(n)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y