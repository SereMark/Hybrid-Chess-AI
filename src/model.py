import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, with_relu=True):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=k//2, groups=in_ch, bias=False)
        self.point = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.with_relu = with_relu
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        return self.relu(x) if self.with_relu else x

class Attention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.q = nn.Conv2d(ch, ch//8, kernel_size=1)
        self.k = nn.Conv2d(ch, ch//8, kernel_size=1)
        self.v = nn.Conv2d(ch, ch, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        q = self.q(x).view(b, -1, h*w).permute(0, 2, 1)
        k = self.k(x).view(b, -1, h*w)
        v = self.v(x).view(b, -1, h*w)
        
        attn = F.softmax(torch.bmm(q, k), dim=2)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        
        return self.gamma * out + x

class ResBlock(nn.Module):
    def __init__(self, ch, use_attn=True):
        super().__init__()
        self.conv1 = DSConv(ch, ch, with_relu=True)
        self.conv2 = DSConv(ch, ch, with_relu=False)
        self.attn = Attention(ch) if use_attn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.attn(out + identity)
        return self.relu(out)

class ChessModel(nn.Module):
    def __init__(self, moves, ch=64, blocks=4, use_attn=True):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(23 * 8, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )
        
        self.blocks = nn.Sequential(*[ResBlock(ch, use_attn) for _ in range(blocks)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(ch, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, moves)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(ch, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value