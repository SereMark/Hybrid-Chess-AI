import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    def __init__(self, blocks=None, channels=None):
        super().__init__()
        blocks = blocks or Config.BLOCKS
        channels = channels or Config.CHANNELS
        policy_planes = Config.POLICY_OUTPUT // 64

        self.conv_in = nn.Conv2d(Config.INPUT_PLANES, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Conv2d(channels, policy_planes, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_planes)
        self.policy_fc = nn.Linear(policy_planes * 64, Config.POLICY_OUTPUT)

        self.value_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.flatten(1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.flatten(1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)

        return policy, value

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
