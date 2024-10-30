import torch
import torch.nn as nn
import src.utils.chess_utils as chess_utils
from src.models.layers import SEResidualUnit


class ChessModel(nn.Module):
    def __init__(self, filters=64, res_blocks=10, num_moves=None):
        super(ChessModel, self).__init__()
        if num_moves is None:
            num_moves = chess_utils.TOTAL_MOVES
        self.num_moves = num_moves

        self.initial_conv = nn.Conv2d(20, filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)

        layers = [SEResidualUnit(filters, filters) for _ in range(res_blocks)]
        self.residual_layers = nn.Sequential(*layers)

        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(2 * 8 * 8, self.num_moves)

        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 128)
        self.value_fc1_bn = nn.BatchNorm1d(128)
        self.value_fc1_relu = nn.ReLU(inplace=True)
        self.value_fc2 = nn.Linear(128, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.initial_bn(self.initial_conv(x)))
        x = self.residual_layers(x)

        p = self.policy_relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        v = self.value_relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = self.value_fc1(v)
        v = self.value_fc1_bn(v)
        v = self.value_fc1_relu(v)
        v = torch.tanh(self.value_fc2(v))

        return p, v