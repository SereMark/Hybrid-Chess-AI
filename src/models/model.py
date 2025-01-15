import torch
import torch.nn as nn

class ResidualUnit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(48)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)


class ChessModel(nn.Module):
    """
    ChessModel: A neural network for chess AI tasks.

    This model predicts:
    1. **Policy Head**: Logits for all possible moves.
    2. **Value Head**: A scalar evaluation of the board state in [-1, 1].

    Architecture:
    - Input: A tensor of shape (batch_size, 25, 8, 8) representing the chessboard state.
    - Initial Block: Convolutional preprocessing layer.
    - Residual Layers: Three stacked residual units to enhance feature extraction.
    - Policy Head: Outputs move logits with shape (batch_size, num_moves).
    - Value Head: Outputs a scalar evaluation with shape (batch_size, 1).

    Parameters:
    - `num_moves` (int): Number of possible moves (output size of the policy head).

    Forward Method:
    - `forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`:
      Returns:
        - `policy_output`: Logits for moves.
        - `value_output`: Scalar evaluation of the board state.
    """
    def __init__(self, num_moves: int) -> None:
        super().__init__()
        # Initial convolutional block
        self.initial_block = nn.Sequential(
            nn.Conv2d(25, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Residual layers
        self.residual_layers = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit()
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(48, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, num_moves)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(48, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_block(x)
        x = self.residual_layers(x)
        policy_output = self.policy_head(x)
        value_output = self.value_head(x)
        return policy_output, value_output