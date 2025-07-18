import logging

import chess
import torch
from config import get_config
from torch import nn
from torch.nn import functional

logger = logging.getLogger(__name__)

LARGE_BATCH_WARNING_THRESHOLD = 160
LARGE_MODEL_THRESHOLD = 50_000_000


class ChessModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        num_heads: int | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim or get_config("model", "hidden_dim")
        num_layers = num_layers or get_config("model", "num_layers")
        num_heads = num_heads or get_config("model", "num_heads")
        assert num_layers is not None
        assert num_heads is not None
        self.device: str = self._setup_device(device)
        board_size = get_config("model", "board_size")
        piece_types = get_config("model", "piece_types")
        colors = get_config("model", "colors")
        board_encoding_size = board_size * board_size * piece_types * colors
        self.board_encoder: nn.Linear = nn.Linear(board_encoding_size, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * get_config("model", "ffn_multiplier"),
            dropout=get_config("model", "dropout"),
            batch_first=True,
        )
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.policy_head: nn.Linear = nn.Linear(
            self.hidden_dim, get_config("model", "move_space_size")
        )
        self.value_head: nn.Linear = nn.Linear(self.hidden_dim, 1)
        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.TransformerEncoderLayer):
            for name, param in module.named_parameters():
                if "weight" in name and param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param, gain=0.8)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)

    def encode_board_vectorized(self, board: chess.Board) -> torch.Tensor:
        board_size = get_config("model", "board_size")
        piece_types = get_config("model", "piece_types")
        colors = get_config("model", "colors")

        tensor = torch.zeros(
            board_size * board_size * piece_types * colors, dtype=torch.float32
        )

        piece_map = board.piece_map()
        if not piece_map:
            return tensor

        piece_offsets = {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11,
        }

        indices = []
        for square, piece in piece_map.items():
            row = square // board_size
            col = square % board_size
            channel = piece_offsets[(piece.piece_type, piece.color)]
            flat_idx = (
                row * board_size * piece_types * colors
                + col * piece_types * colors
                + channel
            )
            indices.append(flat_idx)

        if indices:
            tensor[indices] = 1.0

        return tensor

    def forward(self, board_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = board_tensor.size(0) if board_tensor.dim() > 1 else 1
        board_size = get_config("model", "board_size")
        piece_types = get_config("model", "piece_types")
        colors = get_config("model", "colors")
        board_encoding_size = board_size * board_size * piece_types * colors
        if board_tensor.dim() != 2 or board_tensor.size(1) != board_encoding_size:
            raise RuntimeError(
                f"Expected input shape [batch_size, {board_encoding_size}], "
                + f"got {board_tensor.shape}"
            )
        if batch_size > LARGE_BATCH_WARNING_THRESHOLD:
            logger.warning(f"Large batch size: {batch_size} (may cause memory issues)")
        x = self.board_encoder(board_tensor)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))

        policy_logits = torch.clamp(policy_logits, min=-10.0, max=10.0)

        policy = functional.softmax(policy_logits, dim=-1)

        epsilon = 1e-8
        policy = policy + epsilon
        policy = policy / policy.sum(dim=-1, keepdim=True)

        if torch.any(torch.isnan(policy)) or torch.any(torch.isinf(policy)):
            logger.warning("Invalid policy output detected, using uniform fallback")
            batch_size = policy.size(0)
            move_space_size = policy.size(1)
            policy = (
                torch.ones(batch_size, move_space_size, device=policy.device)
                / move_space_size
            )

        if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
            logger.warning("Invalid value output detected, using zero fallback")
            value = torch.zeros_like(value)

        result = {"policy": policy, "value": value}
        return result

    def _setup_device(self, device: str) -> str:
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.error("CUDA requested but not available")
                raise RuntimeError("CUDA requested but not available")
            selected_device = "cuda"
        elif device == "cpu":
            selected_device = "cpu"
        elif device == "auto":
            if torch.cuda.is_available():
                selected_device = "cuda"
            else:
                selected_device = "cpu"
        else:
            raise ValueError(
                f"Unsupported device: {device}. Use 'cuda', 'cpu', or 'auto'"
            )
        return selected_device
