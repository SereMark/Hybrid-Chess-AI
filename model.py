from typing import ClassVar, NamedTuple

import chess
import torch
from config import config
from torch import nn
from torch.nn import functional


class ModelOutput(NamedTuple):
    policy: torch.Tensor
    value: torch.Tensor


class MoveEncoder:
    def __init__(self) -> None:
        self.move_to_idx: dict[chess.Move, int] = {}
        self._build_move_map()

    def _build_move_map(self) -> None:
        idx = 0

        for from_sq in range(config.model.total_squares):
            for to_sq in range(config.model.total_squares):
                if from_sq != to_sq:
                    move = chess.Move(from_sq, to_sq)
                    self.move_to_idx[move] = idx
                    idx += 1

        promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

        for from_sq in range(config.model.total_squares):
            from_rank = from_sq // config.model.board_size
            from_file = from_sq % config.model.board_size
            for to_sq in range(
                config.model.white_promotion_rank_start,
                config.model.white_promotion_rank_end + 1,
            ):
                to_rank = to_sq // config.model.board_size
                to_file = to_sq % config.model.board_size
                if from_rank == 6 and to_rank == 7 and abs(from_file - to_file) <= 1:
                    for promotion_piece in promotion_pieces:
                        move = chess.Move(from_sq, to_sq, promotion_piece)
                        self.move_to_idx[move] = idx
                        idx += 1

        for from_sq in range(config.model.total_squares):
            from_rank = from_sq // config.model.board_size
            from_file = from_sq % config.model.board_size
            for to_sq in range(
                config.model.black_promotion_rank_start,
                config.model.black_promotion_rank_end + 1,
            ):
                to_rank = to_sq // config.model.board_size
                to_file = to_sq % config.model.board_size
                if from_rank == 1 and to_rank == 0 and abs(from_file - to_file) <= 1:
                    for promotion_piece in promotion_pieces:
                        move = chess.Move(from_sq, to_sq, promotion_piece)
                        self.move_to_idx[move] = idx
                        idx += 1

    def encode_move(self, move: chess.Move) -> int:
        idx = self.move_to_idx.get(move, -1)
        if idx == -1:
            raise ValueError(f"Unknown move: {move}")
        return idx


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim: int = dim
        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        self.scale: float = self.head_dim**-0.5

        self.qkv: nn.Linear = nn.Linear(dim, dim * 3, bias=False)
        self.proj: nn.Linear = nn.Linear(dim, dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_multiplier: int, dropout: float):
        super().__init__()
        self.attention: Attention = Attention(dim, num_heads, dropout)
        self.feed_forward: nn.Sequential = nn.Sequential(
            nn.Linear(dim, dim * ffn_multiplier),
            nn.GELU(),
            nn.Linear(dim * ffn_multiplier, dim),
        )
        self.norm1: nn.LayerNorm = nn.LayerNorm(dim)
        self.norm2: nn.LayerNorm = nn.LayerNorm(dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x


class ChessModel(nn.Module):
    PIECE_OFFSETS: ClassVar[dict[tuple[int, chess.Color], int]] = {
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

    def __init__(self, device: str = "auto") -> None:
        super().__init__()
        self.hidden_dim: int = config.token_dim
        if device == "auto":
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.square_embedding: nn.Linear = nn.Linear(
            config.model.encoding_channels, config.model.encoding_channels
        )
        self.piece_embed: nn.Embedding = nn.Embedding(
            config.model.piece_types * config.model.colors + 1,
            config.token_dim - config.model.encoding_channels,
        )

        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                TransformerBlock(
                    config.token_dim,
                    config.model.num_heads,
                    config.model.ffn_multiplier,
                    config.model.dropout,
                )
                for _ in range(config.model.num_layers)
            ]
        )

        self.norm: nn.LayerNorm = nn.LayerNorm(config.token_dim)

        self.policy: nn.Sequential = nn.Sequential(
            nn.Linear(config.token_dim, config.token_dim // 2),
            nn.ReLU(),
            nn.Linear(config.token_dim // 2, config.model.move_space_size),
        )

        self.value: nn.Sequential = nn.Sequential(
            nn.Linear(config.token_dim, config.token_dim // 2),
            nn.ReLU(),
            nn.Linear(config.token_dim // 2, 1),
        )

        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(
                module.weight, gain=config.model.weight_init_gain
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def encode_board(self, board: chess.Board | list[chess.Board]) -> torch.Tensor:
        boards: list[chess.Board] = [board] if isinstance(board, chess.Board) else board
        batch_size = len(boards)

        tensor_shape = (batch_size, config.board_encoding_size)
        batch_tensor = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )

        for batch_idx, b in enumerate(boards):
            for square, piece in b.piece_map().items():
                channel = self.PIECE_OFFSETS[(piece.piece_type, piece.color)]
                flat_idx = square * config.model.encoding_channels + channel
                batch_tensor[batch_idx, flat_idx] = 1.0

            turn_value = 1.0 if b.turn == chess.WHITE else 0.0
            for square in range(config.model.total_squares):
                flat_idx = (
                    square * config.model.encoding_channels + config.model.turn_channel
                )
                batch_tensor[batch_idx, flat_idx] = turn_value

            castling_values = [
                b.has_kingside_castling_rights(chess.WHITE),
                b.has_queenside_castling_rights(chess.WHITE),
                b.has_kingside_castling_rights(chess.BLACK),
                b.has_queenside_castling_rights(chess.BLACK),
            ]
            for i, has_right in enumerate(castling_values):
                for square in range(config.model.total_squares):
                    flat_idx = (
                        square * config.model.encoding_channels
                        + config.model.castling_white_ks_channel
                        + i
                    )
                    batch_tensor[batch_idx, flat_idx] = float(has_right)

            ep_value = b.ep_square / 63.0 if b.ep_square is not None else -1.0 / 63.0
            for square in range(config.model.total_squares):
                flat_idx = (
                    square * config.model.encoding_channels
                    + config.model.en_passant_channel
                )
                batch_tensor[batch_idx, flat_idx] = ep_value

        return (
            batch_tensor.squeeze(0) if isinstance(board, chess.Board) else batch_tensor
        )

    def forward(self, board_input: torch.Tensor) -> ModelOutput:
        if board_input.dim() == 2:
            square_features, piece_ids = self.decode_flat(board_input)
        else:
            raise ValueError(f"Invalid board_input dimensions: {board_input.shape}")

        x = self.square_embedding(square_features)
        piece_emb = self.piece_embed(piece_ids)
        x = torch.cat([x, piece_emb], dim=-1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        x_pooled = x.mean(dim=1)

        policy_logits = self.policy(x_pooled)
        policy = functional.softmax(policy_logits, dim=-1)

        material_bias = self._compute_material_bias(board_input)
        value = torch.tanh(self.value(x_pooled) + material_bias * 0.1)

        return ModelOutput(policy=policy, value=value)

    def _compute_material_bias(self, board_input: torch.Tensor) -> torch.Tensor:
        piece_values = torch.tensor(
            [1, 3, 3, 5, 9, 0], device=board_input.device, dtype=torch.float32
        )

        board_reshaped = board_input.view(board_input.shape[0], 64, 18)
        white_pieces = board_reshaped[:, :, :6]
        black_pieces = board_reshaped[:, :, 6:12]

        white_material = white_pieces * piece_values.unsqueeze(0).unsqueeze(0)
        black_material = black_pieces * piece_values.unsqueeze(0).unsqueeze(0)

        material_balance = (
            white_material.sum(dim=(1, 2)) - black_material.sum(dim=(1, 2))
        ).unsqueeze(1)

        return torch.tanh(material_balance / 10.0)

    def decode_flat(
        self, flat_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = flat_tensor.shape[0]
        square_features = flat_tensor.reshape(
            batch_size, config.model.total_squares, config.model.encoding_channels
        )

        piece_ids = torch.zeros(
            (batch_size, config.model.total_squares),
            dtype=torch.long,
            device=self.device,
        )
        for batch_idx in range(batch_size):
            for square in range(config.model.total_squares):
                piece_channels = square_features[batch_idx, square, :12]
                has_piece = torch.any(piece_channels > 0)
                piece_ids[batch_idx, square] = torch.where(
                    has_piece,
                    piece_channels.argmax(),
                    config.model.piece_types * config.model.colors,
                )

        return square_features, piece_ids
