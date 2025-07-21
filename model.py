from typing import ClassVar

import chess
import torch
from config import config
from torch import nn
from torch.nn.functional import softmax

BOARD_SIZE = 8
PIECE_TYPES = 6
COLORS = 2
TOTAL_SQUARES = 64

TURN_CHANNEL = 12
CASTLING_WHITE_KS_CHANNEL = 13
CASTLING_WHITE_QS_CHANNEL = 14
CASTLING_BLACK_KS_CHANNEL = 15
CASTLING_BLACK_QS_CHANNEL = 16
EN_PASSANT_CHANNEL = 17


class MoveEncoder:
    def __init__(self):
        self.move_to_idx: dict[chess.Move, int] = {}
        self._build_move_map()

    def _build_move_map(self):
        idx = 0

        for from_sq in range(TOTAL_SQUARES):
            for to_sq in range(TOTAL_SQUARES):
                if from_sq != to_sq:
                    is_white_promotion = (48 <= from_sq <= 55) and (56 <= to_sq <= 63)
                    is_black_promotion = (8 <= from_sq <= 15) and (0 <= to_sq <= 7)

                    if not (is_white_promotion or is_black_promotion):
                        move = chess.Move(from_sq, to_sq)
                        self.move_to_idx[move] = idx
                        idx += 1

        promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

        for from_sq in range(48, 56):
            for to_sq in range(56, 64):
                if abs((from_sq % 8) - (to_sq % 8)) <= 1:
                    for promotion_piece in promotion_pieces:
                        move = chess.Move(from_sq, to_sq, promotion_piece)
                        self.move_to_idx[move] = idx
                        idx += 1

        for from_sq in range(8, 16):
            for to_sq in range(8):
                if abs((from_sq % 8) - (to_sq % 8)) <= 1:
                    for promotion_piece in promotion_pieces:
                        move = chess.Move(from_sq, to_sq, promotion_piece)
                        self.move_to_idx[move] = idx
                        idx += 1

    def encode_move(self, move: chess.Move) -> int:
        return self.move_to_idx.get(move, 0)

    def size(self) -> int:
        return len(self.move_to_idx)


class ChessModel(nn.Module):
    PIECE_OFFSETS: ClassVar = {
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

    def __init__(self, device: str = "auto"):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        board_encoding_size = (
            BOARD_SIZE * BOARD_SIZE * config.ENCODING_CHANNELS
        )
        self.board_encoder = nn.Linear(board_encoding_size, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.NUM_HEADS,
            dim_feedforward=self.hidden_dim * config.FFN_MULTIPLIER,
            dropout=config.DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.NUM_LAYERS
        )

        self.policy_head = nn.Linear(self.hidden_dim, config.MOVE_SPACE_SIZE)
        self.value_head = nn.Linear(self.hidden_dim, 1)

        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _fill_channel(self, tensor: torch.Tensor, channel: int, value: float):
        for square in range(TOTAL_SQUARES):
            row, col = square // BOARD_SIZE, square % BOARD_SIZE
            flat_idx = (
                row * BOARD_SIZE * config.ENCODING_CHANNELS
                + col * config.ENCODING_CHANNELS
                + channel
            )
            tensor[flat_idx] = value

    def encode_board(self, board: chess.Board) -> torch.Tensor:
        tensor = torch.zeros(
            BOARD_SIZE * BOARD_SIZE * config.ENCODING_CHANNELS,
            dtype=torch.float32,
        )

        for square, piece in board.piece_map().items():
            row, col = square // BOARD_SIZE, square % BOARD_SIZE
            channel = self.PIECE_OFFSETS[(piece.piece_type, piece.color)]
            flat_idx = (
                row * BOARD_SIZE * config.ENCODING_CHANNELS
                + col * config.ENCODING_CHANNELS
                + channel
            )
            tensor[flat_idx] = 1.0

        turn_value = 1.0 if board.turn == chess.WHITE else 0.0
        self._fill_channel(tensor, TURN_CHANNEL, turn_value)

        castling_channels = [CASTLING_WHITE_KS_CHANNEL, CASTLING_WHITE_QS_CHANNEL, CASTLING_BLACK_KS_CHANNEL, CASTLING_BLACK_QS_CHANNEL]
        castling_rights = [
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ]

        for i, has_right in enumerate(castling_rights):
            channel = castling_channels[i]
            value = 1.0 if has_right else 0.0
            self._fill_channel(tensor, channel, value)

        if board.ep_square is not None:
            ep_file = board.ep_square % 8
            ep_value = ep_file / 7.0
        else:
            ep_value = -1.0 / 7.0
        self._fill_channel(tensor, EN_PASSANT_CHANNEL, ep_value)

        return tensor

    def forward(self, board_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.board_encoder(board_tensor)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)

        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        policy = softmax(policy_logits, dim=-1)

        return {"policy": policy, "value": value}
