from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from main import get_config

logger = logging.getLogger(__name__)


class ChessModel(nn.Module):
    def __init__(self, 
                 hidden_dim: int = None, 
                 num_layers: int = None, 
                 num_heads: int = None,
                 device: str = 'auto') -> None:
        super().__init__()
        self.hidden_dim = hidden_dim or get_config('model', 'hidden_dim')
        num_layers = num_layers or get_config('model', 'num_layers')
        num_heads = num_heads or get_config('model', 'num_heads')
        self.device = self._setup_device(device)
        board_size = get_config('model', 'board_size')
        piece_types = get_config('model', 'piece_types')
        colors = get_config('model', 'colors')
        board_encoding_size = board_size * board_size * piece_types * colors
        self.board_encoder = nn.Linear(board_encoding_size, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * get_config('model', 'ffn_multiplier'),
            dropout=get_config('model', 'dropout'),
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(self.hidden_dim, get_config('model', 'move_space_size'))
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self.apply(self._init_weights)
        device_start = time.time()
        self.to(self.device)
        device_time = time.time() - device_start
        logger.debug(f"Device transfer: {device_time:.3f}s")
        param_count = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.board_encoder.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        value_params = sum(p.numel() for p in self.value_head.parameters())
        logger.info(f"Model ready: {param_count:,} parameters on {self.device}")
        logger.debug(f"Parameter breakdown:")
        logger.debug(f"  Board encoder: {encoder_params:,} ({encoder_params / param_count * 100:.1f}%)")
        logger.debug(f"  Transformer: {transformer_params:,} ({transformer_params / param_count * 100:.1f}%)")
        logger.debug(f"  Policy head: {policy_params:,} ({policy_params / param_count * 100:.1f}%)")
        logger.debug(f"  Value head: {value_params:,} ({value_params / param_count * 100:.1f}%)")
        model_size_mb = param_count * 4 / 1024**2
        logger.debug(f"Model memory footprint: {model_size_mb:.1f}MB")
        if param_count > 50_000_000:
            logger.warning(f"Large model: {param_count:,} parameters ({model_size_mb:.1f}MB)")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            logger.debug(f"Initializing Linear layer: {module.weight.shape}")
            torch.nn.init.normal_(module.weight, mean=0.0, std=get_config('training', 'weight_init_std'))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                logger.debug(f"  - Bias initialized to zero: {module.bias.shape}")

    def encode_board(self, board: chess.Board) -> torch.Tensor:
        if logger.isEnabledFor(logging.DEBUG):
            encode_start = time.time()
        board_size = get_config('model', 'board_size')
        piece_types = get_config('model', 'piece_types')
        colors = get_config('model', 'colors')
        tensor = torch.zeros(board_size, board_size, piece_types * colors, dtype=torch.float32)
        piece_count = 0
        piece_to_index = get_config('game', 'piece_to_index')
        
        piece_type_to_name = {
            chess.PAWN: 'pawn',
            chess.KNIGHT: 'knight', 
            chess.BISHOP: 'bishop',
            chess.ROOK: 'rook',
            chess.QUEEN: 'queen',
            chess.KING: 'king'
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = square // board_size
                col = square % board_size
                piece_name = piece_type_to_name[piece.piece_type]
                piece_idx = piece_to_index[piece_name]
                if piece.color == chess.BLACK:
                    piece_idx += piece_types
                tensor[row, col, piece_idx] = 1.0
                piece_count += 1
        if logger.isEnabledFor(logging.DEBUG):
            encode_time = time.time() - encode_start
            logger.debug(f"Board encoded: {piece_count} pieces in {encode_time * 1000:.2f}ms")
        return tensor.flatten()

    def forward(self, board_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = board_tensor.size(0) if board_tensor.dim() > 1 else 1
        board_size = get_config('model', 'board_size')
        piece_types = get_config('model', 'piece_types')
        colors = get_config('model', 'colors')
        board_encoding_size = board_size * board_size * piece_types * colors
        if board_tensor.dim() != 2 or board_tensor.size(1) != board_encoding_size:
            raise RuntimeError(f"Expected input shape [batch_size, {board_encoding_size}], "
                             f"got {board_tensor.shape}")
        if batch_size > 128:
            logger.warning(f"Large batch size: {batch_size} (may cause memory issues)")
        if logger.isEnabledFor(logging.DEBUG):
            forward_start = time.time()
            logger.debug(f"Forward pass: batch_size={batch_size}, input_shape={board_tensor.shape}")
        if logger.isEnabledFor(logging.DEBUG):
            encoder_start = time.time()
        x = self.board_encoder(board_tensor)
        if logger.isEnabledFor(logging.DEBUG):
            encoder_time = time.time() - encoder_start
            logger.debug(f"Board encoder: {encoder_time * 1000:.2f}ms, output_shape={x.shape}")
        x = x.unsqueeze(1)
        if logger.isEnabledFor(logging.DEBUG):
            transformer_start = time.time()
        x = self.transformer(x)
        if logger.isEnabledFor(logging.DEBUG):
            transformer_time = time.time() - transformer_start
            logger.debug(f"Transformer: {transformer_time * 1000:.2f}ms, output_shape={x.shape}")
        x = x.squeeze(1)
        if logger.isEnabledFor(logging.DEBUG):
            heads_start = time.time()
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        if logger.isEnabledFor(logging.DEBUG):
            heads_time = time.time() - heads_start
            logger.debug(f"Output heads: {heads_time * 1000:.2f}ms")
        result = {
            'policy': F.softmax(policy_logits, dim=-1),
            'value': value
        }
        if logger.isEnabledFor(logging.DEBUG):
            total_time = time.time() - forward_start
            logger.debug(f"Total forward pass: {total_time * 1000:.2f}ms")
            logger.debug(f"Output shapes - policy: {result['policy'].shape}, value: {result['value'].shape}")
        return result

    def _setup_device(self, device: str) -> str:
        logger.debug(f"Setting up device: {device}")
        if device == 'cuda':
            if not torch.cuda.is_available():
                logger.error("CUDA requested but not available")
                raise RuntimeError("CUDA requested but not available")
            selected_device = 'cuda'
        elif device == 'cpu':
            logger.debug("CPU device explicitly selected")
            selected_device = 'cpu'
        elif device == 'auto':
            if torch.cuda.is_available():
                selected_device = 'cuda'
                logger.debug("Auto-selected CUDA device")
            else:
                selected_device = 'cpu'
                logger.warning("CUDA not available, falling back to CPU")
                logger.warning("Training will be slower on CPU")
        else:
            raise ValueError(f"Unsupported device: {device}. Use 'cuda', 'cpu', or 'auto'")
        if selected_device == 'cuda':
            props = torch.cuda.get_device_properties(0)
            logger.info(f"Using GPU: {props.name}")
            logger.debug(f"GPU compute capability: {props.major}.{props.minor}")
            logger.debug(f"GPU memory: {props.total_memory / 1024**3:.1f}GB")
            logger.debug(f"GPU multiprocessors: {props.multi_processor_count}")
        else:
            logger.info("Using CPU for computations")
        return selected_device


class MoveEncoder:
    def __init__(self) -> None:
        self.move_to_idx: Dict[chess.Move, int] = {}
        self.idx_to_move: Dict[int, chess.Move] = {}
        self._build_move_map()

    def _build_move_map(self) -> None:
        idx = 0
        total_squares = get_config('game', 'total_squares')
        for from_sq in range(total_squares):
            for to_sq in range(total_squares):
                if from_sq != to_sq:
                    move = chess.Move(from_sq, to_sq)
                    self.move_to_idx[move] = idx
                    self.idx_to_move[idx] = move
                    idx += 1
        promotion_pieces = get_config('game', 'promotion_pieces')
        pawn_promotion_ranks = get_config('game', 'pawn_promotion_ranks')
        for color_name in ['white', 'black']:
            color = chess.WHITE if color_name == 'white' else chess.BLACK
            from_start, from_end, to_start, to_end = pawn_promotion_ranks[color_name]
            for from_sq in range(from_start, from_end):
                for to_sq in range(to_start, to_end):
                    for promo in promotion_pieces:
                        move = chess.Move(from_sq, to_sq, promotion=promo)
                        self.move_to_idx[move] = idx
                        self.idx_to_move[idx] = move
                        idx += 1

    def encode_move(self, move: chess.Move) -> int:
        return self.move_to_idx.get(move, 0)

    def decode_move(self, idx: int) -> Optional[chess.Move]:
        return self.idx_to_move.get(idx)

    def encode_legal_moves(self, legal_moves: List[chess.Move]) -> torch.Tensor:
        move_space_size = get_config('model', 'move_space_size')
        mask = torch.zeros(move_space_size, dtype=torch.float32)
        for move in legal_moves:
            idx = self.encode_move(move)
            if idx < move_space_size:
                mask[idx] = 1.0
        return mask

    def sample_move(self, 
                   policy: torch.Tensor, 
                   legal_moves: List[chess.Move],   
                   temperature: float = 1.0) -> Optional[chess.Move]:
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if not legal_moves:
            logger.debug("No legal moves available")
            return None
        logger.debug(f"Sampling from {len(legal_moves)} legal moves with temperature {temperature}")
        move_probs = []
        valid_moves = []
        for move in legal_moves:
            idx = self.encode_move(move)
            if idx < len(policy):
                move_probs.append(policy[idx].item())
                valid_moves.append(move)
        if not valid_moves:
            logger.warning(f"No valid moves in policy (checked {len(legal_moves)} moves), using first legal move")
            return legal_moves[0]
        move_probs = np.array(move_probs, dtype=np.float64)
        if temperature != 1.0:
            move_probs = np.power(move_probs, 1.0 / temperature)
        move_probs = move_probs / np.sum(move_probs)
        selected_move = np.random.choice(valid_moves, p=move_probs)
        if logger.isEnabledFor(logging.DEBUG):
            top_indices = np.argsort(move_probs)[-3:][::-1]
            top_moves_info = [f"{valid_moves[i]} ({move_probs[i]:.3f})" for i in top_indices]
            logger.debug(f"Top moves: {', '.join(top_moves_info)}")
            logger.debug(f"Selected: {selected_move}")
        return selected_move
