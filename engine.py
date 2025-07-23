import math
from collections import OrderedDict
from typing import NamedTuple

import chess
import numpy as np
import torch
from config import config
from torch import nn
from torch.nn import functional as f


class ModelOutput(NamedTuple):
    policy: torch.Tensor
    value: torch.Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return f.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = f.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = f.relu(x + residual)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = (channels // num_heads) ** 0.5

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.temperature
        attn = f.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x = self.proj(x)

        x = x.mean(dim=1)
        return x


class MoveEncoder:
    def __init__(self):
        self.move_to_idx = {}
        self.idx_to_move = {}
        self._build_move_mappings()

    def _build_move_mappings(self):
        idx = 0

        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq == to_sq:
                    continue

                from_file = from_sq % 8
                from_rank = from_sq // 8
                to_file = to_sq % 8
                to_rank = to_sq // 8

                file_diff = abs(to_file - from_file)
                rank_diff = abs(to_rank - from_rank)

                if (
                    (file_diff == 2 and rank_diff == 1)
                    or (file_diff == 1 and rank_diff == 2)
                    or (file_diff <= 1 and rank_diff <= 1)
                    or file_diff == 0
                    or rank_diff in (0, file_diff)
                ):
                    move = chess.Move(from_sq, to_sq)
                    self.move_to_idx[move] = idx
                    self.idx_to_move[idx] = move
                    idx += 1

        promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

        for from_file in range(8):
            from_sq = 48 + from_file
            for to_file_offset in [-1, 0, 1]:
                to_file = from_file + to_file_offset
                if 0 <= to_file < 8:
                    to_sq = 56 + to_file
                    for piece in promotion_pieces:
                        move = chess.Move(from_sq, to_sq, promotion=piece)
                        self.move_to_idx[move] = idx
                        self.idx_to_move[idx] = move
                        idx += 1

        for from_file in range(8):
            from_sq = 8 + from_file
            for to_file_offset in [-1, 0, 1]:
                to_file = from_file + to_file_offset
                if 0 <= to_file < 8:
                    to_sq = to_file
                    for piece in promotion_pieces:
                        move = chess.Move(from_sq, to_sq, promotion=piece)
                        self.move_to_idx[move] = idx
                        self.idx_to_move[idx] = move
                        idx += 1

        self.num_moves = idx

    def encode_move(self, move: chess.Move) -> int:
        return self.move_to_idx.get(move, -1)

    def decode_move(self, idx: int) -> chess.Move | None:
        return self.idx_to_move.get(idx)


class ChessModel(nn.Module):
    def __init__(self, device: str = "auto"):
        super().__init__()
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.input_channels = config.model.encoding_channels
        self.conv_channels = config.model.hidden_dim
        self.num_blocks = config.model.num_layers

        self.input_conv = ConvBlock(self.input_channels, self.conv_channels)

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(self.conv_channels) for _ in range(self.num_blocks)]
        )

        self.attention_pool = AttentionPooling(
            self.conv_channels, config.model.num_heads
        )

        self.policy_conv = ConvBlock(self.conv_channels, 32)
        self.policy_fc = nn.Linear(32 * 64, config.model.move_space_size)

        self.value_conv = ConvBlock(self.conv_channels, 8)
        self.value_fc1 = nn.Linear(self.conv_channels, 128)
        self.value_fc2 = nn.Linear(128, 1)

        self.unified_cache = UnifiedCache(max_size=10000, device=self.device)

        self.to(self.device)

    def forward(self, board_input: torch.Tensor) -> ModelOutput:
        batch_size = board_input.shape[0]
        x = board_input.view(batch_size, self.input_channels, 8, 8)

        x = self.input_conv(x)
        for block in self.residual_blocks:
            x = block(x)

        policy = self.policy_conv(x)
        policy = policy.view(batch_size, -1)
        policy = self.policy_fc(policy)
        policy = f.softmax(policy, dim=-1)

        value_features = self.attention_pool(x)
        value = f.relu(self.value_fc1(value_features))
        value = torch.tanh(self.value_fc2(value))

        return ModelOutput(policy=policy, value=value)

    def encode_board(self, board: chess.Board | list[chess.Board]) -> torch.Tensor:
        boards = [board] if isinstance(board, chess.Board) else board
        batch_size = len(boards)

        cached_tensors = []
        uncached_boards = []
        uncached_indices = []

        for i, b in enumerate(boards):
            cached_tensor = self.unified_cache.get_board_encoding(b)
            if cached_tensor is not None:
                cached_tensors.append((i, cached_tensor))
            else:
                uncached_boards.append(b)
                uncached_indices.append(i)

        tensor_shape = (batch_size, self.input_channels * 64)
        batch_tensor = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )

        for idx, cached_tensor in cached_tensors:
            batch_tensor[idx] = cached_tensor

        if uncached_boards:
            uncached_tensors = self._encode_boards_vectorized(uncached_boards)

            for i, board_idx in enumerate(uncached_indices):
                tensor = uncached_tensors[i]
                batch_tensor[board_idx] = tensor
                self.unified_cache.put_board_encoding(uncached_boards[i], tensor)

        return (
            batch_tensor.squeeze(0) if isinstance(board, chess.Board) else batch_tensor
        )

    def _encode_boards_vectorized(self, boards: list[chess.Board]) -> torch.Tensor:
        batch_size = len(boards)
        tensor_shape = (batch_size, self.input_channels * 64)
        batch_tensor = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )

        square_indices = torch.arange(64, device=self.device)
        turn_indices = square_indices * self.input_channels + 6
        meta_indices = square_indices * self.input_channels + 7

        all_batch_indices = []
        all_flat_indices = []
        all_values = []

        for batch_idx, b in enumerate(boards):
            piece_map = b.piece_map()
            if piece_map:
                squares_list = list(piece_map.keys())
                pieces_list = list(piece_map.values())

                channels_list = [p.piece_type - 1 for p in pieces_list]
                values_list = [
                    1.0 if p.color == chess.WHITE else -1.0 for p in pieces_list
                ]
                flat_indices_list = [
                    sq * self.input_channels + ch
                    for sq, ch in zip(squares_list, channels_list, strict=False)
                ]

                all_batch_indices.extend([batch_idx] * len(squares_list))
                all_flat_indices.extend(flat_indices_list)
                all_values.extend(values_list)

        if all_batch_indices:
            batch_indices_tensor = torch.tensor(
                all_batch_indices, device=self.device, dtype=torch.long
            )
            flat_indices_tensor = torch.tensor(
                all_flat_indices, device=self.device, dtype=torch.long
            )
            values_tensor = torch.tensor(
                all_values, device=self.device, dtype=torch.float32
            )

            batch_tensor[batch_indices_tensor, flat_indices_tensor] = values_tensor

        for batch_idx, b in enumerate(boards):
            turn_value = 1.0 if b.turn == chess.WHITE else 0.0
            batch_tensor[batch_idx, turn_indices] = turn_value

            meta_value = (
                (0.1 if b.has_kingside_castling_rights(chess.WHITE) else 0.0)
                + (0.2 if b.has_queenside_castling_rights(chess.WHITE) else 0.0)
                + (0.3 if b.has_kingside_castling_rights(chess.BLACK) else 0.0)
                + (0.4 if b.has_queenside_castling_rights(chess.BLACK) else 0.0)
                + ((b.ep_square % 8 + 1) * 0.01 if b.ep_square is not None else 0.0)
            )
            batch_tensor[batch_idx, meta_indices] = meta_value

        return batch_tensor


def uniform_probs(legal_moves: list[chess.Move]) -> dict[chess.Move, float]:
    return {} if not legal_moves else dict.fromkeys(legal_moves, 1.0 / len(legal_moves))


def zobrist_hash(board: chess.Board) -> int:
    return hash((board.board_fen(), board.turn, board.castling_rights, board.ep_square))


class UnifiedCache:
    def __init__(self, max_size: int = 50000, device: str = "cuda"):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.device = device

    def get_board_encoding(self, board: chess.Board) -> torch.Tensor | None:
        key = f"board_{zobrist_hash(board)}"
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put_board_encoding(self, board: chess.Board, tensor: torch.Tensor) -> None:
        key = f"board_{zobrist_hash(board)}"
        self.cache[key] = tensor.clone()
        self.cache.move_to_end(key)
        self._evict_if_needed()

    def get_mcts_result(self, board: chess.Board) -> tuple[torch.Tensor, float] | None:
        key = f"mcts_{zobrist_hash(board)}"
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put_mcts_result(
        self, board: chess.Board, policy: torch.Tensor, value: float
    ) -> None:
        key = f"mcts_{zobrist_hash(board)}"
        self.cache[key] = (policy, value)
        self.cache.move_to_end(key)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


class Node:
    def __init__(
        self,
        board: chess.Board,
        parent: "Node | None" = None,
        move: chess.Move | None = None,
        prior: float = config.mcts.default_prior,
        copy_board: bool = True,
    ):
        self.board: chess.Board = board.copy() if copy_board else board
        self.parent: Node | None = parent
        self.move: chess.Move | None = move
        self.prior: float = prior
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: dict[chess.Move, Node] = {}
        self.is_expanded: bool = False
        self.virtual_loss: float = 0.0

    def get_value(self) -> float:
        if self.visits > 0:
            denominator = self.visits + self.virtual_loss / config.mcts.virtual_loss
            if denominator <= 0.0:
                return 0.0
            return (self.value_sum - self.virtual_loss) / denominator
        return 0.0

    def get_ucb_score(self) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.get_value()

        exploration = (
            config.mcts.c_puct
            * self.prior
            * math.sqrt(self.parent.visits)
            / (1 + self.visits)
        )
        return self.get_value() + exploration

    def select_child(self) -> "Node | None":
        return (
            max(self.children.values(), key=lambda child: child.get_ucb_score())
            if self.children
            else None
        )

    def add_virtual_loss(self) -> None:
        self.virtual_loss += config.mcts.virtual_loss

    def remove_virtual_loss(self) -> None:
        self.virtual_loss -= config.mcts.virtual_loss

    def backup(self, value: float) -> None:
        self.visits += 1
        self.value_sum += value
        self.remove_virtual_loss()
        if self.parent:
            self.parent.backup(-value)


class MCTS:
    def __init__(
        self,
        model: ChessModel,
        move_encoder: MoveEncoder,
        device: str,
    ) -> None:
        self.model: ChessModel = model
        self.move_encoder: MoveEncoder = move_encoder
        self.device: str = device

        self.unified_cache = self.model.unified_cache

    def search_batch(self, boards: list[chess.Board]) -> list[dict[chess.Move, float]]:
        if not boards:
            return []

        try:
            roots = [Node(board) for board in boards]

            for _ in range(config.mcts.simulations):
                leaves_to_eval = []
                paths = []

                for root in roots:
                    if not root.board.is_game_over():
                        path = []
                        leaf = self._select_with_virtual_loss(root, path)
                        if leaf:
                            leaves_to_eval.append(leaf)
                            paths.append(path)

                if leaves_to_eval:
                    self._eval_nodes_batch(leaves_to_eval)

                    for path in paths:
                        for node in path:
                            node.remove_virtual_loss()

            results = []
            for root in roots:
                visit_counts = {
                    move: child.visits for move, child in root.children.items()
                }
                total_visits = sum(visit_counts.values())

                if total_visits > 0:
                    move_probs = {
                        move: count / total_visits
                        for move, count in visit_counts.items()
                    }
                else:
                    legal_moves = list(root.board.legal_moves)
                    move_probs = uniform_probs(legal_moves)

                results.append(move_probs)

            return results

        except RuntimeError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [uniform_probs(list(board.legal_moves)) for board in boards]

    def _select_with_virtual_loss(self, node: Node, path: list[Node]) -> Node | None:
        current = node

        while current.is_expanded and not current.board.is_game_over():
            current.add_virtual_loss()
            path.append(current)

            child = current.select_child()
            if child is None:
                break
            current = child

        if not current.board.is_game_over():
            current.add_virtual_loss()
            path.append(current)
            return current
        return None

    def _eval_nodes_batch(self, nodes: list[Node]) -> None:
        if not nodes:
            return

        terminal_nodes = []
        active_nodes = []
        cached_results = []

        for node in nodes:
            if node.board.is_game_over():
                terminal_nodes.append(node)
            else:
                cached = self.unified_cache.get_mcts_result(node.board)
                if cached is not None:
                    cached_results.append((node, cached))
                else:
                    active_nodes.append(node)

        for node in terminal_nodes:
            value = self._terminal_value(node)
            node.backup(value)

        for node, (policy, value) in cached_results:
            self._expand_with_policy(node, policy)
            node.backup(value)

        if active_nodes:
            for i in range(0, len(active_nodes), config.mcts.batch_size):
                batch_nodes = active_nodes[i : i + config.mcts.batch_size]
                batch_boards = [node.board for node in batch_nodes]

                try:
                    board_tensors = self.model.encode_board(batch_boards)

                    with torch.no_grad():
                        outputs = self.model(board_tensors)
                        policies = outputs.policy
                        values = outputs.value.squeeze(-1)

                    values_cpu = values.cpu().numpy()

                    for j, node in enumerate(batch_nodes):
                        policy = policies[j]
                        value = float(values_cpu[j])

                        self.unified_cache.put_mcts_result(node.board, policy, value)

                        self._expand_with_policy(node, policy)
                        node.backup(value)

                except RuntimeError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for node in batch_nodes:
                        self._expand_uniform(node)
                        node.backup(0.0)

    def _expand_with_policy(self, node: Node, policy: torch.Tensor) -> None:
        if node.is_expanded:
            return

        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            node.is_expanded = True
            return

        priors = []
        for move in legal_moves:
            move_idx = self.move_encoder.encode_move(move)
            if 0 <= move_idx < len(policy):
                priors.append(float(policy[move_idx].item()))
            else:
                priors.append(config.mcts.default_prior)

        prior_sum = sum(priors)
        priors = (
            [p / prior_sum for p in priors]
            if prior_sum > 0
            else [1.0 / len(legal_moves)] * len(legal_moves)
        )

        if node.parent is None:
            rng = np.random.default_rng()
            noise = rng.dirichlet([config.mcts.dirichlet_alpha] * len(priors))
            priors = [
                (1 - config.mcts.dirichlet_epsilon) * p
                + config.mcts.dirichlet_epsilon * n
                for p, n in zip(priors, noise, strict=False)
            ]

        for move, prior in zip(legal_moves, priors, strict=False):
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = Node(
                child_board, parent=node, move=move, prior=prior, copy_board=False
            )

        node.is_expanded = True

    def _expand_uniform(self, node: Node) -> None:
        if node.is_expanded:
            return

        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            node.is_expanded = True
            return

        prior = 1.0 / len(legal_moves)
        for move in legal_moves:
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = Node(
                child_board, parent=node, move=move, prior=prior, copy_board=False
            )

        node.is_expanded = True

    def _terminal_value(self, node: Node) -> float:
        return {"1-0": 1.0, "0-1": -1.0}.get(node.board.result(), 0.0)
