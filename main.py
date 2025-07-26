import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

import chess
import numpy as np
import psutil
import torch
import torch.nn.functional
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

ITERATIONS = 150
BUFFER_SIZE = 25000
SAVE_EVERY = 25
HIDDEN_DIM = 256
NUM_LAYERS = 10
GAMES_PER_ITER = 8
MAX_MOVES = 45
TEMP_MOVES = 20
RESIGN_THRESHOLD = -0.85
SIMULATIONS = 60
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
LEARNING_RATE = 0.0005
BATCH_SIZE = 512
GRADIENT_ACCUMULATION = 2

MOVE_COUNT = 1858
BOARD_SIZE = 512
CACHE_SIZE = 10000


def get_system_stats():
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0
        return {
            "cpu_percent": cpu_percent,
            "ram_used_gb": memory.used / (1024**3),
            "ram_total_gb": memory.total / (1024**3),
            "load_avg": load_avg,
        }
    except Exception:
        return {"cpu_percent": 0, "ram_used_gb": 0, "ram_total_gb": 0, "load_avg": 0}


class ModelOutput(NamedTuple):
    policy: torch.Tensor
    value: torch.Tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.nn.functional.relu(x + residual)


class MoveEncoder:
    def __init__(self):
        self.move_to_idx = {}
        idx = 0

        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq != to_sq and self._is_pseudo_legal(from_sq, to_sq):
                    self.move_to_idx[chess.Move(from_sq, to_sq)] = idx
                    idx += 1

        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        for from_sq in range(48, 56):
            for offset in [7, 8, 9]:
                to_sq = from_sq + offset
                if 56 <= to_sq < 64 and 0 <= (from_sq % 8 + offset - 8) < 8:
                    for piece in pieces:
                        self.move_to_idx[chess.Move(from_sq, to_sq, piece)] = idx
                        idx += 1

        for from_sq in range(8, 16):
            for offset in [7, 8, 9]:
                to_sq = from_sq - offset
                if 0 <= to_sq < 8 and 0 <= (from_sq % 8 - offset + 8) < 8:
                    for piece in pieces:
                        self.move_to_idx[chess.Move(from_sq, to_sq, piece)] = idx
                        idx += 1

    def _is_pseudo_legal(self, from_sq: int, to_sq: int) -> bool:
        from_file, from_rank = from_sq % 8, from_sq // 8
        to_file, to_rank = to_sq % 8, to_sq // 8
        file_diff = abs(to_file - from_file)
        rank_diff = abs(to_rank - from_rank)

        if rank_diff in {0, file_diff} or file_diff == 0:
            return True
        if (file_diff == 2 and rank_diff == 1) or (file_diff == 1 and rank_diff == 2):
            return True
        return file_diff <= 1 and rank_diff <= 1

    def encode_move(self, move: chess.Move) -> int:
        return self.move_to_idx.get(move, -1)


class ChessModel(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device

        self.input_conv = nn.Conv2d(8, HIDDEN_DIM, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(HIDDEN_DIM)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(HIDDEN_DIM) for _ in range(NUM_LAYERS)]
        )

        self.policy_conv = nn.Conv2d(HIDDEN_DIM, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 64, MOVE_COUNT)

        self.value_conv = nn.Conv2d(HIDDEN_DIM, 8, 1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self.cache = OrderedDict()

    def forward(self, board_input: torch.Tensor) -> ModelOutput:
        x = board_input.view(-1, 8, 8, 8)

        x = torch.nn.functional.relu(self.bn_input(self.input_conv(x)))
        for block in self.residual_blocks:
            x = block(x)

        policy = torch.nn.functional.relu(self.policy_bn(self.policy_conv(x)))
        policy = self.policy_fc(policy.view(-1, 32 * 64))
        policy = torch.nn.functional.softmax(policy, dim=-1)

        value = torch.nn.functional.relu(self.value_bn(self.value_conv(x)))
        value = self.value_fc1(value.view(-1, 8 * 64))
        value = torch.tanh(self.value_fc2(value))

        return ModelOutput(policy=policy, value=value)

    def encode_board(self, board: chess.Board | list[chess.Board]) -> torch.Tensor:
        boards = [board] if isinstance(board, chess.Board) else board
        batch_size = len(boards)
        tensor = torch.zeros(
            (batch_size, BOARD_SIZE), dtype=torch.float32, device=self.device
        )

        for i, b in enumerate(boards):
            board_key = (b.board_fen(), b.turn, b.castling_rights, b.ep_square)
            if board_key in self.cache:
                self.cache.move_to_end(board_key)
                tensor[i] = self.cache[board_key]
                continue

            board_tensor = torch.zeros(
                BOARD_SIZE, dtype=torch.float32, device=self.device
            )

            for sq, piece in b.piece_map().items():
                board_tensor[sq * 8 + piece.piece_type - 1] = (
                    1.0 if piece.color else -1.0
                )

            board_tensor[6::8] = 1.0 if b.turn else 0.0

            board_tensor[7::8] = (
                0.1 * b.has_kingside_castling_rights(chess.WHITE)
                + 0.2 * b.has_queenside_castling_rights(chess.WHITE)
                + 0.3 * b.has_kingside_castling_rights(chess.BLACK)
                + 0.4 * b.has_queenside_castling_rights(chess.BLACK)
                + 0.01 * (b.ep_square % 8 + 1 if b.ep_square else 0)
            )

            tensor[i] = board_tensor
            self.cache[board_key] = board_tensor
            if len(self.cache) > CACHE_SIZE:
                self.cache.popitem(last=False)

        return tensor.squeeze(0) if isinstance(board, chess.Board) else tensor


class Node:
    def __init__(self, board: chess.Board, parent=None, move=None, prior=0.001):
        self.board = board.copy() if parent is None else board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False

    def ucb_score(self) -> float:
        if self.visits == 0:
            return float("inf")
        q = self.value_sum / self.visits
        if self.parent:
            u = C_PUCT * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
            return q + u
        return q

    def select_child(self):
        return (
            max(self.children.values(), key=lambda c: c.ucb_score())
            if self.children
            else None
        )

    def backup(self, value: float) -> None:
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)


class MCTS:
    def __init__(self, model: ChessModel, move_encoder: MoveEncoder):
        self.model = model
        self.move_encoder = move_encoder

    def search_batch(self, boards: list[chess.Board]) -> list[dict[chess.Move, float]]:
        roots = [Node(board) for board in boards]

        for _ in range(SIMULATIONS):
            leaves = []

            for root in roots:
                node = root
                while node.is_expanded and not node.board.is_game_over():
                    node = node.select_child()
                    if node is None:
                        break
                if node and not node.board.is_game_over():
                    leaves.append(node)

            if not leaves:
                continue

            boards_to_eval = []
            terminal_nodes = []

            for node in leaves:
                if node.board.is_game_over():
                    result = {"1-0": 1.0, "0-1": -1.0}.get(node.board.result(), 0.0)
                    terminal_nodes.append((node, result))
                else:
                    boards_to_eval.append(node)

            if boards_to_eval:
                board_tensors = self.model.encode_board(
                    [n.board for n in boards_to_eval]
                )
                with torch.no_grad():
                    outputs = self.model(board_tensors)
                    policies = outputs.policy
                    values = outputs.value.squeeze(-1)

                for i, node in enumerate(boards_to_eval):
                    if not node.is_expanded:
                        legal_moves = list(node.board.legal_moves)
                        if legal_moves:
                            priors = []
                            for move in legal_moves:
                                idx = self.move_encoder.encode_move(move)
                                prior = (
                                    policies[i][idx].item()
                                    if 0 <= idx < MOVE_COUNT
                                    else 0.001
                                )
                                priors.append(max(prior, 0.001))

                            prior_sum = sum(priors)
                            priors = [p / prior_sum for p in priors]

                            if node.parent is None:
                                noise = np.random.dirichlet(
                                    [DIRICHLET_ALPHA] * len(priors)
                                )
                                priors = [
                                    (1 - DIRICHLET_EPSILON) * p + DIRICHLET_EPSILON * n
                                    for p, n in zip(priors, noise, strict=False)
                                ]

                            for move, prior in zip(legal_moves, priors, strict=False):
                                child_board = node.board.copy()
                                child_board.push(move)
                                node.children[move] = Node(
                                    child_board, node, move, prior
                                )

                            node.is_expanded = True

                    node.backup(values[i].item())

            for node, value in terminal_nodes:
                node.backup(value)

        results = []
        for root in roots:
            visits = {move: child.visits for move, child in root.children.items()}
            total = sum(visits.values())

            if total > 0:
                probs = {move: count / total for move, count in visits.items()}
            else:
                moves = list(root.board.legal_moves)
                probs = {move: 1.0 / len(moves) for move in moves} if moves else {}

            results.append(probs)

        return results


class AlphaZeroTrainer:
    def __init__(self, device: str):
        self.device = device
        self.move_encoder = MoveEncoder()
        self.model = ChessModel(device).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=ITERATIONS, eta_min=LEARNING_RATE * 0.1
        )

        self.buffer_size = 0
        self.buffer_pos = 0
        self.buffer_boards = torch.zeros(
            (BUFFER_SIZE, BOARD_SIZE), dtype=torch.float32, device=device
        )
        self.buffer_values = torch.zeros(
            BUFFER_SIZE, dtype=torch.float32, device=device
        )
        self.buffer_policies = torch.zeros(
            (BUFFER_SIZE, MOVE_COUNT), dtype=torch.float32, device=device
        )

        self.mcts = MCTS(self.model, self.move_encoder)
        self.games_played = 0
        self.accumulation_step = 0
        self.training_start_time = time.time()
        self.iteration_times = []

    def self_play(self) -> tuple[list[tuple], dict[str, float]]:
        games_data = [[] for _ in range(GAMES_PER_ITER)]
        boards = [chess.Board() for _ in range(GAMES_PER_ITER)]
        move_counts = [0] * GAMES_PER_ITER
        active = list(range(GAMES_PER_ITER))

        while active:
            remaining = []
            for i in active:
                if (
                    boards[i].is_game_over()
                    or move_counts[i] >= MAX_MOVES
                    or self._should_resign(boards[i])
                ):
                    continue
                remaining.append(i)
            active = remaining

            if not active:
                break

            active_boards = [boards[i] for i in active]
            policies = self.mcts.search_batch(active_boards)

            for idx, game_idx in enumerate(active):
                board = boards[game_idx]
                policy = policies[idx]

                if not policy:
                    continue

                board_tensor = self.model.encode_board(board)
                games_data[game_idx].append(
                    (board_tensor.clone(), policy.copy(), board.turn)
                )

                temp = 1.0 if move_counts[game_idx] < TEMP_MOVES else 0.1
                move = self._sample_move(policy, temp)

                if move in board.legal_moves:
                    board.push(move)
                    move_counts[game_idx] += 1

        self.games_played += GAMES_PER_ITER

        all_data = []
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        completed_games = sum(1 for board in boards if board.is_game_over())
        total_moves = sum(len(game_data) for game_data in games_data)

        for game_idx, board in enumerate(boards):
            result_str = board.result()
            if board.is_game_over():
                results[result_str] += 1

            result_value = {"1-0": 1.0, "0-1": -1.0}.get(result_str, 0.0)
            for tensor, probs, turn in games_data[game_idx]:
                value = result_value if turn == chess.WHITE else -result_value
                all_data.append((tensor, probs, value))

        game_stats = {
            "completed": completed_games,
            "total": GAMES_PER_ITER,
            "wins": results["1-0"],
            "losses": results["0-1"],
            "draws": results["1/2-1/2"],
            "avg_moves": total_moves / GAMES_PER_ITER,
        }

        return all_data, game_stats

    def _should_resign(self, board: chess.Board) -> bool:
        if board.is_game_over():
            return False

        with torch.no_grad():
            tensor = self.model.encode_board(board).unsqueeze(0)
            value = self.model(tensor).value.item()
            return (
                (value < RESIGN_THRESHOLD)
                if board.turn
                else (value > -RESIGN_THRESHOLD)
            )

    def _sample_move(
        self, probs: dict[chess.Move, float], temperature: float
    ) -> chess.Move:
        moves = list(probs.keys())
        if not moves:
            raise ValueError("No moves available for sampling")

        values = np.array(list(probs.values()), dtype=np.float32)

        if temperature != 1.0:
            values = np.power(values, 1.0 / temperature)

        values_sum = values.sum()
        if values_sum == 0:
            values = np.ones_like(values)
        else:
            values = values / values_sum

        idx = np.random.choice(len(moves), p=values)
        return moves[idx]

    def add_to_buffer(self, data: list[tuple]) -> None:
        for board_tensor, move_probs, value in data:
            if not move_probs:
                continue

            self.buffer_boards[self.buffer_pos] = board_tensor
            self.buffer_values[self.buffer_pos] = value

            policy_tensor = self.buffer_policies[self.buffer_pos]
            policy_tensor.zero_()
            for move, prob in move_probs.items():
                idx = self.move_encoder.encode_move(move)
                if 0 <= idx < MOVE_COUNT:
                    policy_tensor[idx] = prob

            self.buffer_pos = (self.buffer_pos + 1) % BUFFER_SIZE
            self.buffer_size = min(self.buffer_size + 1, BUFFER_SIZE)

    def train_step(self) -> dict:
        if self.buffer_size == 0:
            return {}

        indices = torch.randint(
            0,
            self.buffer_size,
            (min(BATCH_SIZE, self.buffer_size),),
            device=self.device,
        )
        boards = self.buffer_boards[indices]
        values = self.buffer_values[indices]
        policies = self.buffer_policies[indices]

        if self.accumulation_step == 0:
            self.optimizer.zero_grad()

        self.model.train()
        outputs = self.model(boards)

        value_loss = torch.nn.functional.mse_loss(outputs.value.squeeze(), values)
        policy_loss = torch.nn.functional.kl_div(
            torch.log(outputs.policy + 1e-8), policies, reduction="batchmean"
        )

        total_loss = (value_loss + policy_loss) / GRADIENT_ACCUMULATION
        total_loss.backward()

        self.accumulation_step += 1
        if self.accumulation_step >= GRADIENT_ACCUMULATION:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            self.accumulation_step = 0

        return {
            "loss": (value_loss + policy_loss).item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def iteration(self) -> dict:
        start = time.time()

        game_data, game_stats = self.self_play()
        self.add_to_buffer(game_data)

        losses = self.train_step()

        if self.accumulation_step == 0:
            self.scheduler.step()

        iteration_time = time.time() - start
        self.iteration_times.append(iteration_time)

        result = {
            "time": iteration_time,
            "buffer": self.buffer_size,
        }
        result.update(losses)
        result.update(game_stats)
        return result

    def save(self, path: Path) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "games_played": self.games_played,
            },
            path,
        )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path("./checkpoints")
    save_dir.mkdir(exist_ok=True)

    trainer = AlphaZeroTrainer(device)
    print(f"Training on {device}")
    print(f"Model: {sum(p.numel() for p in trainer.model.parameters()):,} parameters")

    cuda_available = torch.cuda.is_available()
    gb_bytes = 1024**3

    for i in range(1, ITERATIONS + 1):
        metrics = trainer.iteration()

        if metrics:
            sys_stats = get_system_stats()
            gpu_mem = torch.cuda.memory_allocated() / gb_bytes if cuda_available else 0

            elapsed = time.time() - trainer.training_start_time
            avg_time = sum(trainer.iteration_times) / len(trainer.iteration_times)
            eta_min = (ITERATIONS - i) * avg_time / 60
            games_per_sec = trainer.games_played / elapsed if elapsed > 0 else 0
            completion_rate = metrics["completed"] / metrics["total"] * 100

            loss = metrics.get("loss", 0)
            value_loss = metrics.get("value_loss", 0)
            policy_loss = metrics.get("policy_loss", 0)
            lr = metrics.get("learning_rate", 0)

            print(
                f"[{i:3d}/{ITERATIONS}] Loss: {loss:.4f} (V:{value_loss:.3f} P:{policy_loss:.3f}) LR: {lr:.6f}"
            )
            print(
                f"  Games: {metrics['completed']}/{metrics['total']} ({completion_rate:.0f}%) "
                f"W:{metrics['wins']} L:{metrics['losses']} D:{metrics['draws']} Moves: {metrics['avg_moves']:.1f}"
            )
            print(
                f"  System: CPU {sys_stats['cpu_percent']:.1f}% RAM {sys_stats['ram_used_gb']:.1f}/{sys_stats['ram_total_gb']:.1f}GB "
                f"GPU {gpu_mem:.1f}GB Load {sys_stats['load_avg']:.2f}"
            )
            print(
                f"  Speed: {metrics['time']:.1f}s/iter {games_per_sec:.1f} games/s "
                f"Buffer: {metrics['buffer']}/{BUFFER_SIZE} ETA: {eta_min:.0f}m\n"
            )

        if i % SAVE_EVERY == 0:
            trainer.save(save_dir / f"model_{i}.pt")

    trainer.save(save_dir / "model_final.pt")
    print(f"\nTraining complete! Games: {trainer.games_played:,}")


if __name__ == "__main__":
    main()
