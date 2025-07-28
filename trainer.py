import contextlib
import multiprocessing
import os
import time
from pathlib import Path

import chess
import torch
from config import (
    BATCH_SIZE,
    BOARD_SIZE,
    BUFFER_SIZE,
    GAMES_PER_ITER,
    GRADIENT_ACCUMULATION,
    ITERATIONS,
    LEARNING_RATE,
    MAX_MOVES,
    MOVE_COUNT,
    TEMP_MOVES,
    USE_MIXED_PRECISION,
)
from mcts import MCTS
from model import ChessModel
from move_encoder import MoveEncoder
from torch import optim
from torch.nn import functional
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import sample_move_from_probabilities, should_resign_position

WORKER_COUNT = 12
worker_model = None
worker_mcts = None
worker_move_encoder = None


def init_worker_process(model_state_dict: dict, device_str: str) -> bool:
    global worker_model, worker_mcts, worker_move_encoder

    try:
        device = torch.device(device_str)

        worker_model = ChessModel(device_str).to(device)
        worker_model.load_state_dict(model_state_dict)
        worker_model.eval()

        worker_move_encoder = MoveEncoder()
        worker_mcts = MCTS(worker_model, worker_move_encoder)

        return True

    except Exception as e:
        print(f"Worker initialization failed: {e}")
        return False


def distribute_games_to_workers() -> list[list[int]]:
    assignments = []
    game_id = 0

    if GAMES_PER_ITER <= WORKER_COUNT:
        for game_id in range(GAMES_PER_ITER):
            assignments.append([game_id])
    else:
        base_games_per_worker = GAMES_PER_ITER // WORKER_COUNT
        extra_games = GAMES_PER_ITER % WORKER_COUNT

        for worker_id in range(WORKER_COUNT):
            games_for_this_worker = base_games_per_worker + (1 if worker_id < extra_games else 0)
            worker_games = list(range(game_id, game_id + games_for_this_worker))
            assignments.append(worker_games)
            game_id += games_for_this_worker

    return assignments


def should_resign_worker(board: chess.Board) -> bool:
    return should_resign_position(board, worker_model)


def sample_move_worker(probs: dict[chess.Move, float], temperature: float) -> chess.Move | None:
    return sample_move_from_probabilities(probs, temperature)


def play_single_game_worker(game_id: int) -> tuple[list[tuple], dict]:
    if worker_model is None or worker_mcts is None:
        return [], {
            "game_id": game_id,
            "move_count": 0,
            "resigned": False,
            "result": "*",
            "is_game_over": False,
            "error": True
        }

    board = chess.Board()
    game_data = []
    move_count = 0
    resigned = False

    while not board.is_game_over() and move_count < MAX_MOVES:
        if should_resign_worker(board):
            resigned = True
            break

        policies = worker_mcts.search_batch([board])
        policy = policies[0]

        if not policy:
            break

        board_tensor = worker_model.encode_board(board)
        game_data.append((board_tensor, policy, board.turn))

        temperature = 1.0 if move_count < TEMP_MOVES else 0.1
        move = sample_move_worker(policy, temperature)

        if move is not None and move in board.legal_moves:
            board.push(move)
            move_count += 1
        else:
            break

    result_info = {
        "game_id": game_id,
        "move_count": move_count,
        "resigned": resigned,
        "result": board.result(),
        "is_game_over": board.is_game_over()
    }

    return game_data, result_info


def execute_worker_games(game_assignments: list[int]) -> dict:
    if worker_model is None or worker_mcts is None:
        return {
            "game_data": [],
            "game_results": [{
                "game_id": gid,
                "move_count": 0,
                "resigned": False,
                "result": "*",
                "is_game_over": False,
                "error": True
            } for gid in game_assignments],
            "mcts_stats": {},
            "model_stats": {},
            "worker_id": os.getpid()
        }

    all_game_data = []
    worker_game_results = []

    worker_mcts.reset_search_stats()
    worker_model.reset_inference_stats()

    for game_id in game_assignments:
        try:
            game_data, game_result = play_single_game_worker(game_id)
            all_game_data.extend(game_data)
            worker_game_results.append(game_result)
        except Exception as e:
            print(f"Game {game_id} failed in worker: {e}")
            error_result = {
                "game_id": game_id,
                "move_count": 0,
                "resigned": False,
                "result": "*",
                "is_game_over": False,
                "error": True
            }
            worker_game_results.append(error_result)

    mcts_stats = worker_mcts.get_search_stats()
    model_stats = worker_model.get_inference_stats()

    return {
        "game_data": all_game_data,
        "game_results": worker_game_results,
        "mcts_stats": mcts_stats,
        "model_stats": model_stats,
        "worker_id": os.getpid()
    }


class ChessTrainer:
    def __init__(self, device: str):
        self.device = device
        self.move_encoder = MoveEncoder()
        self.model = ChessModel(device).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=ITERATIONS, eta_min=LEARNING_RATE * 0.1
        )
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if USE_MIXED_PRECISION and device == "cuda"
            else None
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
        self.iteration_times: list[float] = []

    def self_play_sequential(self) -> tuple[list[tuple], dict[str, float]]:
        games_data = [[] for _ in range(GAMES_PER_ITER)]
        boards = [chess.Board() for _ in range(GAMES_PER_ITER)]
        move_counts = [0] * GAMES_PER_ITER
        resigned_games = [False] * GAMES_PER_ITER
        active = list(range(GAMES_PER_ITER))

        while active:
            remaining = []
            for i in active:
                if boards[i].is_game_over() or move_counts[i] >= MAX_MOVES:
                    continue
                elif self._should_resign(boards[i]):
                    resigned_games[i] = True
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
                games_data[game_idx].append((board_tensor, policy, board.turn))

                temp = 1.0 if move_counts[game_idx] < TEMP_MOVES else 0.1
                move = self._sample_move(policy, temp)

                if move in board.legal_moves:
                    board.push(move)
                    move_counts[game_idx] += 1

        self.games_played += GAMES_PER_ITER

        all_data = []
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        completed_games = sum(1 for board in boards if board.is_game_over())
        resigned_count = sum(resigned_games)
        move_limit_count = sum(
            1
            for i in range(GAMES_PER_ITER)
            if move_counts[i] >= MAX_MOVES
            and not boards[i].is_game_over()
            and not resigned_games[i]
        )
        game_lengths = [len(game_data) for game_data in games_data]
        total_moves = sum(game_lengths)
        temp_transitions = sum(
            1 for i in range(GAMES_PER_ITER) if move_counts[i] >= TEMP_MOVES
        )

        for game_idx, board in enumerate(boards):
            result_str = board.result()
            if board.is_game_over():
                results[result_str] += 1
                result_value = {"1-0": 1.0, "0-1": -1.0}.get(result_str, 0.0)
            elif resigned_games[game_idx]:
                last_turn = (
                    chess.WHITE
                    if len(games_data[game_idx]) == 0
                    else games_data[game_idx][-1][2]
                )
                if last_turn == chess.WHITE:
                    result_value = -1.0
                    results["0-1"] += 1
                else:
                    result_value = 1.0
                    results["1-0"] += 1
            else:
                result_value = 0.0

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
            "resigned": resigned_count,
            "move_limit": move_limit_count,
            "temp_transitions": temp_transitions,
            "min_moves": min(game_lengths) if game_lengths else 0,
            "max_moves": max(game_lengths) if game_lengths else 0,
        }

        return all_data, game_stats

    def aggregate_worker_results(self, worker_results) -> dict[str, int | float]:
        combined_stats: dict[str, int | float] = {
            "completed": 0, "total": GAMES_PER_ITER, "wins": 0, "losses": 0,
            "draws": 0, "resigned": 0, "move_limit": 0, "temp_transitions": 0
        }

        all_game_lengths = []

        for worker_result in worker_results:
            data_idx = 0

            for game_result in worker_result["game_results"]:
                if game_result.get("error", False):
                    continue

                if game_result["is_game_over"]:
                    combined_stats["completed"] += 1
                    result_str = game_result["result"]
                    if result_str == "1-0":
                        combined_stats["wins"] += 1
                    elif result_str == "0-1":
                        combined_stats["losses"] += 1
                    elif result_str == "1/2-1/2":
                        combined_stats["draws"] += 1
                elif game_result["resigned"]:
                    combined_stats["resigned"] += 1
                    combined_stats["completed"] += 1

                    expected_count = game_result["move_count"]
                    actual_count = min(expected_count, len(worker_result["game_data"]) - data_idx)

                    if actual_count == 0:
                        last_turn_white = True
                    else:
                        last_turn_white = worker_result["game_data"][data_idx + actual_count - 1][2] == chess.WHITE

                    if last_turn_white:
                        combined_stats["losses"] += 1
                    else:
                        combined_stats["wins"] += 1

                elif game_result["move_count"] >= MAX_MOVES:
                    combined_stats["move_limit"] += 1

                actual_game_length = min(game_result["move_count"], len(worker_result["game_data"]) - data_idx)
                all_game_lengths.append(actual_game_length)
                if game_result["move_count"] >= TEMP_MOVES:
                    combined_stats["temp_transitions"] += 1

                data_idx += actual_game_length

        if all_game_lengths:
            combined_stats["avg_moves"] = sum(all_game_lengths) / len(all_game_lengths)
            combined_stats["min_moves"] = min(all_game_lengths)
            combined_stats["max_moves"] = max(all_game_lengths)
        else:
            combined_stats["avg_moves"] = 0
            combined_stats["min_moves"] = 0
            combined_stats["max_moves"] = 0

        return combined_stats

    def process_worker_game_data(self, worker_results):
        all_processed_data = []

        for worker_result in worker_results:
            data_idx = 0

            for game_result in worker_result["game_results"]:
                if game_result.get("error", False):
                    continue

                expected_count = game_result["move_count"]
                actual_count = min(expected_count, len(worker_result["game_data"]) - data_idx)

                if game_result["is_game_over"]:
                    result_str = game_result["result"]
                    result_value = {"1-0": 1.0, "0-1": -1.0}.get(result_str, 0.0)
                elif game_result["resigned"]:
                    if actual_count == 0:
                        last_turn_white = True
                    else:
                        last_turn_white = worker_result["game_data"][data_idx + actual_count - 1][2] == chess.WHITE
                    result_value = -1.0 if last_turn_white else 1.0
                else:
                    result_value = 0.0

                for _ in range(actual_count):
                    if data_idx < len(worker_result["game_data"]):
                        tensor, probs, turn = worker_result["game_data"][data_idx]
                        value = result_value if turn == chess.WHITE else -result_value
                        all_processed_data.append((tensor, probs, value))
                        data_idx += 1

        return all_processed_data

    def update_trainer_stats_from_workers(self, worker_results):
        total_simulations = sum(wr["mcts_stats"]["total_simulations"] for wr in worker_results)
        total_nodes_expanded = sum(wr["mcts_stats"]["nodes_expanded"] for wr in worker_results)
        terminal_nodes_hit = sum(wr["mcts_stats"]["terminal_nodes_hit"] for wr in worker_results)
        model_forward_calls = sum(wr["mcts_stats"]["model_forward_calls"] for wr in worker_results)
        searches_performed = sum(wr["mcts_stats"]["searches_performed"] for wr in worker_results)

        self.mcts.total_simulations = total_simulations
        self.mcts.total_nodes_expanded = total_nodes_expanded
        self.mcts.terminal_nodes_hit = terminal_nodes_hit
        self.mcts.model_forward_calls = model_forward_calls
        self.mcts.searches_performed = searches_performed

        total_forward_calls = sum(wr["model_stats"]["forward_calls"] for wr in worker_results)
        total_cache_hits = sum(wr["model_stats"]["cache_hits"] for wr in worker_results)
        total_cache_misses = sum(wr["model_stats"]["cache_misses"] for wr in worker_results)

        self.model.forward_calls = total_forward_calls
        self.model.cache_hits = total_cache_hits
        self.model.cache_misses = total_cache_misses

    def self_play_parallel(self) -> tuple[list[tuple], dict[str, float]]:
        try:
            with contextlib.suppress(RuntimeError):
                torch.multiprocessing.set_start_method('spawn', force=True)

            model_state_dict = self.model.state_dict()
            device_str = self.device

            game_assignments = distribute_games_to_workers()
            optimal_processes = len(game_assignments)

            start_time = time.time()

            with torch.multiprocessing.Pool(
                processes=optimal_processes,
                initializer=init_worker_process,
                initargs=(model_state_dict, device_str)
            ) as pool:
                worker_results = pool.map(execute_worker_games, game_assignments)

            parallel_time = time.time() - start_time

            aggregated_stats = self.aggregate_worker_results(worker_results)
            processed_data = self.process_worker_game_data(worker_results)

            self.update_trainer_stats_from_workers(worker_results)

            self.games_played += GAMES_PER_ITER

            print(f"Parallel execution completed in {parallel_time:.1f}s")
            return processed_data, aggregated_stats

        except Exception as e:
            print(f"Parallel execution failed: {e}")
            print("Falling back to sequential processing...")
            return self.self_play_sequential()

    def should_use_multiprocessing(self):
        if multiprocessing.cpu_count() < 4:
            return False
        return torch.cuda.is_available()

    def self_play(self) -> tuple[list[tuple], dict[str, float]]:
        if self.should_use_multiprocessing():
            return self.self_play_parallel()
        else:
            return self.self_play_sequential()

    def _should_resign(self, board: chess.Board) -> bool:
        return should_resign_position(board, self.model)

    def _sample_move(
        self, probs: dict[chess.Move, float], temperature: float
    ) -> chess.Move:
        move = sample_move_from_probabilities(probs, temperature)
        if move is None:
            raise ValueError("No moves available for sampling")
        return move

    def add_to_buffer(self, data: list[tuple]) -> None:
        for board_tensor, move_probs, value in data:
            if not move_probs:
                continue

            self.buffer_boards[self.buffer_pos] = board_tensor
            self.buffer_values[self.buffer_pos] = value

            policy_tensor = self.buffer_policies[self.buffer_pos]
            policy_tensor.zero_()
            valid_indices = []
            valid_probs = []
            for move, prob in move_probs.items():
                idx = self.move_encoder.encode_move(move)
                if 0 <= idx < MOVE_COUNT:
                    valid_indices.append(idx)
                    valid_probs.append(prob)

            if valid_indices:
                policy_tensor[valid_indices] = torch.tensor(
                    valid_probs, device=self.device
                )

            self.buffer_pos = (self.buffer_pos + 1) % BUFFER_SIZE
            self.buffer_size = min(self.buffer_size + 1, BUFFER_SIZE)

    def train_step(self) -> dict[str, float]:
        if self.buffer_size == 0:
            return {}

        actual_batch_size = min(BATCH_SIZE, self.buffer_size)
        indices = torch.randint(
            0, self.buffer_size, (actual_batch_size,), device=self.device
        )
        boards = self.buffer_boards[indices]
        values = self.buffer_values[indices]
        policies = self.buffer_policies[indices]

        if self.accumulation_step == 0:
            self.optimizer.zero_grad()

        self.model.train()

        with torch.amp.autocast("cuda", enabled=self.scaler is not None):
            outputs = self.model(boards)
            value_loss = functional.mse_loss(outputs.value.squeeze(), values)
            policy_loss = functional.kl_div(
                torch.log(outputs.policy + 1e-8), policies, reduction="batchmean"
            )
            total_loss = (value_loss + policy_loss) / GRADIENT_ACCUMULATION

        if self.scaler:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        grad_norm = 0.0
        optimizer_stepped = False
        self.accumulation_step += 1
        if self.accumulation_step >= GRADIENT_ACCUMULATION:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                ).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                ).item()
                self.optimizer.step()
            self.accumulation_step = 0
            optimizer_stepped = True

        return {
            "loss": (value_loss + policy_loss).item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "actual_batch_size": actual_batch_size,
            "accumulation_step": self.accumulation_step,
            "optimizer_stepped": optimizer_stepped,
        }

    def iteration(self) -> dict[str, float]:
        start = time.time()

        self.mcts.reset_search_stats()
        self.model.reset_inference_stats()

        self_play_start = time.time()
        game_data, game_stats = self.self_play()
        self_play_time = time.time() - self_play_start

        buffer_start = time.time()
        self.add_to_buffer(game_data)
        buffer_time = time.time() - buffer_start

        train_start = time.time()
        losses = self.train_step()
        train_time = time.time() - train_start

        if self.accumulation_step == 0:
            self.scheduler.step()

        iteration_time = time.time() - start
        self.iteration_times.append(iteration_time)

        timing_breakdown = {
            "time": iteration_time,
            "self_play_time": self_play_time,
            "buffer_time": buffer_time,
            "train_time": train_time,
            "other_time": iteration_time - (self_play_time + buffer_time + train_time),
            "self_play_pct": (self_play_time / max(iteration_time, 0.001)) * 100,
            "train_pct": (train_time / max(iteration_time, 0.001)) * 100,
        }

        result: dict[str, float] = {
            "buffer": float(self.buffer_size),
        }
        result.update(timing_breakdown)
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
