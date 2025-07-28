import os

import chess
import torch
from config import (
    GAMES_PER_ITER,
    MAX_MOVES,
    TEMP_HIGH,
    TEMP_LOW,
    TEMP_MOVES,
    WORKER_COUNT,
)
from mcts import MCTS
from model import ChessModel
from move_encoder import MoveEncoder
from utils import sample_move_from_probabilities, should_resign_position

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

    except Exception:
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
            games_for_this_worker = base_games_per_worker + (
                1 if worker_id < extra_games else 0
            )
            worker_games = list(range(game_id, game_id + games_for_this_worker))
            assignments.append(worker_games)
            game_id += games_for_this_worker

    return assignments


def play_single_game_worker(game_id: int) -> tuple[list[tuple], dict]:
    if worker_model is None or worker_mcts is None:
        return [], {
            "game_id": game_id,
            "move_count": 0,
            "resigned": False,
            "result": "*",
            "is_game_over": False,
            "error": True,
        }

    board = chess.Board()
    game_data = []
    move_count = 0
    resigned = False

    while not board.is_game_over() and move_count < MAX_MOVES:
        if should_resign_position(board, worker_model):
            resigned = True
            break

        policies = worker_mcts.search_batch([board])
        policy = policies[0]

        if not policy:
            break

        board_tensor = worker_model.encode_board(board)
        game_data.append((board_tensor, policy, board.turn))

        temperature = TEMP_HIGH if move_count < TEMP_MOVES else TEMP_LOW
        move = sample_move_from_probabilities(policy, temperature)

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
        "is_game_over": board.is_game_over(),
    }

    return game_data, result_info


def execute_worker_games(game_assignments: list[int]) -> dict:
    if worker_model is None or worker_mcts is None:
        return {
            "game_data": [],
            "game_results": [
                {
                    "game_id": gid,
                    "move_count": 0,
                    "resigned": False,
                    "result": "*",
                    "is_game_over": False,
                    "error": True,
                }
                for gid in game_assignments
            ],
            "mcts_stats": {},
            "model_stats": {},
            "worker_id": os.getpid(),
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
        except Exception:
            error_result = {
                "game_id": game_id,
                "move_count": 0,
                "resigned": False,
                "result": "*",
                "is_game_over": False,
                "error": True,
            }
            worker_game_results.append(error_result)

    mcts_stats = worker_mcts.get_search_stats()
    model_stats = worker_model.get_inference_stats()

    return {
        "game_data": all_game_data,
        "game_results": worker_game_results,
        "mcts_stats": mcts_stats,
        "model_stats": model_stats,
        "worker_id": os.getpid(),
    }
