CONFIG = {
    "mcts_simulations": 25,
    "c_puct": 1.0,
    "move_prior": 0.001,
    "hidden_dim": 256,
    "num_layers": 4,
    "num_heads": 8,
    "move_space_size": 4096,
    "board_size": 8,
    "piece_types": 6,
    "colors": 2,
    "dropout": 0.1,
    "ffn_multiplier": 4,
    "learning_rate": 0.001,
    "batch_size": 128,
    "games_per_iteration": 3,
    "max_moves_per_game": 50,
    "iterations": 100,
    "high_temperature": 1.0,
    "low_temperature": 0.1,
    "exploration_temperature_moves": 10,
    "total_squares": 64,
    "piece_to_index": {
        "pawn": 0,
        "knight": 1,
        "bishop": 2,
        "rook": 3,
        "queen": 4,
        "king": 5,
    },
    "promotion_pieces": [4, 3, 2, 1],
    "pawn_promotion_ranks": {"white": [48, 56, 0, 8], "black": [8, 16, 56, 64]},
    "parallel_games": 8,
    "mcts_batch_size": 16,
    "chess_white_win": "1-0",
    "chess_black_win": "0-1",
    "win_value": 1.0,
    "loss_value": -1.0,
    "draw_value": 0.0,
    "timeout_game_values": [-0.1, 0.0, 0.1],
}


def get_config(section: str, key: str):
    if section == "model":
        return CONFIG.get(key)
    elif section == "mcts":
        return CONFIG.get(f"mcts_{key}")
    elif section == "training":
        return CONFIG.get(f"training_{key}", CONFIG.get(key))
    elif section == "batch":
        return CONFIG.get(key)
    else:
        return CONFIG.get(key)
