from batch_mcts import BatchMCTS
from config import get_config
from game import Game
from move_utils import get_temperature_for_move, select_move_with_temperature


class ParallelGameManager:
    def __init__(
        self,
        model,
        move_encoder,
        device: str,
        num_parallel_games: int = 8,
        batch_size: int = 16,
        num_simulations=None,
    ):
        self.model = model
        self.move_encoder = move_encoder
        self.device = device
        self.num_parallel_games = num_parallel_games
        self.max_moves_per_game = get_config("training", "max_moves_per_game") or 50

        self.batch_mcts = BatchMCTS(
            model=model,
            move_encoder=move_encoder,
            device=device,
            batch_size=batch_size,
            num_simulations=num_simulations,
        )

    def play_games(self):
        games = [Game() for _ in range(self.num_parallel_games)]
        move_counts = [0] * self.num_parallel_games
        completed_games = []

        while len(completed_games) < self.num_parallel_games:
            active_indices = []
            active_boards = []

            for i, game in enumerate(games):
                if i not in completed_games:
                    if game.is_game_over() or move_counts[i] >= self.max_moves_per_game:
                        completed_games.append(i)
                    else:
                        active_indices.append(i)
                        active_boards.append(game.board)

            if not active_indices:
                break

            move_probs_batch = self.batch_mcts.search_batch(active_boards)

            for idx, game_idx in enumerate(active_indices):
                game = games[game_idx]
                board = active_boards[idx]
                move_probs = move_probs_batch[idx]

                if not move_probs:
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        move_probs = {
                            move: 1.0 / len(legal_moves) for move in legal_moves
                        }
                    else:
                        completed_games.append(game_idx)
                        continue

                board_tensor = self.model.encode_board_vectorized(board)
                game.add_position(board_tensor, move_probs, None)

                temperature = get_temperature_for_move(move_counts[game_idx])
                move = select_move_with_temperature(move_probs, temperature, board)

                if game.history:
                    game.history[-1]["move_played"] = move

                if game.make_move(move):
                    move_counts[game_idx] += 1
                else:
                    completed_games.append(game_idx)

        return games
