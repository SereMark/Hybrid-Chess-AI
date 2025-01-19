import time
from typing import List, Tuple
import numpy as np
import chess
import chess.pgn
import torch
from src.models.model import ChessModel
from src.utils.chess_utils import convert_board_to_tensor, get_move_mapping, get_total_moves
from src.utils.common_utils import initialize_random_seeds, wait_if_paused, get_game_result
from src.training.reinforcement.mcts import MCTS

class PlayAndCollectWorker:
    @classmethod
    def run_process(cls, args: Tuple) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], List[int], List[chess.pgn.Game]]:
        worker = cls(args)
        return worker.run()

    def __init__(self, args: Tuple):
        (self.model_state_dict, self.device_type, self.simulations, self.c_puct, self.temperature, self.games_per_process, self.stop_event, self.pause_event, self.seed, self.stats_queue) = args

    def run(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], List[int], List[chess.pgn.Game]]:
        initialize_random_seeds(self.seed)
        device = torch.device(self.device_type)
        total_moves = get_total_moves()

        # Load model
        model = ChessModel(total_moves).to(device)
        try:
            model.load_state_dict(self.model_state_dict)
        except Exception as e:
            self.stats_queue.put({"error": f"Failed to load state_dict in worker: {str(e)}"})
            return ([], [], [], [], [], [])

        model.eval()

        inputs_list, policy_targets_list, value_targets_list, results_list, game_lengths_list, pgn_games_list = [], [], [], [], [], []

        try:
            for _ in range(self.games_per_process):
                if self.stop_event.is_set():
                    break

                wait_if_paused(self.pause_event)
                board = chess.Board()

                # Initialize MCTS
                mcts = MCTS(model=model, device=device, c_puct=self.c_puct, n_simulations=self.simulations)
                mcts.set_root_node(board)

                states: List[np.ndarray] = []
                mcts_probs: List[np.ndarray] = []
                current_players: List[bool] = []

                move_count = 0
                max_moves = 200

                game = chess.pgn.Game()
                game.headers["Event"] = "Reinforcement Self-Play"
                game.headers["Site"] = "Self-Play"
                game.headers["Date"] = time.strftime("%Y.%m.%d")
                game.headers["Round"] = "-"
                game.headers["White"] = "Agent"
                game.headers["Black"] = "Opponent"
                game.headers["Result"] = "*"
                node = game

                # Self-play
                while not board.is_game_over() and move_count < max_moves:
                    action_probs = mcts.get_move_probs(self.temperature)

                    # Dirichlet noise on the first move
                    if move_count == 0:
                        moves = list(action_probs.keys())
                        noise = np.random.dirichlet([0.3] * len(moves))
                        for i, mv in enumerate(moves):
                            action_probs[mv] = (action_probs[mv] * 0.75 + noise[i] * 0.25)

                    if not action_probs:
                        break

                    moves_list = list(action_probs.keys())
                    probs_array = np.array(list(action_probs.values()), dtype=np.float32)
                    probs_array /= probs_array.sum()
                    chosen_move = np.random.choice(moves_list, p=probs_array)

                    # Record game state
                    states.append(convert_board_to_tensor(board))

                    # Convert probabilities to a dense vector
                    prob_arr = np.zeros(total_moves, dtype=np.float32)
                    for mv, prob in action_probs.items():
                        idx = get_move_mapping().get_index_by_move(mv)
                        if idx is not None and 0 <= idx < total_moves:
                            prob_arr[idx] = prob
                    mcts_probs.append(prob_arr)

                    current_players.append(board.turn)

                    # Push move
                    try:
                        board.push(chosen_move)
                    except ValueError:
                        break
                    node = node.add_variation(chosen_move)
                    mcts.update_with_move(chosen_move)
                    move_count += 1

                # Game concluded
                result = get_game_result(board)
                if board.is_checkmate():
                    # Last move made by the winner
                    last_player = not board.turn
                    winners = [
                        result if (pl == last_player) else -result
                        for pl in current_players
                    ]
                else:
                    winners = [0.0 for _ in current_players]

                game_length = len(states)
                if result > 0:
                    game.headers["Result"] = "1-0"
                elif result < 0:
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1/2-1/2"

                pgn_games_list.append(game)
                inputs_list.extend(states)
                policy_targets_list.extend(mcts_probs)
                value_targets_list.extend(winners)
                results_list.append(result)
                game_lengths_list.append(game_length)

            # Collect stats
            total_games = len(results_list)
            wins = results_list.count(1.0)
            losses = results_list.count(-1.0)
            draws = results_list.count(0.0)
            if game_lengths_list:
                avg_length = sum(game_lengths_list) / len(game_lengths_list)
            else:
                avg_length = 0.0

            self.stats_queue.put({
                "total_games": total_games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "avg_game_length": avg_length,
            })

        except Exception as e:
            self.stats_queue.put({"error": f"Exception in PlayAndCollectWorker: {str(e)}"})

        return (inputs_list, policy_targets_list, value_targets_list, results_list, game_lengths_list, pgn_games_list)