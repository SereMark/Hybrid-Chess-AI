import time, chess, torch, numpy as np
from typing import List, Tuple
from src.training.reinforcement.mcts import MCTS
from src.models.transformer import TransformerChessModel
from src.utils.train_utils import initialize_random_seeds
from src.utils.chess_utils import convert_board_to_tensor, get_move_mapping, get_total_moves

class PlayAndCollectWorker:
    @classmethod
    def run_process(cls, args: Tuple) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], List[int], List[chess.pgn.Game]]:
        model_state_dict, device_type, simulations, c_puct, temperature, games_per_process, seed, stats_queue = args
        initialize_random_seeds(seed)
        device = torch.device(device_type)
        model = TransformerChessModel(get_total_moves())
        try:
            model.load_state_dict(model_state_dict)
        except Exception as e:
            stats_queue.put({"error": f"Load state_dict error: {str(e)}"})
            return ([], [], [], [], [], [])
        model.to(device).eval()
        inputs, policy_targets, value_targets, results, game_lengths, pgn_games = [], [], [], [], [], []
        try:
            for _ in range(games_per_process):
                board, mcts = chess.Board(), MCTS(model, device, c_puct, simulations)
                mcts.set_root_node(board)
                states, mcts_probs, current_players = [], [], []
                move_count, max_moves = 0, 200
                game = chess.pgn.Game()
                game.headers.update({"Event":"Reinforcement Self-Play","Site":"Self-Play","Date":time.strftime("%Y.%m.%d"),"Round":"-","White":"Agent","Black":"Opponent","Result":"*"})
                node = game
                while not board.is_game_over() and move_count < max_moves:
                    action_probs = mcts.get_move_probs(temperature)
                    if move_count == 0:
                        moves = list(action_probs.keys())
                        noise = np.random.dirichlet([0.3]*len(moves))
                        action_probs = {mv: 0.75*p + 0.25*n for mv, p, n in zip(moves, action_probs.values(), noise)}
                    if not action_probs:
                        break
                    moves, probs = list(action_probs.keys()), np.array(list(action_probs.values()), dtype=np.float32)
                    probs /= probs.sum()
                    chosen_move = np.random.choice(moves, p=probs)
                    states.append(convert_board_to_tensor(board))
                    prob_arr = np.zeros(get_total_moves(), dtype=np.float32)
                    move_mapping = get_move_mapping()
                    for mv, p in action_probs.items():
                        idx = move_mapping.get_index_by_move(mv)
                        if idx is not None and 0 <= idx < get_total_moves():
                            prob_arr[idx] = p
                    mcts_probs.append(prob_arr)
                    current_players.append(board.turn)
                    try:
                        board.push(chosen_move)
                    except ValueError:
                        break
                    node = node.add_variation(chosen_move)
                    mcts.update_with_move(chosen_move)
                    move_count +=1
                result_map = {'1-0':1.0, '0-1':-1.0, '1/2-1/2':0.0}
                result = result_map.get(board.result(), 0.0)
                if board.is_checkmate():
                    last_player = not board.turn
                    winners = [result if pl == last_player else -result for pl in current_players]
                else:
                    winners = [0.0]*len(current_players)
                game_length = len(states)
                game.headers["Result"] = "1-0" if result > 0 else "0-1" if result < 0 else "1/2-1/2"
                pgn_games.append(game)
                inputs.extend(states)
                policy_targets.extend(mcts_probs)
                value_targets.extend(winners)
                results.append(result)
                game_lengths.append(game_length)
            total_games = len(results)
            wins = results.count(1.0)
            losses = results.count(-1.0)
            draws = results.count(0.0)
            avg_length = sum(game_lengths)/len(game_lengths) if game_lengths else 0.0
            stats_queue.put({"total_games": total_games, "wins": wins, "losses": losses, "draws": draws, "avg_game_length": avg_length})
        except Exception as e:
            stats_queue.put({"error": f"Worker exception: {str(e)}"})
        return (inputs, policy_targets, value_targets, results, game_lengths, pgn_games)