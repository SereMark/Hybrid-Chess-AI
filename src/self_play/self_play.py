import numpy as np, chess, torch, random
from src.self_play.mcts import MCTS
from src.utils.chess_utils import INDEX_MAPPING, convert_board_to_tensor, TOTAL_MOVES


class SelfPlay:
    def __init__(self, model, device, n_simulations=800, c_puct=1.4, temperature=1.0):
        self.model = model
        self.device = device
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.log_update = None

    def set_logger(self, log_update):
        self.log_update = log_update

    def log(self, message):
        if self.log_update:
            self.log_update.emit(message)

    @torch.no_grad()
    def policy_value_fn(self, board):
        try:
            board_tensor = convert_board_to_tensor(board)
            board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(self.device)

            policy_logits, value = self.model(board_tensor)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.cpu().item()

            legal_moves = list(board.legal_moves)
            action_probs = {}
            total_legal_prob = 0

            for move in legal_moves:
                move_index = INDEX_MAPPING.get(move)
                if move_index is not None and move_index < len(policy):
                    prob = max(policy[move_index], 1e-8)
                    action_probs[move] = prob
                    total_legal_prob += prob
                else:
                    action_probs[move] = 1e-8

            if total_legal_prob > 0:
                action_probs = {k: v / total_legal_prob for k, v in action_probs.items()}
            else:
                action_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}

            return action_probs, value

        except Exception as e:
            self.log(f"Error in policy_value_fn: {str(e)}")
            legal_moves = list(board.legal_moves)
            return {move: 1.0 / len(legal_moves) for move in legal_moves}, 0.0

    def play_game(self):
        try:
            board = chess.Board()
            mcts = MCTS(self.policy_value_fn, self.c_puct, self.n_simulations)
            mcts.set_root_node(board)
            states, mcts_probs, current_players = [], [], []
            move_count = 0
            max_moves = 200

            while not board.is_game_over() and move_count < max_moves:
                if board.is_repetition(2):
                    break

                piece_count = len(board.piece_map())
                if piece_count > 24:
                    mcts.n_simulations = self.n_simulations // 2
                elif piece_count < 10:
                    mcts.n_simulations = self.n_simulations
                else:
                    mcts.n_simulations = self.n_simulations * 3 // 4

                temperature = 1.0 if move_count < 30 else self.temperature
                action_probs = mcts.get_move_probs(temperature)

                captures = [move for move in board.legal_moves if board.is_capture(move)]
                if captures and random.random() < 0.8:
                    move = random.choice(captures)
                    probs = np.zeros(len(action_probs))
                    move_indices = list(action_probs.keys())
                    move_pos = move_indices.index(move) if move in move_indices else -1
                    if move_pos != -1:
                        probs[move_pos] = 1.0
                    else:
                        probs = np.ones(len(action_probs)) / len(action_probs)
                else:
                    moves = list(action_probs.keys())
                    probs = np.array(list(action_probs.values()))
                    probs /= np.sum(probs)
                    move = np.random.choice(moves, p=probs)

                board_tensor = convert_board_to_tensor(board)
                states.append(board_tensor)

                prob_array = np.zeros(TOTAL_MOVES, dtype=np.float32)
                for m, p in action_probs.items():
                    move_index = INDEX_MAPPING.get(m)
                    if move_index is not None and 0 <= move_index < TOTAL_MOVES:
                        prob_array[move_index] = p
                mcts_probs.append(prob_array)
                current_players.append(board.turn)

                board.push(move)
                mcts.update_with_move(move)
                move_count += 1

            result = self.get_game_result(board)
            if board.is_checkmate():
                last_player = not board.turn
                winners = [result if player == last_player else -result for player in current_players]
            else:
                winners = [0.0 for _ in current_players]
            game_length = len(states)

            return states, mcts_probs, winners, game_length, result

        except Exception as e:
            self.log(f"Error in play_game: {str(e)}")
            raise

    @staticmethod
    def get_game_result(board):
        result = board.result()
        if result == '1-0':
            return 1.0
        elif result == '0-1':
            return -1.0
        else:
            return 0.0