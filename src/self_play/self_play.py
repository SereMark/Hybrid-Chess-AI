import numpy as np, chess, torch
from src.utils.mcts import MCTS
from src.utils.chess_utils import get_move_mapping, convert_board_to_tensor, get_total_moves
from src.models.model import ChessModel

class SelfPlay:
    def __init__(self, model_state_dict, device, n_simulations=800, c_puct=1.4, temperature=1.0, stats_fn=None):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.stats_fn = stats_fn
        self.move_mapping = get_move_mapping()
        self.total_moves = get_total_moves()
        self.model = ChessModel(num_moves=self.total_moves)
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def policy_value_fn(self, board):
        board_tensor = convert_board_to_tensor(board)
        board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(self.device)
        policy_logits, value = self.model(board_tensor)
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value.cpu().item()
        legal_moves = list(board.legal_moves)
        action_probs = {}
        total_legal_prob = 0
        for move in legal_moves:
            move_index = self.move_mapping.get_index_by_move(move)
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

    def play_game(self):
        board = chess.Board()
        mcts = MCTS(self.policy_value_fn, self.c_puct, self.n_simulations)
        mcts.set_root_node(board)
        states, mcts_probs, current_players = [], [], []
        move_count = 0
        max_moves = 200
        total_visits = 0
        num_moves = 0
        while not board.is_game_over() and move_count < max_moves:
            action_probs = mcts.get_move_probs(self.temperature)
            moves = list(action_probs.keys())
            probs = np.array(list(action_probs.values()))
            probs /= np.sum(probs)
            move = np.random.choice(moves, p=probs)
            board_tensor = convert_board_to_tensor(board)
            states.append(board_tensor)
            prob_array = np.zeros(self.total_moves, dtype=np.float32)
            for m, p in action_probs.items():
                move_index = self.move_mapping.get_index_by_move(m)
                if move_index is not None and 0 <= move_index < self.total_moves:
                    prob_array[move_index] = p
            mcts_probs.append(prob_array)
            current_players.append(board.turn)
            board.push(move)
            mcts.update_with_move(move)
            if mcts.root:
                total_visits += mcts.root.n_visits
                num_moves += 1
            move_count += 1

        result = self.get_game_result(board)
        if board.is_checkmate():
            last_player = not board.turn
            winners = [result if player == last_player else -result for player in current_players]
        else:
            winners = [0.0 for _ in current_players]
        game_length = len(states)
        avg_mcts_visits = (total_visits / num_moves) if num_moves > 0 else 0
        return states, mcts_probs, winners, game_length, result, avg_mcts_visits

    @staticmethod
    def get_game_result(board):
        result = board.result()
        if result == '1-0':
            return 1.0
        elif result == '0-1':
            return -1.0
        else:
            return 0.0