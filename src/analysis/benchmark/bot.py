import chess, torch, numpy as np
from typing import Dict
from src.training.reinforcement.mcts import MCTS
from src.utils.chess_utils import get_total_moves, convert_board_to_tensor, get_move_mapping
from src.models.transformer import TransformerChessModel

class Bot:
    def __init__(self, path: str, use_mcts: bool, use_opening_book: bool):
        self.use_mcts, self.use_opening_book = use_mcts, use_opening_book
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TransformerChessModel(get_total_moves()).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device)["model_state_dict"])
        self.model.eval()
        self.mcts = MCTS(model=self.model, device=torch.device(self.device), c_puct=1.4, n_simulations=100) if self.use_mcts else None

    def get_move(self, board: chess.Board, opening_book: Dict[str, Dict[str, Dict[str, int]]]) -> chess.Move:
        try:
            if self.use_opening_book:
                fen = board.fen()
                moves_data = opening_book.get(fen, {})
                best_move, best_score = None, -1.0
                for uci_move, stats in moves_data.items():
                    total = stats.get("win", 0) + stats.get("draw", 0) + stats.get("loss", 0)
                    if total:
                        score = (stats.get("win", 0) + 0.5 * stats.get("draw", 0)) / total
                        move_candidate = chess.Move.from_uci(uci_move)
                        if move_candidate in board.legal_moves and score > best_score:
                            best_move, best_score = move_candidate, score
                if best_move:
                    return best_move
            if self.use_mcts:
                self.mcts.set_root_node(board.copy())
                move_probs = self.mcts.get_move_probs(temperature=1e-3)
                if board.fullmove_number == 1 and board.turn == chess.WHITE:
                    moves = list(move_probs.keys())
                    noise = np.random.dirichlet([0.3] * len(moves))
                    move_probs = {mv: 0.75 * move_probs[mv] + 0.25 * noise[i] for i, mv in enumerate(moves)}
                    total_prob = sum(move_probs.values())
                    if total_prob > 1e-8:
                        move_probs = {mv: p / total_prob for mv, p in move_probs.items()}
                return max(move_probs, key=move_probs.get)
            board_tensor = torch.from_numpy(convert_board_to_tensor(board)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_logits, _ = self.model(board_tensor)
            policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
            legal_moves = list(board.legal_moves)
            move_mapping = get_move_mapping()
            action_probs, total_prob = {}, 0.0
            for move in legal_moves:
                idx = move_mapping.get_index_by_move(move)
                prob = max(policy[idx], 1e-8) if idx is not None and idx < len(policy) else 1e-8
                action_probs[move], total_prob = prob, total_prob + prob
            if total_prob > 0:
                action_probs = {mv: p / total_prob for mv, p in action_probs.items()}
            else:
                uniform_prob = 1.0 / len(legal_moves)
                action_probs = {mv: uniform_prob for mv in legal_moves}
            return max(action_probs, key=action_probs.get)
        except:
            return chess.Move.null()