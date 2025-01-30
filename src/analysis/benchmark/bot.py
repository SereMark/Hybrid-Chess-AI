import os, chess, torch, numpy as np
from src.training.reinforcement.mcts import MCTS
from src.utils.chess_utils import get_total_moves, convert_board_to_transformer_input, get_move_mapping
from src.models.transformer import SimpleTransformerChessModel

class Bot:
    def __init__(self, path, use_mcts, use_opening_book):
        self.use_mcts = use_mcts
        self.use_opening_book = use_opening_book
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleTransformerChessModel(get_total_moves()).to(self.device)
        self._load_model_checkpoint(path)
        self.mcts = MCTS(self.model, self.device, c_puct=1.4, n_simulations=100) if self.use_mcts else None
        self.move_map = get_move_mapping()
    def _load_model_checkpoint(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            c = torch.load(path, map_location=self.device)
            if isinstance(c, dict) and "model_state_dict" in c:
                self.model.load_state_dict(c["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(c, strict=False)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    def get_move(self, board, opening_book):
        try:
            if self.use_opening_book:
                bm = self._choose_book_move(board, opening_book)
                if bm is not None:
                    return bm
            if self.use_mcts and self.mcts:
                return self._choose_mcts_move(board)
            return self._choose_direct_policy_move(board)
        except:
            return chess.Move.null()
    def _choose_book_move(self, board, opening_book):
        d = opening_book.get(board.fen(), {})
        if not d:
            return None
        best, best_score = None, -1
        for uci, stats in d.items():
            t = stats.get("win",0)+stats.get("draw",0)+stats.get("loss",0)
            if t>0:
                s = (stats.get("win",0)+0.5*stats.get("draw",0))/t
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves and s>best_score:
                    best, best_score = mv, s
        return best
    def _choose_mcts_move(self, board):
        self.mcts.set_root_node(board.copy())
        probs = self.mcts.get_move_probs(temperature=1e-3)
        if board.fullmove_number==1 and board.turn==chess.WHITE and len(probs)>1:
            mv_list = list(probs.keys())
            arr = np.array(list(probs.values()), dtype=np.float32)
            noise = np.random.dirichlet([0.3]*len(mv_list))
            arr = 0.75*arr + 0.25*noise
            arr /= arr.sum()
            probs = dict(zip(mv_list, arr))
        return max(probs, key=probs.get) if probs else chess.Move.null()
    def _choose_direct_policy_move(self, board):
        x = torch.from_numpy(convert_board_to_transformer_input(board)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x)
            policy = torch.softmax(logits[0], dim=0).cpu().numpy()
        legals = list(board.legal_moves)
        if not legals:
            return chess.Move.null()
        ap = {}
        for mv in legals:
            idx = self.move_map.get_index_by_move(mv)
            pr = policy[idx] if idx is not None else 1e-12
            if pr<1e-12: pr=1e-12
            ap[mv] = pr
        s = sum(ap.values())
        if s>0:
            for mv in ap: ap[mv]/=s
        else:
            for mv in ap: ap[mv]=1/len(legals)
        return max(ap, key=ap.get)