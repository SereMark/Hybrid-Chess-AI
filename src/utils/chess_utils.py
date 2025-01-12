import chess, numpy as np
import torch
from typing import Tuple, Dict

class MoveMapping:
    def __init__(self):
        self.MOVE_MAPPING = {}
        self.INDEX_MAPPING = {}
        self.TOTAL_MOVES = 0
        idx = 0
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq == to_sq:
                    continue
                move = chess.Move(from_sq, to_sq)
                if move != chess.Move.null():
                    self.MOVE_MAPPING[idx] = move
                    self.INDEX_MAPPING[move] = idx
                    idx += 1
                    if chess.square_rank(to_sq) in [0, 7]:
                        for promo in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                            promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                            self.MOVE_MAPPING[idx] = promo_move
                            self.INDEX_MAPPING[promo_move] = idx
                            idx += 1
        self.TOTAL_MOVES = idx

    def get_move_by_index(self, index):
        return self.MOVE_MAPPING.get(index)

    def get_index_by_move(self, move):
        return self.INDEX_MAPPING.get(move)

move_mapping = MoveMapping()

def get_total_moves():
    return move_mapping.TOTAL_MOVES

def get_move_mapping():
    return move_mapping

def flip_board(board):
    return board.mirror()

def flip_move(move):
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion
    )

def convert_board_to_tensor(board):
    planes = np.zeros((20, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    piece_type_indices = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    for sq, piece in piece_map.items():
        idx = piece_type_indices.get((piece.piece_type, piece.color))
        if idx is not None:
            row = sq // 8
            col = sq % 8
            planes[idx, row, col] = 1

    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]
    for i, c_right in enumerate(castling_rights):
        if c_right:
            planes[12 + i, :, :] = 1

    if board.ep_square is not None:
        ep_row = board.ep_square // 8
        ep_col = board.ep_square % 8
        planes[16, ep_row, ep_col] = 1

    planes[17, :, :] = board.halfmove_clock / 100.0
    planes[18, :, :] = board.fullmove_number / 100.0
    planes[19, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return planes

@torch.no_grad()
def policy_value_fn(board: chess.Board, model, device) -> Tuple[Dict[chess.Move, float], float]:
    board_tensor = convert_board_to_tensor(board)
    board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(device)
    policy_logits, value_out = model(board_tensor)
    policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
    value_float = value_out.cpu().item()
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return {}, value_float

    action_probs = {}
    total_prob = 0.0
    for move in legal_moves:
        idx = move_mapping.get_index_by_move(move)
        if idx is not None and idx < len(policy):
            prob = max(policy[idx], 1e-8)
            action_probs[move] = prob
            total_prob += prob
        else:
            action_probs[move] = 1e-8

    if total_prob > 0:
        for move in action_probs:
            action_probs[move] /= total_prob
    else:
        uniform_prob = 1.0 / len(legal_moves)
        for move in action_probs:
            action_probs[move] = uniform_prob

    return action_probs, value_float