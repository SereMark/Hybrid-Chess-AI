from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import chesscore as ccore
import config as C


DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

_FILE_NAMES = "abcdefgh"
_PIECE_SAN = {0: "", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K"}
_PROMO_SAN = {"q": "Q", "r": "R", "b": "B", "n": "N"}


# ---------- result

@dataclass(slots=True)
class ArenaResult:
    """Summary of a candidate-versus-baseline evaluation match."""
    games: int
    candidate_wins: int
    baseline_wins: int
    draws: int
    score_pct: float
    draw_pct: float
    decisive_pct: float
    elapsed_s: float
    notes: list[str] = field(default_factory=list)


# ---------- routed evaluator

class _ColourRoutedEvaluator:
    """Dispatch network calls to white/black evaluators based on side to move."""

    def __init__(self, white_eval: Any, black_eval: Any) -> None:
        self.white_eval = white_eval
        self.black_eval = black_eval

    def infer_positions_legal(
        self, positions: list[Any], moves_per_position: list[list[Any]]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        if not positions:
            return [], np.zeros((0,), dtype=np.float32)

        w_idx: list[int] = []
        b_idx: list[int] = []
        w_pos: list[Any] = []
        b_pos: list[Any] = []
        w_moves: list[list[Any]] = []
        b_moves: list[list[Any]] = []

        for i, pos in enumerate(positions):
            if getattr(pos, "turn", ccore.WHITE) == ccore.WHITE:
                w_idx.append(i); w_pos.append(pos); w_moves.append(moves_per_position[i])
            else:
                b_idx.append(i); b_pos.append(pos); b_moves.append(moves_per_position[i])

        pol_out: list[np.ndarray] = [np.zeros((0,), dtype=np.float32)] * len(positions)
        val_out = np.zeros((len(positions),), dtype=np.float32)

        if w_pos:
            pol_list, val_arr = self.white_eval.infer_positions_legal(w_pos, w_moves)
            for dst, pol, val in zip(w_idx, pol_list, val_arr, strict=False):
                pol_out[dst] = pol; val_out[dst] = float(val)
        if b_pos:
            pol_list, val_arr = self.black_eval.infer_positions_legal(b_pos, b_moves)
            for dst, pol, val in zip(b_idx, pol_list, val_arr, strict=False):
                pol_out[dst] = pol; val_out[dst] = float(val)
        return pol_out, val_out

    def infer_values(self, positions: list[Any]) -> np.ndarray:
        if not positions:
            return np.zeros((0,), dtype=np.float32)

        w_idx: list[int] = []
        b_idx: list[int] = []
        w_pos: list[Any] = []
        b_pos: list[Any] = []
        for i, pos in enumerate(positions):
            if getattr(pos, "turn", ccore.WHITE) == ccore.WHITE:
                w_idx.append(i); w_pos.append(pos)
            else:
                b_idx.append(i); b_pos.append(pos)

        values = np.zeros((len(positions),), dtype=np.float32)
        if w_pos:
            arr = self.white_eval.infer_values(w_pos)
            for dst, v in zip(w_idx, arr, strict=False):
                values[dst] = float(v)
        if b_pos:
            arr = self.black_eval.infer_values(b_pos)
            for dst, v in zip(b_idx, arr, strict=False):
                values[dst] = float(v)
        return values


# ---------- public API

def play_match(
    candidate_eval: Any,
    baseline_eval: Any,
    *,
    games: int,
    seed: int | None = None,
    start_fen_fn: Callable[[np.random.Generator], str] | None = None,
    pgn_dir: Path | None = None,
    label: str | None = None,
) -> ArenaResult:
    """Head-to-head match between candidate and baseline evaluators."""
    rng = np.random.default_rng(seed)
    games = int(max(0, games))
    t0 = time.time()

    cand_w = base_w = draws = 0
    cand_score = 0.0

    if pgn_dir is not None:
        pgn_dir.mkdir(parents=True, exist_ok=True)

    for g in range(games):
        cand_white = (g % 2) == 0
        white_eval = candidate_eval if cand_white else baseline_eval
        black_eval = baseline_eval if cand_white else candidate_eval
        evaluator = _ColourRoutedEvaluator(white_eval, black_eval)

        start_fen = start_fen_fn(rng) if start_fen_fn is not None else DEFAULT_START_FEN
        record_moves = pgn_dir is not None
        result, moves_san = _play_single_game(evaluator, rng, start_fen, record_moves=record_moves)

        if result == ccore.WHITE_WIN:
            if cand_white:
                cand_w += 1; cand_score += 1.0
            else:
                base_w += 1
        elif result == ccore.BLACK_WIN:
            if cand_white:
                base_w += 1
            else:
                cand_w += 1; cand_score += 1.0
        else:
            draws += 1; cand_score += 0.5

        if record_moves and pgn_dir is not None:
            _save_pgn(
                pgn_dir,
                label or "arena",
                g,
                start_fen,
                moves_san,
                _result_to_str(result),
                white_name="Candidate" if cand_white else "Baseline",
                black_name="Baseline" if cand_white else "Candidate",
            )

    elapsed = time.time() - t0
    tot = max(1, games)
    decisive = cand_w + base_w
    return ArenaResult(
        games=games,
        candidate_wins=cand_w,
        baseline_wins=base_w,
        draws=draws,
        score_pct=100.0 * cand_score / tot,
        draw_pct=100.0 * draws / tot,
        decisive_pct=100.0 * decisive / tot,
        elapsed_s=elapsed,
        notes=[],
    )


# ---------- internals

def _play_single_game(
    evaluator: _ColourRoutedEvaluator,
    rng: np.random.Generator,
    start_fen: str,
    *,
    record_moves: bool = False,
) -> tuple[ccore.Result, list[str]]:
    pos = ccore.Position()
    try:
        pos.from_fen(start_fen)
    except Exception:
        pos.from_fen(DEFAULT_START_FEN)

    mcts = _build_mcts(rng)
    moves_done = 0
    max_plies = int(max(1, C.ARENA.max_game_plies))
    resign_enabled = bool(C.ARENA.resign_enable)
    resign_threshold = float(C.RESIGN.value_threshold)
    resign_consecutive = int(max(1, C.RESIGN.consecutive_required))
    resign_streak = 0
    moves_san: list[str] = [] if record_moves else []

    while moves_done < max_plies:
        result = pos.result()
        if result != ccore.ONGOING:
            return result, moves_san

        legal = list(pos.legal_moves())
        if not legal:
            return pos.result(), moves_san

        counts = mcts.search_batched_legal(pos, evaluator.infer_positions_legal, int(max(1, C.EVAL.batch_size_max)))
        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape[0] != len(legal):
            return ccore.DRAW, moves_san

        temperature = C.ARENA.temperature if moves_done < C.ARENA.temperature_moves else C.SELFPLAY.deterministic_temp_eps
        mv = legal[_select_move(counts, temperature, rng)]
        mv_str = str(mv)

        if record_moves:
            moves_san.append(_move_to_san(pos, mv, mv_str, legal))

        val = float(evaluator.infer_values([pos])[0])
        stm_white = bool(getattr(pos, "turn", ccore.WHITE) == ccore.WHITE)
        player_view = val if stm_white else -val
        if resign_enabled:
            if player_view <= resign_threshold:
                resign_streak += 1
                if resign_streak >= resign_consecutive:
                    res = ccore.BLACK_WIN if stm_white else ccore.WHITE_WIN
                    return res, moves_san
            else:
                resign_streak = 0

        pos.make_move(mv)
        try:
            mcts.advance_root(pos, mv)
        except Exception:
            mcts = _build_mcts(rng)
        moves_done += 1

    return ccore.DRAW, moves_san


def _build_mcts(rng: np.random.Generator) -> ccore.MCTS:
    m = ccore.MCTS(
        int(C.ARENA.mcts_simulations),
        float(C.MCTS.c_puct),
        float(C.MCTS.dirichlet_alpha),
        float(C.MCTS.dirichlet_weight),
    )
    m.set_c_puct_params(float(C.MCTS.c_puct_base), float(C.MCTS.c_puct_init))
    m.set_fpu_reduction(float(C.MCTS.fpu_reduction))
    m.seed(int(rng.integers(1, np.iinfo(np.int64).max)))
    return m


def _select_move(visit_counts: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if visit_counts.ndim != 1 or visit_counts.size == 0:
        return 0
    if temperature <= C.SELFPLAY.deterministic_temp_eps:
        return int(np.argmax(visit_counts))
    scaled = np.maximum(visit_counts, 0.0) ** (1.0 / max(temperature, 1e-6))
    total = float(scaled.sum())
    if not np.isfinite(total) or total <= 0.0:
        return int(np.argmax(visit_counts))
    probs = np.asarray(scaled / total, dtype=np.float64)
    return int(rng.choice(len(probs), p=probs))


# ---------- SAN helpers

def _square_file(square: int) -> int:
    return int(square) & 7


def _square_rank(square: int) -> int:
    return int(square) >> 3


def _square_name(square: int) -> str:
    return f"{_FILE_NAMES[_square_file(square)]}{_square_rank(square) + 1}"


def _piece_on(pos: ccore.Position, square: int) -> tuple[ccore.Color, int] | None:
    mask = 1 << int(square)
    for idx, pair in enumerate(pos.pieces):
        try:
            w_bb, b_bb = int(pair[0]), int(pair[1])
        except Exception:
            continue
        if w_bb & mask:
            return ccore.Color.WHITE, idx
        if b_bb & mask:
            return ccore.Color.BLACK, idx
    return None


def _move_to_san(
    pos: ccore.Position,
    move: ccore.Move,
    move_str: str,
    legal_moves: Sequence[ccore.Move],
) -> str:
    from_sq = int(getattr(move, "from_square", 0))
    to_sq = int(getattr(move, "to_square", 0))
    info = _piece_on(pos, from_sq)
    if info is None:
        return move_str
    mover_color, piece_type = info

    capture = False
    tgt = _piece_on(pos, to_sq)
    if tgt is not None:
        capture = tgt[0] != mover_color
    elif piece_type == 0:
        ep_sq = int(getattr(pos, "ep_square", -1))
        capture = ep_sq >= 0 and ep_sq == to_sq

    # Castling
    if piece_type == 5:
        if abs(_square_file(to_sq) - _square_file(from_sq)) == 2:
            return "O-O" if _square_file(to_sq) > _square_file(from_sq) else "O-O-O"

    # Piece and disambiguation
    if piece_type == 0:
        san = (_FILE_NAMES[_square_file(from_sq)] + "x") if capture else ""
        san += _square_name(to_sq)
        if len(move_str) >= 5:
            san += "=" + _PROMO_SAN.get(move_str[-1].lower(), move_str[-1].upper())
    else:
        san = _PIECE_SAN.get(piece_type, "")
        ambiguous: list[int] = []
        for other in legal_moves:
            if other is move:
                continue
            if int(getattr(other, "to_square", -1)) != to_sq:
                continue
            other_info = _piece_on(pos, int(getattr(other, "from_square", -1)))
            if other_info and other_info[0] == mover_color and other_info[1] == piece_type:
                ambiguous.append(int(getattr(other, "from_square", -1)))
        if ambiguous:
            same_file = any(_square_file(s) == _square_file(from_sq) for s in ambiguous)
            same_rank = any(_square_rank(s) == _square_rank(from_sq) for s in ambiguous)
            if same_file and same_rank:
                san += _square_name(from_sq)
            elif same_file:
                san += str(_square_rank(from_sq) + 1)
            elif same_rank:
                san += _FILE_NAMES[_square_file(from_sq)]
            else:
                san += _square_name(from_sq)
        if capture:
            san += "x"
        san += _square_name(to_sq)

    # Checkmate marker
    nxt = ccore.Position(pos)
    try:
        nxt.make_move(move)
        res = nxt.result()
        if (mover_color == ccore.Color.WHITE and res == ccore.WHITE_WIN) or (
            mover_color == ccore.Color.BLACK and res == ccore.BLACK_WIN
        ):
            san += "#"
    except Exception:
        pass
    return san


# ---------- PGN helpers

def _result_to_str(result: ccore.Result) -> str:
    if result == ccore.WHITE_WIN:
        return "1-0"
    if result == ccore.BLACK_WIN:
        return "0-1"
    return "1/2-1/2"


def _save_pgn(
    directory: Path,
    label: str,
    game_index: int,
    start_fen: str,
    moves_san: Sequence[str],
    result_str: str,
    *,
    white_name: str,
    black_name: str,
) -> Path:
    filename = f"{label}_g{game_index + 1:02d}.pgn"
    path = directory / filename
    date_str = datetime.utcnow().strftime("%Y.%m.%d")
    headers = [
        ("Event", "Arena Evaluation"),
        ("Site", "Local"),
        ("Date", date_str),
        ("Round", str(game_index + 1)),
        ("White", white_name),
        ("Black", black_name),
        ("Result", result_str),
    ]
    if start_fen != DEFAULT_START_FEN:
        headers += [("SetUp", "1"), ("FEN", start_fen)]

    with path.open("w", encoding="utf-8") as h:
        for k, v in headers:
            h.write(f'[{k} "{v}"]\n')
        h.write("\n")
        if moves_san:
            chunks: list[str] = []
            for i, san in enumerate(moves_san):
                move_no = i // 2 + 1
                if i % 2 == 0:
                    chunks.append(f"{move_no}. {san}")
                else:
                    chunks[-1] += f" {san}"
            h.write(" ".join(chunks))
            h.write(f" {result_str}\n")
        else:
            h.write(f"{result_str}\n")
    return path