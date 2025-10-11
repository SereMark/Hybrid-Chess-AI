"""Evaluation arena for candidate gating."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import chesscore as ccore
import config as C
import numpy as np

DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

__all__ = ["ArenaResult", "play_match"]

_FILE_NAMES = "abcdefgh"
_PIECE_SAN = {0: "", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K"}
_PROMO_SAN = {"q": "Q", "r": "R", "b": "B", "n": "N"}


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


class _ColourRoutedEvaluator:
    """Dispatches evaluation requests to the appropriate player model."""

    def __init__(self, white_eval: Any, black_eval: Any) -> None:
        self.white_eval = white_eval
        self.black_eval = black_eval

    def infer_positions_legal(
        self, positions: list[Any], moves_per_position: list[list[Any]]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        if not positions:
            return [], np.zeros((0,), dtype=np.float32)
        white_indices: list[int] = []
        black_indices: list[int] = []
        white_pos: list[Any] = []
        black_pos: list[Any] = []
        white_moves: list[list[Any]] = []
        black_moves: list[list[Any]] = []

        for idx, pos in enumerate(positions):
            turn = getattr(pos, "turn", ccore.WHITE)
            if turn == ccore.WHITE:
                white_indices.append(idx)
                white_pos.append(pos)
                white_moves.append(moves_per_position[idx])
            else:
                black_indices.append(idx)
                black_pos.append(pos)
                black_moves.append(moves_per_position[idx])

        pol_out: list[np.ndarray] = [np.zeros((0,), dtype=np.float32)] * len(positions)
        val_out = np.zeros((len(positions),), dtype=np.float32)

        if white_pos:
            pol_list, val_arr = self.white_eval.infer_positions_legal(white_pos, white_moves)
            for dest, pol, val in zip(white_indices, pol_list, val_arr, strict=False):
                pol_out[dest] = pol
                val_out[dest] = float(val)
        if black_pos:
            pol_list, val_arr = self.black_eval.infer_positions_legal(black_pos, black_moves)
            for dest, pol, val in zip(black_indices, pol_list, val_arr, strict=False):
                pol_out[dest] = pol
                val_out[dest] = float(val)
        return pol_out, val_out

    def infer_values(self, positions: list[Any]) -> np.ndarray:
        if not positions:
            return np.zeros((0,), dtype=np.float32)

        white_indices: list[int] = []
        black_indices: list[int] = []
        white_pos: list[Any] = []
        black_pos: list[Any] = []
        for idx, pos in enumerate(positions):
            turn = getattr(pos, "turn", ccore.WHITE)
            if turn == ccore.WHITE:
                white_indices.append(idx)
                white_pos.append(pos)
            else:
                black_indices.append(idx)
                black_pos.append(pos)

        values = np.zeros((len(positions),), dtype=np.float32)
        if white_pos:
            arr = self.white_eval.infer_values(white_pos)
            for dest, val in zip(white_indices, arr, strict=False):
                values[dest] = float(val)
        if black_pos:
            arr = self.black_eval.infer_values(black_pos)
            for dest, val in zip(black_indices, arr, strict=False):
                values[dest] = float(val)
        return values


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
    """Play a head-to-head match between candidate and baseline evaluators."""
    rng = np.random.default_rng(seed)
    games = int(max(0, games))
    start_time = time.time()

    candidate_wins = 0
    baseline_wins = 0
    draws = 0
    candidate_score = 0.0

    if pgn_dir is not None:
        pgn_dir.mkdir(parents=True, exist_ok=True)

    for game_idx in range(games):
        cand_white = (game_idx % 2) == 0
        white_eval = candidate_eval if cand_white else baseline_eval
        black_eval = baseline_eval if cand_white else candidate_eval
        evaluator = _ColourRoutedEvaluator(white_eval, black_eval)
        start_fen = start_fen_fn(rng) if start_fen_fn is not None else DEFAULT_START_FEN
        record_moves = pgn_dir is not None
        result, moves_san = _play_single_game(
            evaluator,
            rng,
            start_fen,
            record_moves=record_moves,
        )
        if result == ccore.WHITE_WIN:
            if cand_white:
                candidate_wins += 1
                candidate_score += 1.0
            else:
                baseline_wins += 1
        elif result == ccore.BLACK_WIN:
            if cand_white:
                baseline_wins += 1
            else:
                candidate_wins += 1
                candidate_score += 1.0
        else:
            draws += 1
            candidate_score += 0.5
        if record_moves and pgn_dir is not None:
            result_str = _result_to_str(result)
            pgn_path = _save_pgn(
                pgn_dir,
                label or "arena",
                game_idx,
                start_fen,
                moves_san,
                result_str,
                white_name="Candidate" if cand_white else "Baseline",
                black_name="Baseline" if cand_white else "Candidate",
            )

    elapsed = time.time() - start_time
    total_games = max(1, games)
    decisive = candidate_wins + baseline_wins
    score_pct = 100.0 * candidate_score / total_games
    draw_pct = 100.0 * draws / total_games
    decisive_pct = 100.0 * decisive / total_games
    return ArenaResult(
        games=games,
        candidate_wins=candidate_wins,
        baseline_wins=baseline_wins,
        draws=draws,
        score_pct=score_pct,
        draw_pct=draw_pct,
        decisive_pct=decisive_pct,
        elapsed_s=elapsed,
        notes=[],
    )


def _play_single_game(
    evaluator: _ColourRoutedEvaluator,
    rng: np.random.Generator,
    start_fen: str,
    *,
    record_moves: bool = False,
) -> tuple[ccore.Result, list[str]]:
    position = ccore.Position()
    try:
        position.from_fen(start_fen)
    except Exception:
        position = ccore.Position()
        position.from_fen(DEFAULT_START_FEN)
    mcts = _build_mcts(rng)
    move_count = 0
    max_plies = int(max(1, C.ARENA.max_game_plies))
    resign_enabled = bool(C.ARENA.resign_enable)
    resign_threshold = float(C.RESIGN.value_threshold)
    resign_consecutive = int(max(1, C.RESIGN.consecutive_required))
    resign_streak = 0
    moves_san: list[str] = [] if record_moves else []

    while move_count < max_plies:
        result = position.result()
        if result != ccore.ONGOING:
            return result, moves_san

        moves = list(position.legal_moves())
        if not moves:
            return position.result(), moves_san

        visit_counts = mcts.search_batched_legal(
            position,
            evaluator.infer_positions_legal,
            int(max(1, C.EVAL.batch_size_max)),
        )
        visit_counts = np.asarray(visit_counts, dtype=np.float64)
        if visit_counts.shape[0] != len(moves):
            return ccore.DRAW, moves_san

        temperature = (
            C.ARENA.temperature if move_count < C.ARENA.temperature_moves else C.SELFPLAY.deterministic_temp_eps
        )
        move_idx = _select_move(visit_counts, temperature, rng)
        move = moves[move_idx]
        move_str = str(move)

        if record_moves:
            san = _move_to_san(position, move, move_str, moves)
            moves_san.append(san)

        value = float(evaluator.infer_values([position])[0])
        stm_is_white = bool(getattr(position, "turn", ccore.WHITE) == ccore.WHITE)
        player_view = value if stm_is_white else -value
        if resign_enabled:
            if player_view <= resign_threshold:
                resign_streak += 1
                if resign_streak >= resign_consecutive:
                    result = ccore.BLACK_WIN if stm_is_white else ccore.WHITE_WIN
                    return result, moves_san
            else:
                resign_streak = 0

        position.make_move(move)
        try:
            mcts.advance_root(position, move)
        except Exception:
            mcts = _build_mcts(rng)
        move_count += 1

    return ccore.DRAW, moves_san


def _build_mcts(rng: np.random.Generator) -> ccore.MCTS:
    mcts = ccore.MCTS(
        int(C.ARENA.mcts_simulations),
        float(C.MCTS.c_puct),
        float(C.MCTS.dirichlet_alpha),
        float(C.MCTS.dirichlet_weight),
    )
    mcts.set_c_puct_params(float(C.MCTS.c_puct_base), float(C.MCTS.c_puct_init))
    mcts.set_fpu_reduction(float(C.MCTS.fpu_reduction))
    mcts.seed(int(rng.integers(1, np.iinfo(np.int64).max)))
    return mcts


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
            white_bb, black_bb = int(pair[0]), int(pair[1])
        except Exception:
            continue
        if white_bb & mask:
            return ccore.Color.WHITE, idx
        if black_bb & mask:
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
    piece_info = _piece_on(pos, from_sq)
    if piece_info is None:
        return move_str
    mover_color, piece_type = piece_info
    capture = False
    target_info = _piece_on(pos, to_sq)
    if target_info is not None:
        capture = target_info[0] != mover_color
    elif piece_type == 0:
        ep_square = int(getattr(pos, "ep_square", -1))
        if ep_square >= 0 and ep_square == to_sq:
            capture = True

    if piece_type == 5:
        delta_file = abs(_square_file(to_sq) - _square_file(from_sq))
        if delta_file == 2:
            if _square_file(to_sq) > _square_file(from_sq):
                return "O-O"
            return "O-O-O"

    san = ""
    if piece_type == 0:
        if capture:
            san += _FILE_NAMES[_square_file(from_sq)] + "x"
        san += _square_name(to_sq)
        if len(move_str) >= 5:
            promo_char = move_str[-1].lower()
            san += "=" + _PROMO_SAN.get(promo_char, promo_char.upper())
    else:
        san += _PIECE_SAN.get(piece_type, "")
        ambiguous_sources: list[int] = []
        for other in legal_moves:
            if other is move:
                continue
            if int(getattr(other, "to_square", -1)) != to_sq:
                continue
            other_info = _piece_on(pos, int(getattr(other, "from_square", -1)))
            if other_info is None:
                continue
            other_color, other_type = other_info
            if other_color == mover_color and other_type == piece_type:
                ambiguous_sources.append(int(getattr(other, "from_square", -1)))
        if ambiguous_sources:
            same_file = any(_square_file(src) == _square_file(from_sq) for src in ambiguous_sources)
            same_rank = any(_square_rank(src) == _square_rank(from_sq) for src in ambiguous_sources)
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
    next_pos = ccore.Position(pos)
    try:
        next_pos.make_move(move)
    except Exception:
        return san
    result = next_pos.result()
    try:
        if (mover_color == ccore.Color.WHITE and result == ccore.WHITE_WIN) or (
            mover_color == ccore.Color.BLACK and result == ccore.BLACK_WIN
        ):
            san += "#"
    except Exception:
        pass
    return san


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
    use_setup = start_fen != DEFAULT_START_FEN
    if use_setup:
        headers.append(("SetUp", "1"))
        headers.append(("FEN", start_fen))
    with path.open("w", encoding="utf-8") as handle:
        for key, value in headers:
            handle.write(f"[{key} \"{value}\"]\n")
        handle.write("\n")
        if moves_san:
            chunks: list[str] = []
            for idx, san in enumerate(moves_san):
                move_no = idx // 2 + 1
                if idx % 2 == 0:
                    chunks.append(f"{move_no}. {san}")
                else:
                    chunks[-1] += f" {san}"
            handle.write(" ".join(chunks))
            handle.write(f" {result_str}\n")
        else:
            handle.write(f"{result_str}\n")
    return path
