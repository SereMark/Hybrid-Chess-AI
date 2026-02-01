from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import chesscore as ccore
import config as C
import numpy as np
from utils import DEFAULT_START_FEN, select_visit_count_move

PGN_EVENT = "Arena Evaluation"

__all__ = ["ArenaResult", "play_match"]

_FILE_NAMES = "abcdefgh"
_PIECE_SAN = {0: "", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K"}
_PROMO_SAN = {"q": "Q", "r": "R", "b": "B", "n": "N"}


@dataclass(slots=True)
class ArenaResult:
    games: int
    candidate_wins: int
    baseline_wins: int
    draws: int
    score_pct: float
    draw_pct: float
    decisive_pct: float
    elapsed_s: float
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _ScoreTracker:
    total_games: int
    candidate_wins: int = 0
    baseline_wins: int = 0
    draws: int = 0
    candidate_points: float = 0.0

    def record(self, candidate_white: bool, result: ccore.Result) -> None:
        if result == ccore.WHITE_WIN:
            if candidate_white:
                self.candidate_wins += 1
                self.candidate_points += 1.0
            else:
                self.baseline_wins += 1
        elif result == ccore.BLACK_WIN:
            if candidate_white:
                self.baseline_wins += 1
            else:
                self.candidate_wins += 1
                self.candidate_points += 1.0
        else:
            self.draws += 1
            self.candidate_points += 0.5

    def to_result(self, elapsed_s: float, *, notes: Sequence[str] | None = None) -> ArenaResult:
        denom = max(1, self.total_games)
        decisive = self.candidate_wins + self.baseline_wins
        result_notes = list(notes) if notes is not None else []
        return ArenaResult(
            games=self.total_games,
            candidate_wins=self.candidate_wins,
            baseline_wins=self.baseline_wins,
            draws=self.draws,
            score_pct=100.0 * self.candidate_points / denom,
            draw_pct=100.0 * self.draws / denom,
            decisive_pct=100.0 * decisive / denom,
            elapsed_s=float(elapsed_s),
            notes=result_notes,
        )


class _ColourRoutedEvaluator:
    def __init__(self, white_eval: Any, black_eval: Any) -> None:
        self._white = white_eval
        self._black = black_eval

    def infer_positions_legal(
        self,
        positions: list[Any],
        moves_per_position: list[list[Any]] | np.ndarray,
        counts: list[int] | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        if not positions:
            return [], np.zeros((0,), dtype=np.float32)

        if counts is not None:
            flat_moves = moves_per_position
            moves_expanded: list[Any] = []
            offset = 0
            for c in counts:
                moves_expanded.append(flat_moves[offset : offset + c])
                offset += c
            moves_per_position = moves_expanded

        white_indices: list[int] = []
        black_indices: list[int] = []
        white_positions: list[Any] = []
        black_positions: list[Any] = []
        white_moves: list[list[Any]] = []
        black_moves: list[list[Any]] = []

        for i, pos in enumerate(positions):
            if getattr(pos, "turn", ccore.WHITE) == ccore.WHITE:
                white_indices.append(i)
                white_positions.append(pos)
                white_moves.append(moves_per_position[i])
            else:
                black_indices.append(i)
                black_positions.append(pos)
                black_moves.append(moves_per_position[i])

        policy_out: list[np.ndarray] = [np.zeros((0,), dtype=np.float32) for _ in range(len(positions))]
        value_out = np.zeros((len(positions),), dtype=np.float32)

        if white_positions:
            policies, values = self._white.infer_positions_legal(white_positions, white_moves)
            for dst, pol, val in zip(white_indices, policies, values, strict=False):
                policy_out[dst] = pol
                value_out[dst] = float(val)

        if black_positions:
            policies, values = self._black.infer_positions_legal(black_positions, black_moves)
            for dst, pol, val in zip(black_indices, policies, values, strict=False):
                policy_out[dst] = pol
                value_out[dst] = float(val)

        return policy_out, value_out

    def infer_values(self, positions: list[Any]) -> np.ndarray:
        if not positions:
            return np.zeros((0,), dtype=np.float32)

        white_indices: list[int] = []
        black_indices: list[int] = []
        white_positions: list[Any] = []
        black_positions: list[Any] = []

        for i, pos in enumerate(positions):
            if getattr(pos, "turn", ccore.WHITE) == ccore.WHITE:
                white_indices.append(i)
                white_positions.append(pos)
            else:
                black_indices.append(i)
                black_positions.append(pos)

        values = np.zeros((len(positions),), dtype=np.float32)

        if white_positions:
            arr = self._white.infer_values(white_positions)
            for dst, val in zip(white_indices, arr, strict=False):
                values[dst] = float(val)

        if black_positions:
            arr = self._black.infer_values(black_positions)
            for dst, val in zip(black_indices, arr, strict=False):
                values[dst] = float(val)

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
    requested_games = int(max(0, games))
    rng = np.random.default_rng(seed)
    tracker = _ScoreTracker(requested_games)
    pgn_target = _prepare_pgn_directory(pgn_dir)
    label = label or "arena"
    start_time = time.time()

    for game_index in range(requested_games):
        candidate_white = (game_index % 2) == 0
        white_eval = candidate_eval if candidate_white else baseline_eval
        black_eval = baseline_eval if candidate_white else candidate_eval
        evaluator = _ColourRoutedEvaluator(white_eval, black_eval)

        start_fen = start_fen_fn(rng) if start_fen_fn is not None else DEFAULT_START_FEN
        record_moves = pgn_target is not None
        try:
            result, moves_san = _play_single_game(
                evaluator,
                rng,
                start_fen,
                record_moves=record_moves,
            )
        except Exception as exc:
            result, moves_san = ccore.DRAW, []
            logging.getLogger("hybridchess.arena").exception("Arena game stopped with error: %s", exc)

        tracker.record(candidate_white, result)

        if record_moves and pgn_target is not None:
            _save_pgn(
                pgn_target,
                label,
                game_index,
                start_fen,
                moves_san,
                _result_to_str(result),
                white_name="Candidate" if candidate_white else "Baseline",
                black_name="Baseline" if candidate_white else "Candidate",
            )

    elapsed = time.time() - start_time
    return tracker.to_result(elapsed_s=elapsed)


def _play_single_game(
    evaluator: _ColourRoutedEvaluator,
    rng: np.random.Generator,
    start_fen: str,
    *,
    record_moves: bool,
) -> tuple[ccore.Result, list[str]]:
    position = ccore.Position()
    try:
        position.from_fen(start_fen)
    except Exception:
        position.from_fen(DEFAULT_START_FEN)

    mcts = _build_mcts(rng)
    moves_played = 0
    max_plies = int(max(1, C.ARENA.max_game_plies))

    resign_enabled = bool(C.ARENA.resign_enable)
    resign_threshold = float(C.RESIGN.value_threshold)
    resign_consecutive = int(max(1, C.RESIGN.consecutive_required))
    resign_streak = 0

    moves_san: list[str] = []

    while moves_played < max_plies:
        result = position.result()
        if result != ccore.ONGOING:
            return result, moves_san

        legal_moves = list(position.legal_moves())
        if not legal_moves:
            return position.result(), moves_san

        visit_counts = mcts.search_batched_legal(
            position,
            evaluator.infer_positions_legal,
            int(max(1, C.EVAL.batch_size_max)),
        )
        counts = np.asarray(visit_counts, dtype=np.float64)
        if counts.shape[0] != len(legal_moves):
            return ccore.DRAW, moves_san

        temperature = (
            C.ARENA.temperature if moves_played < C.ARENA.temperature_moves else C.SELFPLAY.deterministic_temp_eps
        )
        move_index = select_visit_count_move(counts, temperature, rng)
        move = legal_moves[move_index]

        if record_moves:
            moves_san.append(_move_to_san(position, move, str(move), legal_moves))

        value = float(evaluator.infer_values([position])[0])
        stm_is_white = getattr(position, "turn", ccore.WHITE) == ccore.WHITE
        value_from_player = value if stm_is_white else -value
        if resign_enabled:
            if value_from_player <= resign_threshold:
                resign_streak += 1
                if resign_streak >= resign_consecutive:
                    return (ccore.BLACK_WIN if stm_is_white else ccore.WHITE_WIN), moves_san
            else:
                resign_streak = 0

        position.make_move(move)
        try:
            mcts.advance_root(position, move)
        except Exception:
            mcts = _build_mcts(rng)

        moves_played += 1

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


def _square_file(square: int) -> int:
    return int(square) & 7


def _square_rank(square: int) -> int:
    return int(square) >> 3


def _square_name(square: int) -> str:
    return f"{_FILE_NAMES[_square_file(square)]}{_square_rank(square) + 1}"


def _piece_on(position: ccore.Position, square: int) -> tuple[ccore.Color, int] | None:
    mask = 1 << int(square)
    for idx, pair in enumerate(position.pieces):
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
    position: ccore.Position,
    move: ccore.Move,
    move_str: str,
    legal_moves: Sequence[ccore.Move],
) -> str:
    from_sq = int(getattr(move, "from_square", 0))
    to_sq = int(getattr(move, "to_square", 0))
    info = _piece_on(position, from_sq)
    if info is None:
        return move_str
    mover_color, piece_type = info

    capture = False
    target = _piece_on(position, to_sq)
    if target is not None:
        capture = target[0] != mover_color
    elif piece_type == 0:
        ep_sq = int(getattr(position, "ep_square", -1))
        capture = ep_sq >= 0 and ep_sq == to_sq

    if piece_type == 5 and abs(_square_file(to_sq) - _square_file(from_sq)) == 2:
        return "O-O" if _square_file(to_sq) > _square_file(from_sq) else "O-O-O"

    if piece_type == 0:
        san = (_FILE_NAMES[_square_file(from_sq)] + "x") if capture else ""
        san += _square_name(to_sq)
        if len(move_str) >= 5:
            san += "=" + _PROMO_SAN.get(move_str[-1].lower(), move_str[-1].upper())
    else:
        san = _PIECE_SAN.get(piece_type, "")
        ambiguous: list[int] = []
        for other in legal_moves:
            is_same = other is move
            if not is_same:
                with contextlib.suppress(Exception):
                    is_same = bool(other == move)
            if is_same:
                continue
            if int(getattr(other, "to_square", -1)) != to_sq:
                continue
            details = _piece_on(position, int(getattr(other, "from_square", -1)))
            if details and details[0] == mover_color and details[1] == piece_type:
                ambiguous.append(int(getattr(other, "from_square", -1)))
        if ambiguous:
            same_file = any(_square_file(sq) == _square_file(from_sq) for sq in ambiguous)
            same_rank = any(_square_rank(sq) == _square_rank(from_sq) for sq in ambiguous)
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

    try:
        next_position = ccore.Position(position)
        next_position.make_move(move)
        result = next_position.result()

        if (mover_color == ccore.Color.WHITE and result == ccore.WHITE_WIN) or (
            mover_color == ccore.Color.BLACK and result == ccore.BLACK_WIN
        ):
            san += "#"
        elif next_position.in_check():
            san += "+"
    except Exception as exc:
        logging.getLogger("hybridchess.arena").debug("Failed to calculate SAN suffix: %s", exc)
    return san


def _prepare_pgn_directory(path: Path | None) -> Path | None:
    if path is None:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    date_str = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    headers: list[tuple[str, str]] = [
        ("Event", PGN_EVENT),
        ("Site", "Local"),
        ("Date", date_str),
        ("Round", str(game_index + 1)),
        ("White", white_name),
        ("Black", black_name),
        ("Result", result_str),
    ]
    if start_fen != DEFAULT_START_FEN:
        headers.extend([("SetUp", "1"), ("FEN", start_fen)])

    with path.open("w", encoding="utf-8") as handle:
        for key, value in headers:
            handle.write(f'[{key} "{value}"]\n')
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
