from __future__ import annotations

import datetime as _dt
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import chesscore as _ccore
import config as C
import numpy as _np
import torch
from inference import BatchedEvaluator as _BatchedEval

_DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_FILE_NAMES = "abcdefgh"
_PIECE_SAN = {0: "", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K"}
_PROMO_SAN = {"q": "Q", "r": "R", "b": "B", "n": "N"}
_MATERIAL_WEIGHTS = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)


def _square_file(square: int) -> int:
    return int(square) & 7


def _square_rank(square: int) -> int:
    return int(square) >> 3


def _square_name(square: int) -> str:
    return f"{_FILE_NAMES[_square_file(square)]}{_square_rank(square) + 1}"


def _piece_on(pos: _ccore.Position, square: int) -> tuple[_ccore.Color, int] | None:
    mask = 1 << int(square)
    for idx, pair in enumerate(pos.pieces):
        try:
            white_bb, black_bb = int(pair[0]), int(pair[1])
        except Exception:
            continue
        if white_bb & mask:
            return _ccore.Color.WHITE, idx
        if black_bb & mask:
            return _ccore.Color.BLACK, idx
    return None


def _match_move(moves: Iterable[_ccore.Move], uci: str) -> _ccore.Move | None:
    for mv in moves:
        try:
            if str(mv) == uci:
                return mv
        except Exception:
            continue
    return None


def _move_to_san(
    pos: _ccore.Position,
    move: _ccore.Move,
    move_str: str,
    legal_moves: list[_ccore.Move],
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
            return "O-O" if _square_file(to_sq) > _square_file(from_sq) else "O-O-O"

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
                san += _FILE_NAMES[_square_file(from_sq)] + str(_square_rank(from_sq) + 1)
            elif same_file:
                san += str(_square_rank(from_sq) + 1)
            elif same_rank:
                san += _FILE_NAMES[_square_file(from_sq)]
            else:
                san += _FILE_NAMES[_square_file(from_sq)]
        if capture:
            san += "x"
        san += _square_name(to_sq)
    next_pos = _ccore.Position(pos)
    try:
        next_pos.make_move(move)
    except Exception:
        return san
    result = next_pos.result()
    try:
        if (mover_color == _ccore.Color.WHITE and result == _ccore.WHITE_WIN) or (
            mover_color == _ccore.Color.BLACK and result == _ccore.BLACK_WIN
        ):
            san += "#"
    except Exception:
        pass
    return san


def _moves_to_san(start_fen: str, moves: list[str]) -> list[str]:
    position = _ccore.Position()
    try:
        if start_fen and start_fen != _DEFAULT_START_FEN:
            position.from_fen(start_fen)
        else:
            position.from_fen(_DEFAULT_START_FEN)
    except Exception:
        position.from_fen(_DEFAULT_START_FEN)
    san_moves: list[str] = []
    for idx, mv_str in enumerate(moves):
        legal_moves = list(position.legal_moves())
        move_obj = _match_move(legal_moves, mv_str)
        if move_obj is None:
            san_moves.append(mv_str)
            san_moves.extend(moves[idx + 1 :])
            return san_moves
        san = _move_to_san(position, move_obj, mv_str, legal_moves)
        san_moves.append(san)
        try:
            position.make_move(move_obj)
        except Exception:
            san_moves.extend(moves[idx + 1 :])
            return san_moves
    return san_moves


def _material_balance(pos: _ccore.Position) -> float:
    balance = 0.0
    try:
        pieces = pos.pieces
    except Exception:
        return balance
    for idx, weight in enumerate(_MATERIAL_WEIGHTS):
        if weight == 0.0:
            continue
        try:
            white_bb = int(pieces[idx][0])
            black_bb = int(pieces[idx][1])
        except Exception:
            continue
        balance += weight * (white_bb.bit_count() - black_bb.bit_count())
    return balance


def _position_hash(pos: _ccore.Position) -> int | None:
    key_attr = getattr(pos, "hash", None)
    try:
        raw = key_attr() if callable(key_attr) else key_attr
        if raw is None:
            return None
        if isinstance(raw, (int, _np.integer)):
            return int(raw)
        if isinstance(raw, str):
            return int(raw)
    except Exception:
        return None
    return None



def arena_match(
    challenger: torch.nn.Module,
    incumbent: torch.nn.Module,
    *,
    device: torch.device,
    eval_cache_cap: int,
) -> tuple[float, int, int, int, dict[str, Any]]:
    wins = draws = losses = 0

    with (
        _BatchedEval(device) as challenger_eval,
        _BatchedEval(device) as incumbent_eval,
    ):
        challenger_eval.refresh_from(challenger)
        incumbent_eval.refresh_from(incumbent)
        challenger_eval.cache_capacity = eval_cache_cap
        incumbent_eval.cache_capacity = eval_cache_cap

        reason_counter: Counter[str] = Counter()

        def play(
            evaluator_white: _BatchedEval,
            evaluator_black: _BatchedEval,
            start_fen: str,
        ) -> tuple[int, list[str], str]:
            position = _ccore.Position()
            position.from_fen(start_fen)
            dirichlet_weight = float(C.ARENA.DIRICHLET_WEIGHT)
            mcts_white = _ccore.MCTS(
                C.ARENA.MCTS_EVAL_SIMULATIONS,
                C.MCTS.C_PUCT,
                C.MCTS.DIRICHLET_ALPHA,
                dirichlet_weight,
            )
            mcts_white.set_c_puct_params(C.MCTS.C_PUCT_BASE, C.MCTS.C_PUCT_INIT)
            mcts_white.set_fpu_reduction(C.MCTS.FPU_REDUCTION)
            mcts_black = _ccore.MCTS(
                C.ARENA.MCTS_EVAL_SIMULATIONS,
                C.MCTS.C_PUCT,
                C.MCTS.DIRICHLET_ALPHA,
                dirichlet_weight,
            )
            mcts_black.set_c_puct_params(C.MCTS.C_PUCT_BASE, C.MCTS.C_PUCT_INIT)
            mcts_black.set_fpu_reduction(C.MCTS.FPU_REDUCTION)
            ply = 0
            moves_uci: list[str] = []
            max_plies = int(getattr(C.ARENA, "GAME_MAX_PLIES", C.SELFPLAY.GAME_MAX_PLIES))
            resign_enabled = bool(getattr(C.ARENA, "RESIGN_ENABLE", getattr(C.RESIGN, "ENABLED", False)))
            resign_threshold = float(
                getattr(C.ARENA, "RESIGN_VALUE_THRESHOLD", getattr(C.RESIGN, "VALUE_THRESHOLD", -0.35))
            )
            resign_consecutive = int(
                getattr(C.ARENA, "RESIGN_CONSECUTIVE", getattr(C.RESIGN, "CONSECUTIVE_PLIES", 3))
            )
            resign_playthrough = float(
                getattr(C.ARENA, "RESIGN_PLAYTHROUGH_FRACTION", getattr(C.RESIGN, "PLAYTHROUGH_FRACTION", 0.0))
            )
            resign_counts = {_ccore.WHITE: 0, _ccore.BLACK: 0}
            resign_rng = _np.random.default_rng()
            forced_result: int | None = None
            forced_reason: str | None = None
            seen_positions: dict[int, int] = {}
            while position.result() == _ccore.ONGOING and ply < max_plies:
                pos_snapshot = _ccore.Position(position)
                pos_key = _position_hash(position)
                if pos_key is not None:
                    seen_positions[pos_key] = seen_positions.get(pos_key, 0) + 1
                    if seen_positions[pos_key] >= 3:
                        forced_result = _ccore.DRAW
                        forced_reason = "threefold"
                        break
                halfmove_clock = int(getattr(position, "halfmove_clock", 0))
                if halfmove_clock >= 100:
                    forced_result = _ccore.DRAW
                    forced_reason = "fifty_move"
                    break
                evaluator_to_move = evaluator_white if pos_snapshot.turn == _ccore.WHITE else evaluator_black
                if resign_enabled:
                    value_estimate: float | None
                    try:
                        value_estimate = float(evaluator_to_move.infer_values([pos_snapshot])[0])
                    except Exception:
                        value_estimate = None
                    side = pos_snapshot.turn
                    if value_estimate is not None and value_estimate <= resign_threshold:
                        resign_counts[side] = resign_counts.get(side, 0) + 1
                        if resign_counts[side] >= resign_consecutive:
                            if resign_rng.random() < resign_playthrough:
                                resign_counts[side] = 0
                            else:
                                forced_result = _ccore.BLACK_WIN if side == _ccore.WHITE else _ccore.WHITE_WIN
                                forced_reason = "resign"
                                break
                    else:
                        resign_counts[side] = 0

                visits = (
                    mcts_white.search_batched_legal(
                        position,
                        evaluator_white.infer_positions_legal,
                        C.EVAL.BATCH_SIZE_MAX,
                    )
                    if ply % 2 == 0
                    else mcts_black.search_batched_legal(
                        position,
                        evaluator_black.infer_positions_legal,
                        C.EVAL.BATCH_SIZE_MAX,
                    )
                )
                if visits is None:
                    break
                visit_counts = _np.asarray(visits, dtype=_np.float64)
                if visit_counts.size == 0:
                    break
                legal_moves = position.legal_moves()
                if ply < C.ARENA.TEMP_MOVES:
                    temp = float(C.ARENA.TEMPERATURE)
                    if temp <= 0.0 + C.ARENA.OPENING_TEMPERATURE_EPS:
                        idx = int(_np.argmax(visit_counts))
                    else:
                        v_pos = _np.maximum(visit_counts, 0.0)
                        s0 = v_pos.sum()
                        if s0 <= 0:
                            idx = int(_np.argmax(visit_counts))
                        else:
                            probs = v_pos ** (1.0 / temp)
                            s = probs.sum()
                            idx = (
                                int(_np.argmax(visit_counts))
                                if s <= 0
                                else int(_np.random.choice(len(legal_moves), p=probs / s))
                            )
                else:
                    idx = int(_np.argmax(visit_counts))
                mv = legal_moves[idx]
                try:
                    mv_str = str(mv)
                    if mv_str:
                        moves_uci.append(mv_str)
                except Exception:
                    pass
                position.make_move(mv)

                try:
                    mcts_white.advance_root(position, mv)
                    mcts_black.advance_root(position, mv)
                except Exception:
                    pass
                ply += 1

            final_reason = "natural"
            if forced_result is not None:
                result_flag = forced_result
                final_reason = forced_reason or "forced"
            else:
                current_result = position.result()
                if current_result == _ccore.ONGOING:
                    result_flag, final_reason = _ccore.DRAW, "exhausted"
                else:
                    result_flag = current_result
                    final_reason = "natural"

            reason_counter[final_reason] += 1
            return (
                1 if result_flag == _ccore.WHITE_WIN else (-1 if result_flag == _ccore.BLACK_WIN else 0),
                moves_uci,
                final_reason,
            )

        pair_count = max(1, C.ARENA.GAMES_PER_EVAL // C.ARENA.PAIRING_FACTOR)
        pgn_candidates: list[dict[str, object]] = []
        rng = _np.random.default_rng()
        random_opening_cap = int(getattr(C.ARENA, "OPENING_RANDOM_PLIES_MAX", 0))
        for _ in range(pair_count):
            start_position = _ccore.Position()
            if random_opening_cap > 0:
                random_plies = int(rng.integers(0, random_opening_cap + 1))
                for _ in range(random_plies):
                    if start_position.result() != _ccore.ONGOING:
                        break
                    moves = start_position.legal_moves()
                    if not moves:
                        break
                    mv_idx = int(rng.integers(0, len(moves)))
                    start_position.make_move(moves[mv_idx])
            start_fen = start_position.to_fen()
            r1, mv1, reason1 = play(challenger_eval, incumbent_eval, start_fen)
            r2_raw, mv2, reason2 = play(incumbent_eval, challenger_eval, start_fen)
            r2 = -r2_raw
            pgn_candidates.append(
                {
                    "fen": start_fen,
                    "white": "Challenger-EMA",
                    "black": "Incumbent-Best",
                    "result": r1,
                    "moves": mv1,
                    "termination": reason1,
                }
            )
            pgn_candidates.append(
                {
                    "fen": start_fen,
                    "white": "Incumbent-Best",
                    "black": "Challenger-EMA",
                    "result": r2_raw,
                    "moves": mv2,
                    "termination": reason2,
                }
            )
            for outcome, _reason in ((r1, reason1), (r2, reason2)):
                if outcome > 0:
                    wins += 1
                elif outcome < 0:
                    losses += 1
                else:
                    draws += 1

    total_games = max(1, wins + draws + losses)
    score = (wins + C.ARENA.DRAW_SCORE * draws) / total_games

    try:
        if C.LOG.ARENA_SAVE_PGN_ENABLE and pgn_candidates:
            iso_date = _dt.datetime.now(_dt.UTC).strftime("%Y.%m.%d")
            round_tag = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
            try:
                import os as _os
                pgn_dir = str(C.LOG.ARENA_PGN_DIR)
                if pgn_dir:
                    _os.makedirs(pgn_dir, exist_ok=True)
            except Exception:
                pgn_dir = ""

            @dataclass
            class _PGNCandidate:
                fen: str
                white: str
                black: str
                result: int
                moves: list[str]
                termination: str

            def _as_int(x: Any) -> int:
                try:
                    return int(x)
                except Exception:
                    return 0

            def _as_str_list(x: Any) -> list[str]:
                try:
                    return [str(s) for s in list(x)]
                except Exception:
                    return []

            pgn_candidates_typed: list[_PGNCandidate] = []
            for _g in pgn_candidates:
                try:
                    pgn_candidates_typed.append(
                        _PGNCandidate(
                            fen=str(_g.get("fen", "")),
                            white=str(_g.get("white", "")),
                            black=str(_g.get("black", "")),
                            result=_as_int(_g.get("result", 0)),
                            moves=_as_str_list(_g.get("moves", []) or []),
                            termination=str(_g.get("termination", "unknown")),
                        )
                    )
                except Exception:
                    continue

            def res_str(r: int) -> str:
                return "1-0" if r > 0 else ("0-1" if r < 0 else "1/2-1/2")

            def has_promo(moves: list[str]) -> bool:
                return any((len(m) >= 5 and m[-1] in ("q", "r", "b", "n")) for m in moves)

            promo: list[_PGNCandidate] = (
                [g for g in pgn_candidates_typed if has_promo(g.moves)] if C.LOG.ARENA_SAVE_PGN_ON_PROMOTION else []
            )
            saved = 0

            def _important_reason(term: str) -> bool:
                lowered = term.lower()
                if any(keyword in lowered for keyword in ("resign", "mate", "natural", "stalemate")):
                    return True
                return lowered not in {"draw", "exhausted"}

            def _is_important_game(game: _PGNCandidate) -> bool:
                if int(game.result) != 0:
                    return True
                return _important_reason(str(game.termination))

            def write_game(g: _PGNCandidate, name: str) -> None:
                nonlocal saved
                if saved >= int(C.LOG.ARENA_SAVE_PGN_SAMPLES_PER_ROUND):
                    return
                path = f"{round_tag}_{name}_{saved + 1}.pgn"
                full_path = f"{pgn_dir}/{path}" if pgn_dir else path
                start_fen = g.fen or _DEFAULT_START_FEN
                san_moves = _moves_to_san(start_fen, g.moves)
                use_default_start = start_fen == _DEFAULT_START_FEN
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write('[Event "HybridChess Arena"]\n')
                    f.write('[Site "local"]\n')
                    f.write(f'[Date "{iso_date}"]\n')
                    f.write(f'[Round "{round_tag}"]\n')
                    f.write(f'[White "{g.white}"]\n')
                    f.write(f'[Black "{g.black}"]\n')
                    f.write(f'[Result "{res_str(int(g.result))}"]\n')
                    f.write(f'[Termination "{g.termination}"]\n')
                    if not use_default_start:
                        f.write(f'[FEN "{start_fen}"]\n[SetUp "1"]\n')
                    out: list[str] = []
                    move_no = 1
                    for i, m in enumerate(san_moves):
                        if i % 2 == 0:
                            out.append(f"{move_no}. {m}")
                            move_no += 1
                        else:
                            out.append(m)
                    f.write(" ".join(out) + f" {res_str(int(g.result))}\n")
                saved += 1

            selection: list[_PGNCandidate] = []
            if promo:
                for g in promo:
                    if g not in selection:
                        selection.append(g)
            important_games = [g for g in pgn_candidates_typed if _is_important_game(g)]
            for g in important_games:
                if g not in selection:
                    selection.append(g)
            if not selection and pgn_candidates_typed:
                selection.append(pgn_candidates_typed[0])
            for g in selection:
                tag = "promo" if (C.LOG.ARENA_SAVE_PGN_ON_PROMOTION and g in promo) else "important"
                write_game(g, tag)
    except Exception:
        pass

    details = {
        "games": float(wins + draws + losses),
        "reason_counts": dict(reason_counter),
    }
    return score, wins, draws, losses, details


class EloGater:
    def __init__(
        self,
        *,
        z: float = C.ARENA.GATE_Z_LATE,
        min_games: int = C.ARENA.GATE_MIN_GAMES,
        draw_w: float = C.ARENA.DRAW_SCORE,
        baseline_p: float = C.ARENA.GATE_BASELINE_P,
        baseline_margin: float = C.ARENA.GATE_BASELINE_MARGIN,
        decisive_secondary: bool = C.ARENA.GATE_DECISIVE_SECONDARY,
        min_decisive: int = C.ARENA.GATE_MIN_DECISIVES,
        force_decisive: bool = C.ARENA.GATE_FORCE_DECISIVE,
    ) -> None:
        self.z = float(z)
        self.min_games = int(min_games)
        self.draw_w = float(draw_w)
        self.baseline_p = float(baseline_p)
        self.baseline_margin = float(max(0.0, baseline_margin))
        self.decisive_secondary = bool(decisive_secondary)
        self.min_decisive = int(min_decisive)
        self.force_decisive = bool(force_decisive)
        self.reset()

    def reset(self) -> None:
        self.w = 0
        self.d = 0
        self.losses = 0

    def update(self, w: int, d: int, losses: int) -> None:
        self.w += int(w)
        self.d += int(d)
        self.losses += int(losses)

    def decision(self) -> tuple[str, dict[str, float]]:
        n = self.w + self.d + self.losses
        if n < self.min_games:
            return "undecided", {"n": float(n), "decisive": float(self.w + self.losses)}
        p = (self.w + self.draw_w * self.d) / max(1, n)
        w_frac = self.w / n
        d_frac = self.d / n
        var = max(1e-9, w_frac + (self.draw_w**2) * d_frac - (p * p))
        se = math.sqrt(var / n)
        lb = p - self.z * se
        ub = p + self.z * se
        eps = C.ARENA.GATE_EPS
        pc = min(1.0 - eps, max(eps, p))
        elo = 400.0 * math.log10(pc / (1.0 - pc))
        denom = max(eps, pc * (1.0 - pc))
        se_elo = (400.0 / math.log(10.0)) * se / denom
        decisives = self.w + self.losses
        metrics = {
            "n": float(n),
            "p": p,
            "lb": lb,
            "ub": ub,
            "elo": elo,
            "se_elo": se_elo,
            "decisive": float(decisives),
        }
        if self.force_decisive and decisives < self.min_decisive:
            return "undecided", metrics
        margin_target = self.baseline_p + self.baseline_margin
        if lb > margin_target and decisives >= self.min_decisive:
            return "accept", {
                **metrics,
                "margin_target": margin_target,
            }
        if ub < self.baseline_p:
            return "reject", {
                **metrics,
            }
        if self.decisive_secondary and decisives >= self.min_decisive:
            p_dec = self.w / max(1, decisives)
            se_dec = math.sqrt(max(1e-9, p_dec * (1.0 - p_dec) / max(1, decisives)))
            lb_dec = p_dec - self.z * se_dec
            if lb_dec > 0.5:
                pc_dec = min(1.0 - eps, max(eps, p_dec))
                elo_dec = 400.0 * math.log10(pc_dec / (1.0 - pc_dec))
                denom_dec = max(eps, pc_dec * (1.0 - pc_dec))
                se_elo_dec = (400.0 / math.log(10.0)) * se_dec / denom_dec
                return "accept", {
                    **metrics,
                    "elo": elo_dec,
                    "se_elo": se_elo_dec,
                }
        return "undecided", metrics
