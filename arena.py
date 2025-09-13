from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Any

import chesscore as _ccore
import numpy as _np
import torch

import config as C
from inference import BatchedEvaluator as _BatchedEval


def arena_match(
    challenger: torch.nn.Module,
    incumbent: torch.nn.Module,
    *,
    device: torch.device,
    eval_cache_cap: int,
) -> tuple[float, int, int, int]:
    wins = draws = losses = 0

    with (
        _BatchedEval(device) as challenger_eval,
        _BatchedEval(device) as incumbent_eval,
    ):
        challenger_eval.refresh_from(challenger)
        incumbent_eval.refresh_from(incumbent)
        challenger_eval.cache_capacity = eval_cache_cap
        incumbent_eval.cache_capacity = eval_cache_cap

        def play(
            evaluator_white: _BatchedEval,
            evaluator_black: _BatchedEval,
            start_fen: str,
        ) -> tuple[int, list[str]]:
            position = _ccore.Position()
            position.from_fen(start_fen)
            mcts_white = _ccore.MCTS(
                C.ARENA.MCTS_EVAL_SIMULATIONS,
                C.MCTS.C_PUCT,
                C.MCTS.DIRICHLET_ALPHA,
                0.0 if C.ARENA.DETERMINISTIC else float(C.ARENA.DIRICHLET_WEIGHT),
            )
            mcts_white.set_c_puct_params(C.MCTS.C_PUCT_BASE, C.MCTS.C_PUCT_INIT)
            mcts_white.set_fpu_reduction(C.MCTS.FPU_REDUCTION)
            mcts_black = _ccore.MCTS(
                C.ARENA.MCTS_EVAL_SIMULATIONS,
                C.MCTS.C_PUCT,
                C.MCTS.DIRICHLET_ALPHA,
                0.0 if C.ARENA.DETERMINISTIC else float(C.ARENA.DIRICHLET_WEIGHT),
            )
            mcts_black.set_c_puct_params(C.MCTS.C_PUCT_BASE, C.MCTS.C_PUCT_INIT)
            mcts_black.set_fpu_reduction(C.MCTS.FPU_REDUCTION)
            ply = 0
            moves_uci: list[str] = []
            while position.result() == _ccore.ONGOING and ply < C.SELFPLAY.GAME_MAX_PLIES:
                visits = (
                    mcts_white.search_batched(
                        position,
                        evaluator_white.infer_positions,
                        C.EVAL.BATCH_SIZE_MAX,
                    )
                    if ply % 2 == 0
                    else mcts_black.search_batched(
                        position,
                        evaluator_black.infer_positions,
                        C.EVAL.BATCH_SIZE_MAX,
                    )
                )
                if visits is None:
                    break
                visit_counts = _np.asarray(visits, dtype=_np.float64)
                if visit_counts.size == 0:
                    break
                legal_moves = position.legal_moves()
                if C.ARENA.DETERMINISTIC:
                    idx = int(_np.argmax(visit_counts))
                elif ply < C.ARENA.TEMP_MOVES:
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
            r = position.result()
            return (
                1 if r == _ccore.WHITE_WIN else (-1 if r == _ccore.BLACK_WIN else 0),
                moves_uci,
            )

        pair_count = max(1, C.ARENA.GAMES_PER_EVAL // C.ARENA.PAIRING_FACTOR)
        pgn_candidates: list[dict[str, object]] = []
        for _ in range(pair_count):
            start_fen = _ccore.Position().to_fen()
            r1, mv1 = play(challenger_eval, incumbent_eval, start_fen)
            r2_raw, mv2 = play(incumbent_eval, challenger_eval, start_fen)
            r2 = -r2_raw
            pgn_candidates.append(
                {
                    "fen": start_fen,
                    "white": "Challenger-EMA",
                    "black": "Incumbent-Best",
                    "result": r1,
                    "moves": mv1,
                }
            )
            pgn_candidates.append(
                {
                    "fen": start_fen,
                    "white": "Incumbent-Best",
                    "black": "Challenger-EMA",
                    "result": r2_raw,
                    "moves": mv2,
                }
            )
            for outcome in (r1, r2):
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

            @dataclass
            class _PGNCandidate:
                fen: str
                white: str
                black: str
                result: int
                moves: list[str]

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

            def write_game(g: _PGNCandidate, name: str) -> None:
                nonlocal saved
                if saved >= int(C.LOG.ARENA_SAVE_PGN_SAMPLES_PER_ROUND):
                    return
                path = f"{round_tag}_{name}_{saved + 1}.pgn"
                with open(path, "w", encoding="utf-8") as f:
                    f.write('[Event "HybridChess Arena"]\n')
                    f.write('[Site "local"]\n')
                    f.write(f'[Date "{iso_date}"]\n')
                    f.write(f'[Round "{round_tag}"]\n')
                    f.write(f'[White "{g.white}"]\n')
                    f.write(f'[Black "{g.black}"]\n')
                    f.write(f'[Result "{res_str(int(g.result))}"]\n')
                    f.write(f'[FEN "{g.fen}"]\n[SetUp "1"]\n')
                    out: list[str] = []
                    move_no = 1
                    for i, m in enumerate(g.moves):
                        if i % 2 == 0:
                            out.append(f"{move_no}. {m}")
                            move_no += 1
                        else:
                            out.append(m)
                    f.write(" ".join(out) + f" {res_str(int(g.result))}\n")
                saved += 1

            for g in promo:
                write_game(g, "promo")
            if saved < int(C.LOG.ARENA_SAVE_PGN_SAMPLES_PER_ROUND):
                for g in pgn_candidates_typed:
                    if C.LOG.ARENA_SAVE_PGN_ON_PROMOTION and g in promo:
                        continue
                    write_game(g, "sample")
    except Exception:
        pass

    return score, wins, draws, losses


class EloGater:
    def __init__(
        self,
        *,
        z: float = C.ARENA.GATE_Z_LATE,
        min_games: int = C.ARENA.GATE_MIN_GAMES,
        draw_w: float = C.ARENA.DRAW_SCORE,
        baseline_p: float = C.ARENA.GATE_BASELINE_P,
        decisive_secondary: bool = C.ARENA.GATE_DECISIVE_SECONDARY,
        min_decisive: int = C.ARENA.GATE_MIN_DECISIVES,
    ) -> None:
        self.z = float(z)
        self.min_games = int(min_games)
        self.draw_w = float(draw_w)
        self.baseline_p = float(baseline_p)
        self.decisive_secondary = bool(decisive_secondary)
        self.min_decisive = int(min_decisive)
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
            return "undecided", {"n": float(n)}
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
        if lb > self.baseline_p:
            return "accept", {
                "n": float(n),
                "p": p,
                "lb": lb,
                "elo": elo,
                "se_elo": se_elo,
            }
        if ub < self.baseline_p:
            return "reject", {
                "n": float(n),
                "p": p,
                "ub": ub,
                "elo": elo,
                "se_elo": se_elo,
            }
        decisives = self.w + self.losses
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
                    "n": float(n),
                    "p": p,
                    "lb": lb,
                    "elo": elo_dec,
                    "se_elo": se_elo_dec,
                }
        return "undecided", {
            "n": float(n),
            "p": p,
            "lb": lb,
            "ub": ub,
            "elo": elo,
            "se_elo": se_elo,
        }
