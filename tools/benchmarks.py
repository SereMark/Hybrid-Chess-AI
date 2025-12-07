#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("benchmark")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _setup_paths() -> None:
    candidate_dirs = [
        REPO_ROOT / "build" / "python" / "Release",
        REPO_ROOT / "build" / "python" / "Debug",
        REPO_ROOT / "build" / "python",
        REPO_ROOT / "src" / "python",
    ]
    for path in candidate_dirs:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_setup_paths()


def import_module_safe(name: str, error_msg: str) -> ModuleType:
    try:
        return __import__(name)
    except ImportError as exc:
        raise SystemExit(error_msg) from exc


ccore = import_module_safe("chesscore", "Nem sikerült importálni a chesscore kiterjesztést.")
chess = import_module_safe("chess", "Hiányzó python-chess függőség.")

import config as C
from encoder import INPUT_PLANES, POLICY_SIZE
from inference import BatchedEvaluator
from network import ChessNet
from self_play import SelfPlayEngine
from train_loop import train_step
from utils import DEFAULT_START_FEN, select_visit_count_move, select_autocast_dtype

RUN_NAME_MAPPING = {
    "20251125-124632": "Referencia",
    "20251126-100221": "Mély Keresés",
    "20251126-205609": "Nagy Áteresztőképesség",
    "20251127-153749": "Nagy Entrópia",
    "20251128-101517": "Hatékonyság",
}


@dataclass(slots=True)
class Measurement:
    name: str
    samples: list[float]
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        mean = statistics.fmean(self.samples) if self.samples else 0.0
        stdev = statistics.pstdev(self.samples) if len(self.samples) > 1 else 0.0
        return {
            "name": self.name,
            "sample_count": len(self.samples),
            "mean_s": mean,
            "stdev_s": stdev,
            "min_s": min(self.samples) if self.samples else 0.0,
            "max_s": max(self.samples) if self.samples else 0.0,
            **self.extras,
        }


def benchmark_callable(
    name: str,
    fn: Callable[[], Any],
    *,
    warmup: int = 1,
    repeat: int = 5,
) -> Measurement:
    for _ in range(max(0, warmup)):
        fn()

    samples: list[float] = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return Measurement(name=name, samples=samples)


def save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        fieldnames = sorted(list(all_keys))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_measurements(measurements: Sequence[Measurement], *, metadata: dict[str, Any] | None = None) -> dict:
    summaries = []
    for measurement in measurements:
        summary = measurement.summary()
        summary.setdefault("sample_count", len(measurement.samples))
        summaries.append(summary)
    return {
        "metadata": metadata or {},
        "measurements": summaries,
    }


def render_markdown_table(measurements: Sequence[Measurement]) -> str:
    if not measurements:
        return "Nincs összegyűjtött mérés."

    rows = [m.summary() for m in measurements]
    extra_keys = sorted({
        k for row in rows for k in row.keys()
        if k not in {"name", "samples", "mean_s", "stdev_s", "min_s", "max_s", "sample_count"}
    })

    headers = [
        "Scenario", "Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)",
        *(key.replace("_", " ").title() for key in extra_keys)
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]

    for row in rows:
        base = [
            row["name"],
            f"{row['mean_s'] * 1000:.3f}",
            f"{row['stdev_s'] * 1000:.3f}",
            f"{row['min_s'] * 1000:.3f}",
            f"{row['max_s'] * 1000:.3f}",
        ]
        extras = [str(row.get(key, "")) for key in extra_keys]
        lines.append("| " + " | ".join(base + extras) + " |")
    return "\n".join(lines)


def write_reports(
    report: dict,
    *,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
    csv_path: Path | None = None,
) -> None:
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2))

    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        measurements = []
        for entry in report.get("measurements", []):
            extras = {k: v for k, v in entry.items() if k not in {"name", "mean_s", "stdev_s", "min_s", "max_s", "sample_count"}}
            measurements.append(Measurement(name=entry["name"], samples=[], extras=extras))
        markdown_path.write_text(render_markdown_table(measurements))

    if csv_path:
        rows = []
        for row in report.get("measurements", []):
            clean = {k: v for k, v in row.items() if k != "samples"}
            rows.append(clean)
        save_csv(csv_path, rows)


def load_model_from_checkpoint(checkpoint_path: Path, device_obj: Any) -> tuple[Any, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    state_dict = checkpoint["model"]

    num_filters = state_dict["conv_in.weight"].shape[0]
    num_blocks = sum(1 for k in state_dict if k.startswith("residual_stack.") and k.endswith(".conv1.weight"))

    orig_config = C.MODEL
    C.MODEL = replace(C.MODEL, blocks=num_blocks, channels=num_filters)

    evaluator = BatchedEvaluator(device_obj)
    evaluator.eval_model.load_state_dict(state_dict)

    C.MODEL = orig_config
    return evaluator, checkpoint


PIECE_VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 20000}


class RandomPlayer:
    def select_move(self, pos: Any, moves: list[Any]) -> Any:
        return random.choice(moves)


class GreedyPlayer:
    def select_move(self, pos: Any, moves: list[Any]) -> Any:
        best_score = -99999.0
        best_moves = [moves[0]]

        for move in moves:
            score = self._evaluate_move(pos, move)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)

    def _evaluate_move(self, pos: Any, move: Any) -> float:
        next_pos = ccore.Position(pos)
        next_pos.make_move(move)

        board_str = next_pos.to_fen().split()[0]
        white_mat = sum(PIECE_VALUES.get(self._char_to_piece(c.lower()), 0) for c in board_str if c.isupper())
        black_mat = sum(PIECE_VALUES.get(self._char_to_piece(c.lower()), 0) for c in board_str if c.islower())

        diff = white_mat - black_mat
        return float(diff if pos.turn == ccore.WHITE else -diff)

    def _char_to_piece(self, char: str) -> int:
        return {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6}.get(char, 1)


def detect_devices(explicit: Sequence[str] | None = None) -> list[str]:
    if explicit:
        return list(dict.fromkeys(explicit))
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def cmd_inference(args: argparse.Namespace) -> None:
    devices = detect_devices(args.devices)
    logger.info(f"Inferencia benchmark futtatása ezeken az eszközökön: {devices}")

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch.manual_seed(2025)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2025)

    measurements: list[Measurement] = []

    for device_name in devices:
        try:
            device = torch.device(device_name)
        except Exception:
            continue

        for dtype_str in args.dtypes:
            dtype = dtype_map.get(dtype_str)
            if dtype is None:
                continue

            try:
                model = ChessNet().to(device=device, dtype=dtype)
                model.eval()
                model.requires_grad_(False)
            except Exception:
                continue

            for batch_size in args.batch_sizes:
                try:
                    batch = torch.randn(batch_size, INPUT_PLANES, 8, 8, device=device, dtype=dtype)
                except Exception:
                    continue

                @torch.no_grad()
                def _forward() -> None:
                    policy, value = model(batch)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elif device.type == "mps":
                        torch.mps.synchronize()
                    _ = float(policy.sum().item() + value.sum().item())

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.empty_cache()

                meas = benchmark_callable(
                    f"{device_name}-{dtype_str}-b{batch_size}",
                    _forward,
                    warmup=args.warmup,
                    repeat=args.repeats
                )

                mean_sec = statistics.fmean(meas.samples)
                meas.extras.update({
                    "device": device_name,
                    "dtype": dtype_str,
                    "batch_size": batch_size,
                    "throughput_pos_per_sec": (batch_size / mean_sec) if mean_sec > 0 else 0.0
                })

                if device.type == "cuda":
                    meas.extras["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)

                measurements.append(meas)

    print(render_markdown_table(measurements))
    write_reports(summarize_measurements(measurements, metadata=vars(args)), csv_path=args.output_csv)
    logger.info(f"Inferencia riport elmentve ide: {args.output_csv}")


def bench_selfplay_throughput(device: torch.device, worker_counts: list[int], games: int, repeats: int, warmup: int) -> list[Measurement]:
    C.MCTS = replace(C.MCTS, train_simulations=32, train_simulations_min=16)
    C.SELFPLAY = replace(C.SELFPLAY, game_max_plies=40, temperature_moves=4)

    evaluator = BatchedEvaluator(device)
    engine = SelfPlayEngine(evaluator, seed=123)
    engine._curriculum_prob = 0.0
    engine.opening_random_moves = 0

    results = []
    try:
        for workers in worker_counts:
            engine.set_num_workers(workers)
            meas = benchmark_callable(
                f"selfplay-w{workers}",
                lambda: engine.play_games(games),
                warmup=warmup,
                repeat=repeats
            )
            mean_time = statistics.fmean(meas.samples)
            meas.extras.update({
                "type": "selfplay",
                "workers": workers,
                "games_per_batch": games,
                "games_per_sec": (games / mean_time) if mean_time > 0 else 0
            })
            results.append(meas)
    finally:
        engine.close()
        evaluator.close()
    return results


def bench_training_step(device: torch.device, batch_sizes: list[int], repeats: int, warmup: int) -> list[Measurement]:
    results = []
    for bs in batch_sizes:
        rng = np.random.default_rng(42)
        batch = (
            [rng.integers(0, 256, (INPUT_PLANES, 8, 8), dtype=np.uint8) for _ in range(bs)],
            [rng.choice(POLICY_SIZE, 30).astype(np.int32) for _ in range(bs)],
            [rng.integers(1, 10, 30).astype(np.uint16) for _ in range(bs)],
            [np.array(0, dtype=np.int8) for _ in range(bs)]
        )

        model = ChessNet().to(device)
        trainer = SimpleNamespace(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            device=device,
            train_batch_size=bs,
            scheduler=SimpleNamespace(step=lambda: None),
            iteration=0,
            ema=None
        )

        use_amp = device.type == "cuda"
        trainer._amp_enabled = use_amp
        trainer._autocast_dtype = select_autocast_dtype(device)
        trainer.scaler = torch.amp.GradScaler(enabled=use_amp)

        def _step() -> None:
            train_step(trainer, batch)
            if device.type == "cuda":
                torch.cuda.synchronize()

        meas = benchmark_callable(
            f"train-b{bs}-amp{'On' if use_amp else 'Off'}",
            _step,
            warmup=warmup,
            repeat=repeats
        )
        mean_time = statistics.fmean(meas.samples)
        meas.extras.update({
            "type": "training",
            "batch_size": bs,
            "amp": use_amp,
            "steps_per_sec": (1.0 / mean_time) if mean_time > 0 else 0
        })
        results.append(meas)
    return results


def cmd_system(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    measurements = []

    if args.mode in ("all", "selfplay"):
        logger.info("Önjáték áteresztőképességének mérése...")
        measurements.extend(bench_selfplay_throughput(device, [1, 2, 4, 6], 8, 3, 1))

    if args.mode in ("all", "training"):
        logger.info("Tanítási lépés késleltetésének mérése...")
        measurements.extend(bench_training_step(device, [64, 128, 256], 10, 3))

    print(render_markdown_table(measurements))
    write_reports(summarize_measurements(measurements, metadata=vars(args)), csv_path=args.output_csv)
    logger.info(f"Rendszer benchmark riport elmentve ide: {args.output_csv}")


class Player:
    def get_name(self) -> str:
        raise NotImplementedError

    def select_move(self, pos: ccore.Position, mcts_sims: int, rng: np.random.Generator) -> ccore.Move:
        raise NotImplementedError

    def close(self):
        pass


class ModelPlayer(Player):
    def __init__(self, name: str, checkpoint_path: Path, device: str):
        self.name = name
        self.device = torch.device(device)
        self.evaluator, _ = load_model_from_checkpoint(checkpoint_path, self.device)

    def get_name(self) -> str:
        return self.name

    def get_evaluator(self) -> BatchedEvaluator:
        return self.evaluator


class BaselinePlayer(Player):
    def __init__(self, name: str, strategy: Any):
        self.name = name
        self.strategy = strategy

    def get_name(self) -> str:
        return self.name

    def select_move(self, pos: ccore.Position, mcts_sims: int, rng: np.random.Generator) -> ccore.Move:
        legal_moves = list(pos.legal_moves())
        return self.strategy.select_move(pos, legal_moves)


@dataclass
class GameRecord:
    p1: str
    p2: str
    winner: str
    reason: str
    moves: int


def play_game(p1: Any, p2: Any, p1_is_white: bool, mcts_sims: int, rng: np.random.Generator) -> GameRecord:
    pos = ccore.Position()
    pos.from_fen(DEFAULT_START_FEN)

    mcts_white = ccore.MCTS(mcts_sims, float(C.MCTS.c_puct), 0.0, 0.0)
    mcts_black = ccore.MCTS(mcts_sims, float(C.MCTS.c_puct), 0.0, 0.0)

    seed = int(rng.integers(0, 1000000))
    mcts_white.seed(seed)
    mcts_black.seed(seed)

    white_p = p1 if p1_is_white else p2
    black_p = p2 if p1_is_white else p1

    for move_count in range(150):
        res = pos.result()
        if res != ccore.Result.ONGOING:
            if res == ccore.Result.DRAW:
                return GameRecord(p1.name, p2.name, "draw", "draw", move_count)
            winner = white_p.name if res == ccore.Result.WHITE_WIN else black_p.name
            return GameRecord(p1.name, p2.name, "p1" if winner == p1.name else "p2", "checkmate", move_count)

        legal = list(pos.legal_moves())
        if not legal:
            return GameRecord(p1.name, p2.name, "draw", "stalemate", move_count)

        current_p = white_p if pos.turn == ccore.Color.WHITE else black_p

        if isinstance(current_p, ModelPlayer):
            mcts = mcts_white if pos.turn == ccore.Color.WHITE else mcts_black
            visit_counts = mcts.search_batched_legal(
                pos, current_p.evaluator.infer_positions_legal, C.EVAL.batch_size_max
            )
            move_idx = select_visit_count_move(np.asarray(visit_counts, dtype=np.float64), 0.0, rng)
            move = legal[move_idx]
        else:
            move = current_p.select_move(pos, mcts_sims, rng)

        pos.make_move(move)
        mcts_white.advance_root(pos, move)
        mcts_black.advance_root(pos, move)

    return GameRecord(p1.name, p2.name, "draw", "move_limit", 150)


def calculate_elo(matches: List[GameRecord], initial: float = 1200.0, k: float = 1.0) -> Dict[str, float]:
    players = {m.p1 for m in matches} | {m.p2 for m in matches}
    ratings = {p: initial for p in players}

    for _ in range(1000):
        new_ratings = ratings.copy()
        deltas = {p: 0.0 for p in players}

        for m in matches:
            r1, r2 = ratings[m.p1], ratings[m.p2]
            e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
            e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
            s1 = 1.0 if m.winner == "p1" else (0.0 if m.winner == "p2" else 0.5)
            s2 = 1.0 - s1

            deltas[m.p1] += k * (s1 - e1)
            deltas[m.p2] += k * (s2 - e2)

        for p in players:
            new_ratings[p] += deltas[p]

        mean = sum(new_ratings.values()) / len(new_ratings)
        for p in new_ratings:
            new_ratings[p] += (initial - mean)

        if sum(abs(new_ratings[p] - ratings[p]) for p in players) < 0.01:
            break
        ratings = new_ratings
    return ratings


def cmd_ranking(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    players = []
    if args.include_random:
        players.append(BaselinePlayer("Random", RandomPlayer()))
    if args.include_greedy:
        players.append(BaselinePlayer("Greedy", GreedyPlayer()))

    for d in sorted([d for d in args.runs_dir.iterdir() if d.is_dir()]):
        ckpt = d / "checkpoints" / "best.pt"
        if ckpt.exists():
            logger.info(f"Modell betöltése innen: {d.name}...")
            name = RUN_NAME_MAPPING.get(d.name, d.name)
            players.append(ModelPlayer(name, ckpt, args.device))

    if len(players) < 2:
        logger.error("Legalább 2 játékos szükséges.")
        return

    records = []
    rng = np.random.default_rng(2025)
    pairs = list(itertools.combinations(players, 2))

    logger.info(f"Bajnokság: {len(players)} játékos, {len(pairs)} párosítás.")
    total_pairs = len(pairs)
    for idx, (p1, p2) in enumerate(pairs, start=1):
        logger.info(f"Párosítás lejátszása {idx}/{total_pairs}: {p1.get_name()} vs {p2.get_name()}")
        for i in range(args.games_per_pair):
            records.append(play_game(p1, p2, i % 2 == 0, args.mcts_sims, rng))

    elos = calculate_elo(records)
    ranked = sorted(players, key=lambda p: elos[p.name], reverse=True)

    report = {
        "metadata": {"mcts_sims": args.mcts_sims, "device": str(args.device)},
        "elo": elos,
        "matches": [asdict(m) for m in records]
    }
    with (args.output_dir / "ranking_results.json").open("w") as f:
        json.dump(report, f, indent=2)

    print("\nBajnokság állása:")
    for p in ranked:
        print(f"  {p.name:<20}: {elos[p.name]:.0f}")


def generate_random_fens(count: int, max_random_plies: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    fens: list[str] = []
    for _ in range(count):
        board = chess.Board()
        for _ in range(rng.randint(0, max_random_plies)):
            if not list(board.legal_moves) or board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        fens.append(board.fen())
    return fens


def benchmark_chesscore(fens: Sequence[str], loops: int) -> tuple[float, int]:
    pos = ccore.Position()
    total_moves = 0
    start = time.perf_counter()
    for _ in range(loops):
        for fen in fens:
            pos.from_fen(fen)
            moves = pos.legal_moves()
            total_moves += len(moves)
    elapsed = time.perf_counter() - start
    return elapsed, total_moves


def benchmark_python_chess(fens: Sequence[str], loops: int) -> tuple[float, int]:
    total_moves = 0
    start = time.perf_counter()
    for _ in range(loops):
        for fen in fens:
            board = chess.Board(fen)
            moves = list(board.legal_moves)
            total_moves += len(moves)
    elapsed = time.perf_counter() - start
    return elapsed, total_moves


def cmd_chess_bench(args: argparse.Namespace) -> None:
    logger.info("Véletlen állások generálása...")
    fens = generate_random_fens(args.positions, args.max_plies, args.seed)
    logger.info(f"{len(fens)} állás generálva.")

    positions_evaluated = len(fens) * args.loops

    logger.info("ChessCore (C++) benchmark futtatása...")
    cc_time, cc_moves = benchmark_chesscore(fens, args.loops)

    logger.info("Python-Chess benchmark futtatása...")
    py_time, py_moves = benchmark_python_chess(fens, args.loops)

    results = [
        {"impl": "chesscore", "time_s": cc_time, "pos_per_sec": positions_evaluated/cc_time if cc_time > 0 else 0},
        {"impl": "python-chess", "time_s": py_time, "pos_per_sec": positions_evaluated/py_time if py_time > 0 else 0}
    ]
    save_csv(args.output_dir / "chess_cpp.csv", results)

    print("\nEredmények:")
    print(f"  ChessCore:    {cc_time:.3f}s ({positions_evaluated/cc_time:,.0f} állás/mp)")
    print(f"  Python-Chess: {py_time:.3f}s ({positions_evaluated/py_time:,.0f} állás/mp)")
    if cc_time > 0:
        print(f"  Gyorsítás:   {py_time/cc_time:.2f}x")


def uniform_evaluator_batched(positions: Sequence[Any], *args: Any) -> tuple[list[list[float]], list[float]]:
    return [([1.0 / len(list(p.legal_moves()))] * len(list(p.legal_moves()))) if list(p.legal_moves()) else [] for p in positions], [0.0] * len(positions)


def benchmark_cpp_mcts(fens: Sequence[str], simulations: int, max_batch: int) -> tuple[float, int]:
    engine = ccore.MCTS(simulations)
    start = time.perf_counter()
    total_nodes = 0
    for fen in fens:
        engine.reset_tree()
        pos = ccore.Position()
        pos.from_fen(fen)
        engine.search_batched_legal(pos, uniform_evaluator_batched, max_batch)
        total_nodes += simulations
    return time.perf_counter() - start, total_nodes


def cmd_mcts_bench(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    fens = []
    for _ in range(args.positions):
        b = chess.Board()
        for _ in range(rng.randint(0, args.max_plies)):
            if not list(b.legal_moves) or b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        fens.append(b.fen())

    logger.info(f"{len(fens)} állás generálva.")

    times, nodes = [], 0
    for idx in range(args.repetitions):
        logger.info(f"Standard benchmark futás {idx + 1}/{args.repetitions}")
        t, n = benchmark_cpp_mcts(fens, args.simulations, args.max_batch)
        times.append(t)
        nodes += n

    mean_t = statistics.fmean(times)
    print(f"\nEredmények (Batch={args.max_batch}):")
    print(f"  Átlagos idő: {mean_t:.3f}s")
    print(f"  Ráta:        {len(fens)/mean_t:,.0f} keresés/mp")
    print(f"  NPS:         {nodes/sum(times):,.0f} csomópont/mp")

    if args.scaling:
        logger.info("Skálázási benchmark futtatása...")
        rows = []
        for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            t, n = benchmark_cpp_mcts(fens[:20], args.simulations, bs)
            if t > 0:
                rows.append({"batch_size": bs, "nps": n/t, "time_s": t})
        save_csv(args.output_dir / "mcts_scaling.csv", rows)
        logger.info("Skálázási eredmények elmentve.")


def cmd_cost(args: argparse.Namespace) -> None:
    log_path = args.run / "metrics" / "training.jsonl"
    if not log_path.exists():
        return

    events = []
    with open(log_path, 'r') as f:
        events = [json.loads(line) for line in f if line.strip()]

    costs = []
    for e in events:
        if 'iter' in e and 'selfplay_time_s' in e:
            costs.append({
                "iteration": e['iter'],
                "selfplay": float(e.get('selfplay_time_s', 0)),
                "train": float(e.get('train_time_s', 0)),
                "arena": float(e.get('arena_elapsed_s', 0))
            })

    if not costs:
        return

    total_s = sum(c['selfplay'] + c['train'] + c['arena'] for c in costs)
    kwh = (args.gpu_tdp * (total_s / 3600)) / 1000

    print("\nKÖLTSÉGELEMZÉS")
    print(f"Teljes idő:       {total_s/3600:.2f} óra")
    print(f"Becsült energia:  {kwh:.2f} kWh")

    if args.output:
        save_csv(args.output, costs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Chess AI - Benchmark csomag")
    parser.add_argument("--quiet", "-q", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_inf = subparsers.add_parser("inference")
    p_inf.add_argument("--devices", nargs="*")
    p_inf.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 64, 256])
    p_inf.add_argument("--dtypes", nargs="*", default=["float32", "float16"])
    p_inf.add_argument("--repeats", type=int, default=10)
    p_inf.add_argument("--warmup", type=int, default=3)
    p_inf.add_argument("--output-csv", type=Path, default=Path("benchmark_reports/inference_suite.csv"))
    p_inf.set_defaults(func=cmd_inference)

    p_sys = subparsers.add_parser("system")
    p_sys.add_argument("--mode", default="all", choices=["all", "selfplay", "training"])
    p_sys.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_sys.add_argument("--output-csv", type=Path, default=Path("benchmark_reports/system_bench.csv"))
    p_sys.set_defaults(func=cmd_system)

    p_rank = subparsers.add_parser("ranking")
    p_rank.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p_rank.add_argument("--output-dir", type=Path, default=Path("benchmark_reports"))
    p_rank.add_argument("--games-per-pair", type=int, default=6)
    p_rank.add_argument("--mcts-sims", type=int, default=64)
    p_rank.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_rank.add_argument("--include-random", action=argparse.BooleanOptionalAction, default=True)
    p_rank.add_argument("--include-greedy", action=argparse.BooleanOptionalAction, default=True)
    p_rank.set_defaults(func=cmd_ranking)

    p_chess = subparsers.add_parser("chess")
    p_chess.add_argument("--positions", type=int, default=200)
    p_chess.add_argument("--max-plies", type=int, default=80)
    p_chess.add_argument("--loops", type=int, default=50)
    p_chess.add_argument("--seed", type=int, default=2025)
    p_chess.add_argument("--output-dir", type=Path, default=Path("benchmark_reports"))
    p_chess.set_defaults(func=cmd_chess_bench)

    p_mcts = subparsers.add_parser("mcts")
    p_mcts.add_argument("--positions", type=int, default=120)
    p_mcts.add_argument("--max-plies", type=int, default=70)
    p_mcts.add_argument("--simulations", type=int, default=192)
    p_mcts.add_argument("--max-batch", type=int, default=24)
    p_mcts.add_argument("--repetitions", type=int, default=2)
    p_mcts.add_argument("--seed", type=int, default=2025)
    p_mcts.add_argument("--scaling", action="store_true")
    p_mcts.add_argument("--output-dir", type=Path, default=Path("benchmark_reports"))
    p_mcts.set_defaults(func=cmd_mcts_bench)

    p_cost = subparsers.add_parser("cost")
    p_cost.add_argument("--run", type=Path, required=True)
    p_cost.add_argument("--output", type=Path, default=Path("benchmark_reports/cost_breakdown.csv"))
    p_cost.add_argument("--gpu-tdp", type=float, default=220.0)
    p_cost.set_defaults(func=cmd_cost)

    args = parser.parse_args()
    if args.quiet:
        logger.setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
