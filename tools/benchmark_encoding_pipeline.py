# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Iterable

from _bench_common import (
    benchmark_callable,
    import_chesscore,
    import_python_chess,
    prepare_extension_import_paths,
    summarize_measurements,
    write_reports,
    render_markdown_table,
)

prepare_extension_import_paths()

from encoder import HISTORY_LENGTH, encode_position  # noqa: E402

ChessPosition = Any

chess: Any = import_python_chess()
ccore: Any = import_chesscore()


def generate_fens(count: int, max_random_plies: int, seed: int) -> list[list[str]]:
    """Generate sequences of FENs, including history frames."""

    rng = random.Random(seed)
    sequences: list[list[str]] = []
    history_len = max(1, HISTORY_LENGTH)

    for _ in range(count):
        board = chess.Board()
        frames = [board.fen()]
        plies = rng.randint(0, max_random_plies)
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            frames.append(board.fen())
            if board.is_game_over():
                break
        if len(frames) > history_len:
            frames = frames[-history_len:]
        sequences.append(frames)
    return sequences


def build_positions(sequences: Iterable[list[str]]) -> tuple[list[ChessPosition], list[list[ChessPosition]]]:
    """Convert FEN sequences into chesscore Position objects."""

    positions: list[ChessPosition] = []
    histories: list[list[ChessPosition]] = []

    for frames in sequences:
        history_frames = frames[:-1]
        main_fen = frames[-1]

        position = ccore.Position()
        position.from_fen(main_fen)
        positions.append(position)

        hist_positions: list[ChessPosition] = []
        for fen in history_frames:
            hpos = ccore.Position()
            hpos.from_fen(fen)
            hist_positions.append(hpos)
        histories.append(hist_positions)

    return positions, histories


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the encoding pipeline (FEN -> tensor).")
    parser.add_argument("--positions", type=int, default=256, help="Number of random positions to generate.")
    parser.add_argument("--max-random-plies", type=int, default=80, help="Maximum random plies per generated position.")
    parser.add_argument("--seed", type=int, default=2025, help="RNG seed.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed iterations per scenario.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per scenario.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "encoding_pipeline.csv",
        help="Write results to CSV.",
    )
    args = parser.parse_args()

    sequences = generate_fens(args.positions, args.max_random_plies, args.seed)
    positions, histories = build_positions(sequences)

    def load_from_fen() -> None:
        for frames in sequences:
            pos = ccore.Position()
            pos.from_fen(frames[-1])
            _ = pos.hash

    def encode_no_history() -> None:
        for pos in positions:
            encode_position(pos)

    def encode_with_history() -> None:
        for pos, hist in zip(positions, histories, strict=False):
            encode_position(pos, history=hist)

    measurements = [
        benchmark_callable("fen->position", load_from_fen, warmup=args.warmup, repeat=args.repeats),
        benchmark_callable("encode-no-history", encode_no_history, warmup=args.warmup, repeat=args.repeats),
        benchmark_callable("encode-with-history", encode_with_history, warmup=args.warmup, repeat=args.repeats),
    ]

    for measurement in measurements:
        measurement.extras.update(
            {
                "positions": args.positions,
                "history_length": HISTORY_LENGTH,
                "max_random_plies": args.max_random_plies,
            }
        )

    report = summarize_measurements(
        measurements,
        metadata={
            "positions": args.positions,
            "max_random_plies": args.max_random_plies,
            "seed": args.seed,
            "repeats": args.repeats,
        },
    )

    print("Encoding pipeline benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
