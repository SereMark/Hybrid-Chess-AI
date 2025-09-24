# Hybrid Chess AI

A self-play, AlphaZero-style chess training stack that pairs a high-performance C++ engine and MCTS with a PyTorch policy/value network. The C++ core handles chess rules, encoding, and batched MCTS; Python drives self-play, training, evaluation, and model gating.

## Highlights
- C++23 core with fast move generation, encoding, and Monte Carlo Tree Search (pybind11 bindings as `chesscore`)
- PyTorch residual CNN for policy (move logits) and value (win/draw/loss), trained from self-play
- Batched, cached inference for high MCTS throughput
- Replay buffer with sparse policy targets from MCTS visit counts
- Data augmentation (mirror, 180° rotation, vertical flip + color/side swap)
- Arena evaluation with statistical acceptance gate (Elo-style) to promote only stronger models
- Robust logging, checkpointing, and resumption

## Architecture
- core/: C++ engine and bindings
  - chess.hpp: board, moves, rules, Zobrist hashing
  - mcts.hpp: batched-legal MCTS with Dirichlet noise, UCB, root reuse
  - encoder.hpp: position feature planes (with history) for NN input
  - replay_buffer.hpp: compact storage and sampling of training records
  - bindings.cpp: pybind11 module exposing engine as `chesscore`
- python/: training and evaluation
  - network.py: residual CNN (policy head over 73×64 moves, value head)
  - inference.py: BatchedEvaluator for fast, cached GPU inference
  - self_play.py: data generation with MCTS → sparse policy targets + values
  - train_loop.py, train.py: training loop, scheduler, EMA, summaries
  - arena.py: challenger vs. incumbent evaluation and acceptance gating
  - augmentation.py, reporting.py, checkpoint.py, config.py

## Requirements
- OS: Linux/macOS/Windows
- C++ toolchain with C++23 support
- CMake ≥ 3.18 and (optionally) Ninja
- pybind11 (CMake package)
- Python ≥ 3.10
- NVIDIA GPU with CUDA; PyTorch with CUDA support
- Python packages: torch (CUDA build), numpy, psutil

Install PyTorch per official instructions to match your CUDA driver: https://pytorch.org/get-started/locally/

## Build the C++ core (chesscore)
The CMake build outputs the `chesscore` Python extension into the `python/` directory.

On Linux/macOS:
```bash
# From repo root
python -m pip install --upgrade pip
python -m pip install cmake pybind11 numpy psutil

# Configure + build (defaults to Makefiles on Linux/macOS)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# You should see python/chesscore*.so (or .dylib/.pyd on your OS)
```

On Windows (PowerShell):
```powershell
py -m pip install --upgrade pip
py -m pip install cmake pybind11 numpy psutil

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# Verify python\chesscore*.pyd exists
```

If CMake can't locate the pybind11 CMake package automatically, append
`-Dpybind11_DIR="$(python -m pybind11 --cmakedir)"` (or use `py` on Windows)
to the configure command above.

Tip: The build uses `-O3 -march=native`. If you redistribute binaries, build on the target CPU.

## Quickstart: Train
Two ways to run (ensure `python/` is on Python path):

Option A (recommended):
```bash
cd python
python train.py          # add --resume to continue from checkpoint
```

Option B:
```bash
PYTHONPATH=python python python/train.py  # add --resume to continue
```

The first run prints a detailed system and configuration summary and starts self-play → training iterations. Checkpoints and logs are written in the repo root by default.

## Configuration
All knobs live in `python/config.py`. Key groups:
- LOG: paths and CSV logging toggles (training_log.csv, arena_log.csv)
- TORCH: AMP, channels-last, thread counts
- MODEL: blocks/channels and value head dims
- EVAL: batching, cache sizes, dtype for caches
- SELFPLAY/MCTS: simulations, Dirichlet, temperature policy, resign logic
- TRAIN: LR schedule, batch sizes, loss weights, EMA
- ARENA: frequency, games per eval, deterministic settings, acceptance gate (z-score, min games)

Change values directly in config.py and re-run. To resume, use `--resume`.

## What gets written where
- Checkpoints: `checkpoint.pt` (periodic), `best_model.pt` (on arena acceptance)
- CSV logs: `training_log.csv` (iteration metrics), `arena_log.csv` (per-eval outcomes)
- Optional PGN samples from arena (files like `YYYYmmdd_HHMMSS_*.pgn`)

## Training pipeline (how it works)
1. Self-play generates games using MCTS guided by the current EMA model; visit counts → sparse policy targets; final result → scalar value.
2. ReplayBuffer stores (encoded state, move-index/count pairs, value) with recent/older sampling mix.
3. Batches are trained on policy (NLL over sparse targets with smoothing) and value (MSE), optional entropy bonus.
4. LR schedules via warmup+cosine; EMA tracks a smoothed copy for evaluation.
5. Periodic arena pits the EMA challenger vs. the current best; an Elo-like gate decides accept/reject.

## Performance tips
`python/train.py` sets helpful defaults automatically:
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
- CUDA_MODULE_LOADING=LAZY, CUDA_CACHE_MAXSIZE=2147483648
- OMP_NUM_THREADS=1, MKL_NUM_THREADS=1
- Determinism toggles if `SEED != 0` (see config.py)

The code auto-tunes training batch size and evaluator batching to fit available VRAM.

## Repository layout
```
core/      # C++ chess, MCTS, encoder, replay buffer, pybind11 bindings
python/    # Training stack: network, inference, self_play, train_loop, arena, augmentation, reporting, checkpoint, config
CMakeLists.txt
pyproject.toml  # lint/format config for Ruff
```
## Acknowledgements
Inspired by AlphaZero-like self-play systems. Built with PyTorch and pybind11.

## License
MIT License. See `LICENSE` for details.
