# Hybrid Chess AI

**Hybrid Chess AI** is the full AlphaZero-style reinforcement learning stack developed for the bachelor thesis “Hybrid Chess AI: Process-Focused Reinforcement Learning on Commodity Hardware.” The repository is engineered for reproducibility on lower class GPUs while maintaining professional standards for logging, testing, and documentation. This README consolidates the complete thesis run guide alongside build, training, and validation instructions.

---

## Repository Overview

| Module | Role | Highlights |
| --- | --- | --- |
| `src/python/config.py` | Configuration dataclasses | YAML overlay support, tested merge/reset semantics |
| `src/python/trainer.py` | Orchestrates self-play, training, arena gating | Robust logging, checkpointing, EMA, resume support |
| `src/python/self_play.py` | Batched MCTS self-play with resignation/adjudication controls | Opening-book & curriculum aware |
| `src/python/train_loop.py` | Core training iteration, augmentation, metrics | Extensive unit tests for augmentation & resignation logic |
| `src/core/*.cpp` | High-performance chess engine & MCTS | Exposed via `pybind11`; tests cover legal moves, repetition, search outputs |
| `tests/` | Comprehensive Python & C++ coverage | 26 tests covering augmentation, replay buffer, trainer, chesscore |

Quality gates:

```powershell
python -m black src/python tests
python -m ruff check src/python tests
python -m mypy src/python tests
python -m pytest
```

---

## Environment & Build Requirements

### Recommended Platform
- Windows 11 x64 with current NVIDIA drivers
- Python **3.13** (validated end-to-end for CUDA 12.6 wheels)
- Visual Studio 2022 Build Tools (Desktop C++ workload)
- CMake ≥ **3.21** (`winget install Kitware.CMake`)

### Bootstrap Checklist
```powershell
git clone https://github.com/your-account/Hybrid-Chess-AI.git
cd Hybrid-Chess-AI
python -m venv .venv
\.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Install CUDA-enabled PyTorch (adjust compute platform if needed)
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Verify CUDA availability
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Smoke test the environment:
```powershell
python -m pytest
python -m compileall src/python
```

### Compiling the C++ Core
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```
This produces the `chesscore` extension under `src/python/`. Tests in `tests/test_chesscore.py` validate legal move generation, repetition detection, and MCTS search.

---

## Running Training

### Default launch
```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src\python"
python -m main
```
Key runtime defaults (see `config.py`):
- Auto-configured thread counts when `TORCH.threads_intra/inter` are zero
- Mixed precision (TF32 on CUDA) with deterministic CPU fallback
- Two self-play workers generating 24 games/iteration
- Exponential moving average (EMA) weights used for evaluation and gating

Each iteration logs:
- Self-play summary
- Training metrics
- Arena result (if applicable)
- Metrics CSV/JSONL (`runs/<timestamp>/metrics/`)
- Logs (`runs/<timestamp>/logs/`)
- Checkpoints (`runs/<timestamp>/checkpoints/`) containing optimiser, scheduler, EMA, RNG state
- Arena PGNs (`runs/<timestamp>/arena_games/`)
- Run metadata (`runs/<timestamp>/run_info.json`) + config snapshot (`runs/<timestamp>/config/`)

---

## Low GPU Run Plan (Laptop RTX 3070)

### Objectives
- Reproduce the AlphaZero-style training pipeline on laptop-class hardware.
- Collect process-centric evidence: training metrics, self-play statistics, arena evaluations, and checkpoints.
- Frame results around engineering constraints rather than absolute playing strength.

### Hardware Baseline
- **GPU**: NVIDIA RTX 3070 Laptop (8 GB VRAM)
- **CPU**: 8-core mobile CPU (e.g., Ryzen 7 5800H)
- **RAM**: 16 GB
- **Storage**: ≥50 GB free SSD recommended
- **OS**: Windows 11 (WSL optional but not required)

**Known Limitations**
- Thermal throttling can reduce throughput; use active cooling and monitor `nvidia-smi` for clocks/temps.
- 8 GB VRAM constrains network width and search simulations; the preset already accounts for this.
- Self-play workers are CPU-bound; keep the laptop plugged in and minimise background load.

### Configuration Preset (`configs/thesis_3070.yaml`)
- Network: 5 residual blocks, 88 channels (smaller than desktop baseline).
- Self-play: 28 games/iteration with 72→40 simulations (decrease with game length).
- Training batch: adaptive 96-160 samples to prevent VRAM spikes.
- Arena: every 16 iterations with neutral acceptance margin.
- Checkpoints: every 24 iterations; CUDA cache flush every 96 to avoid fragmentation.

### Run Procedure
1. **Environment preparation**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
   cmake --build build --config Release
   ```
2. **Activate thesis preset**
   ```powershell
   $env:HYBRID_CHESS_CONFIG = "$PWD\configs\thesis_3070.yaml"
   ```
3. **Smoke test**
   ```powershell
   python -m pytest
   python -m compileall src/python
   ```
4. **Main training run**
   ```powershell
   python -m main --resume
   ```
   - Target duration: 48-60 hours for ~480 iterations.
   - Pause with `Ctrl+C`; rerun the same command to resume from the latest checkpoint.
5. **Arena monitoring**
   - PGNs written to `runs/<timestamp>/arena_games/`; sample games for qualitative assessment.
   - If arena scores plateau, record the observation instead of retuning mid-run.

### Data & Documentation for Submission
- `runs/<timestamp>/metrics/training.csv` and `.jsonl`
- `runs/<timestamp>/checkpoints/` (`latest.pt`, `best.pt`)
- `runs/<timestamp>/arena_games/` PGNs (include notable games in thesis appendix)
- Hardware snapshot (`nvidia-smi -L`, CPU info, RAM usage)
- Config file used (`runs/<timestamp>/config/merged.yaml`) and commit SHA (`git rev-parse HEAD`)

---

## Support & Contact
The repository is structured for transparent academic review. When reporting an issue, capture:
- Command executed
- Config overrides
- Relevant log excerpts or metrics rows
This ensures reproducible troubleshooting in the thesis timeframe.
