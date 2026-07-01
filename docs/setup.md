# Setup

The project is currently built and tested mainly on Windows. The instructions below match the environment I used during development and the Windows runner used by CI.

## Before you start

You will need:

- Windows 11 x64
- Python 3.13
- Visual Studio 2022 Build Tools
- The **Desktop development with C++** workload
- CMake 3.21 or newer
- Git

An NVIDIA GPU is optional for tests and small smoke runs, but strongly recommended for self-play and training. The supplied configs were tuned around an RTX 3070 Mobile, not around CPU-only training.

## 1. Create the Python environment

From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If the `py` launcher is not installed, use the full path to your Python 3.13 executable instead.

The editable install provides the `hybrid-chess-train` command and installs the packages used by the project. Benchmark-only plotting dependencies can be installed with:

```powershell
python -m pip install -e ".[dev,benchmarks]"
```

## 2. Check PyTorch and CUDA

The default dependency installs PyTorch 2.8.0. You can check what PyTorch sees with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If CUDA is not available even though the machine has a supported NVIDIA GPU, install the PyTorch wheel that matches the driver and CUDA setup. For example, the development environment used:

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

PyTorch wheel availability changes over time, so check the official PyTorch installation page if this exact command is no longer appropriate.

## 3. Build the C++ extension

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The build creates the `chesscore` extension and copies it into `src/python/`. A successful build should leave a file with a name similar to:

```text
src/python/chesscore.cp313-win_amd64.pyd
```

The exact filename depends on the Python version and platform.

To confirm that Python can import it:

```powershell
$env:PYTHONPATH = "$PWD\src\python"
python -c "import chesscore; print(chesscore.__doc__)"
```

## 4. Run the tests

```powershell
$env:PYTHONPATH = "$PWD\src\python"
python -m pytest
```

If many native tests are skipped, `chesscore` was probably not built or cannot be found on `PYTHONPATH`.

## 5. Start training

The baseline experiment is a sensible first run:

```powershell
$env:PYTHONPATH = "$PWD\src\python"
hybrid-chess-train -c configs\run1_baseline.yaml
```

You can force the device:

```powershell
hybrid-chess-train -c configs\run1_baseline.yaml --device cuda
hybrid-chess-train -c configs\run1_baseline.yaml --device cpu
```

To continue a previous run:

```powershell
hybrid-chess-train -c configs\run1_baseline.yaml --resume
```

Resume searches the configured runs directory for the newest usable `latest.pt` checkpoint. You can also set `HYBRID_CHESS_RESUME` to a particular run directory.

## Configuration layering

The first `-c` file replaces the built-in defaults section by section. Additional `-c` files are layered on top. Files passed with `-o` are applied last:

```powershell
hybrid-chess-train `
  -c configs\run1_baseline.yaml `
  -o my-local-overrides.yaml `
  --device cuda
```

This is useful for keeping machine-specific changes, such as worker counts or batch sizes, outside the tracked experiment files.

## What a run produces

Training creates a directory under `runs/` containing:

- `checkpoints/latest.pt` — the newest resumable checkpoint
- `checkpoints/best.pt` — the model currently retained by the arena
- archived checkpoint snapshots, if enabled
- `metrics/training.csv` and `metrics/training.jsonl`
- arena games in PGN format
- `run_info.json`
- the merged configuration used for the run

These files are ignored by Git because checkpoints and PGNs can become large. If a result is worth keeping in the repository, it should be reduced to a small report in `benchmark_reports/`.

## Common problems

### CMake cannot find Python or pybind11

Make sure the virtual environment is active before configuring CMake. If several Python versions are installed, pass the executable explicitly:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DPython3_EXECUTABLE="$PWD\.venv\Scripts\python.exe"
```

If `pybind11` is missing:

```powershell
python -m pip install -e ".[dev]"
```

Then delete or reconfigure the `build/` directory so CMake checks again.

### `import chesscore` fails after a build

Check that:

1. The Release build completed without errors.
2. A `.pyd` file exists in `src/python/`.
3. `PYTHONPATH` includes `src/python`.
4. The extension was built with the same Python version that is running it.

### Training is extremely slow

CPU training is expected to be slow. For a quick end-to-end check, copy a config and reduce:

- `TRAIN.total_iterations`
- `TRAIN.games_per_iter`
- `MCTS.train_simulations`
- `ARENA.games_per_eval`
- `ARENA.mcts_simulations`

It is better to make a small override file than to edit a tracked experiment config just for a smoke test.

### CUDA runs out of memory

Reduce `TRAIN.batch_size` and `EVAL.batch_size_max` first. The network size (`MODEL.channels` and `MODEL.blocks`) has a larger effect but changes the experiment itself.

### Windows blocks virtual-environment activation

PowerShell may require a less restrictive execution policy for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Only change this if it matches your machine's security requirements.
