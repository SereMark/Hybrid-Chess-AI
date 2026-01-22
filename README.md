# Hybrid Chess AI

AlphaZero-style chess engine based on Reinforcement Learning (RL) with a C++ core (MCTS + chess engine, `pybind11` bindings) and a PyTorch policy/value network. Designed to run the complete end-to-end training and evaluation pipeline on an RTX 3070 laptop.

---

## Requirements / Tested Environment

- Windows 11 x64
- Python 3.13
- Visual Studio 2022 Build Tools (Desktop C++)
- CMake ≥ 3.21
- NVIDIA driver + CUDA-compatible GPU (RTX 30xx recommended)

---

## Quick Start

1. **Virtual Environment and Packages**:
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   # PyTorch: check the command for your setup at https://pytorch.org/get-started/locally/
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```

2. **Compile C++ Core**:
   ```powershell
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
   cmake --build build --config Release
   ```

3. **Basic Training Run**:
   ```powershell
   $env:PYTHONPATH = "$PWD\src\python"
   python -m main -c configs\run1_baseline.yaml
   ```

Checkpoints and results will appear in the `runs/` directory.

---

## Advanced Usage

- **Training with Configuration + Overrides**:
  ```powershell
  $env:PYTHONPATH = "$PWD\src\python"
  python -m main -c configs\run1_baseline.yaml -o configs\my_overrides.yaml
  ```

---

## Directory Structure

- `src/core/` - C++ chess engine, MCTS, and `pybind11` bindings.
- `src/python/` - Training pipeline (self-play, replay buffer, network, trainer, main CLI).
- `configs/` - YAML configurations for different runs.
- `runs/` - Automatically generated run directories (checkpoints, logs).
- `tools/` - Benchmark and artifact generation scripts.

---

## Testing and Quality

```powershell
$env:PYTHONPATH = "$PWD\src\python"
python -m pytest
python -m black src/python tests
python -m ruff check src/python tests
python -m mypy src/python
```

---

## Citation and License

This project is licensed under the MIT License, with the following **exceptions**:

1. **Thesis**: The `thesis/` directory is **excluded** from the open-source license. All rights are reserved for the thesis document. Do not copy or redistribute the thesis text/figures.
2. **Citation**: If you use this codebase in your work, you **must** cite the author:

```bibtex
@software{HybridChessAI2026,
  author = {Sere Gergő Márk},
  title = {Hybrid Chess AI: AlphaZero-style RL with C++ Core},
  year = {2026},
  url = {https://github.com/SereMark/Hybrid-Chess-AI}
}
```