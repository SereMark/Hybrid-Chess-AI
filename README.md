# Hybrid Chess AI

AlphaZero-style chess RL stack with a C++ engine + MCTS core (pybind11) and a PyTorch policy/value net. Designed to run end-to-end on a laptop RTX 3070.

---

## Requirements

- Windows 11 x64
- Python 3.12
- Visual Studio 2022 Build Tools (Desktop C++)
- CMake ≥ 3.21
- NVIDIA RTX 30-series driver

---

## Setup

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install -U pip

pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
````

Build the C++ core:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

---

## Training

Recommended preset for RTX 3070 laptop: `configs/thesis_3070.yaml`.

```powershell
$env:PYTHONPATH = "$PWD\src\python"
python -m main -c configs\thesis_3070.yaml
```

Overlay extra configs if needed:

```powershell
python -m main -c configs\thesis_3070.yaml -o configs\my_overrides.yaml
```

---

## Testing and Quality

```powershell
python -m pytest
python -m black src/python tests
python -m ruff check src/python tests
python -m mypy src/python
```