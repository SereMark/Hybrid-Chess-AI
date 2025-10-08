# Hybrid Chess AI – Windows Setup Guide

This project combines a C++ core with a Python training stack to build and train a hybrid chess engine. The instructions below walk through a full Windows install, from prerequisites to launching the training loop, so you can reproduce a working environment from scratch.

## Quick View

1. **Install** Python 3.13 (64-bit), CMake ≥ 3.18, and Visual Studio Build Tools (Desktop C++ workload). Update your NVIDIA driver if you plan to train on GPU.
2. **Clone** the repo and create a virtual environment.
3. **Install** Python dependencies (`numpy`, `psutil`, `pybind11`, and PyTorch—CUDA wheels recommended if you have an NVIDIA GPU).
4. **Build** the C++ extension with CMake from Developer PowerShell.
5. **Set** `PYTHONPATH` to include `python` and `python\Release`, then run `python .\python\train.py`.

## Step-by-Step

### 1. Install Prerequisites

| Tool | Notes |
| --- | --- |
| Windows 10/11 (64-bit) | Fully patched |
| Python 3.13 (64-bit) | Install from [python.org](https://www.python.org/downloads/) and add to `PATH` |
| CMake ≥ 3.18 | `winget install Kitware.CMake` |
| Visual Studio 2022 Build Tools | Select “Desktop development with C++”; use Developer PowerShell for VS |
| NVIDIA GPU (optional) | Update drivers; PyTorch wheels include CUDA 12.4 runtime |

### 2. Clone the Repository

```powershell
git clone https://github.com/your-account/Hybrid-Chess-AI.git
cd Hybrid-Chess-AI
```

### 3. Create and Activate a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 4. Install Python Dependencies

Common packages used across both CPU and GPU builds:

```powershell
pip install numpy psutil pybind11
```

#### 4.1 PyTorch (choose one)

**GPU build (recommended if you have an NVIDIA GPU):**

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**CPU-only build (if you do not have an NVIDIA GPU):**

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify the installation:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda)"
```

If `CUDA available` is `False` despite having a GPU, double-check that you installed the CUDA wheel (`cu124`) and that the NVIDIA driver is up to date. At the time of writing, the PyTorch index provides `torch-2.6.0+cu124` for Python 3.13—future releases may appear under newer CUDA versions; adjust the index URL accordingly.

### 5. Build the C++ Core (pybind11 Extension)

Run the following from the project root **inside a Developer PowerShell for VS 2022 window** (so `cl.exe` is on `PATH`):

```powershell
$pybind11Dir = python -c "import pybind11, pathlib; print(pathlib.Path(pybind11.__file__).parent / 'share' / 'cmake' / 'pybind11')"
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$pybind11Dir"
cmake --build build --config Release
```

If CMake still warns about `FindPythonInterp`, you can silence it by adding `set(PYBIND11_FINDPYTHON ON)` before `find_package(pybind11)` in `CMakeLists.txt`. The build output goes to `python\Release\chesscore.cp313-win_amd64.pyd`, the module imported by Python.

### 6. Configure Runtime Paths

Before running Python entry points, ensure the interpreter can find both the Python package folder and the compiled extension:

```powershell
$env:PYTHONPATH = "$PWD\python" + [System.IO.Path]::PathSeparator + "$PWD\python\Release"
```

You can add that line to your shell profile (e.g., `Microsoft.PowerShell_profile.ps1`) for convenience.

### 7. Launch Training

Activate the virtual environment (if not already active) and start the trainer:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\python" + [System.IO.Path]::PathSeparator + "$PWD\python\Release"
python .\python\train.py
```

The trainer will log detailed progress to the console and write metrics to `logs\training_log.csv`. The first iteration is the slowest as it performs self-play, builds the replay buffer, and compiles CUDA kernels (if available).

### 8. GPU Troubleshooting

- **CUDA unavailable:** Run the verification snippet in section 4.1. If it still reports `False`, confirm you used the `cu124` wheel and your GPU driver is current.
- **cuDNN warnings:** PyTorch wheels include cuDNN internally. Messages about `cudnn*.dll` missing from `CUDA_PATH` are safe to ignore unless you rely on the standalone CUDA Toolkit.
- **Compile errors about `cl.exe`:** Always build from a Developer PowerShell/Command Prompt, or add `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<version>\bin\Hostx64\x64` to your `PATH`.

### 9. Clean Rebuild

If you need to rebuild from scratch:

```powershell
Remove-Item build -Recurse -Force
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### 10. Project Layout

- `core/` - C++ implementation for chess logic, MCTS, replay buffer
- `python/` - Training loop, network, metrics, and reporting scripts
- `python/Release/` - Location of the compiled `chesscore` extension after building

### 11. Reporting Issues

If you run into problems not covered here, collect:

1. The console output (especially build or Python tracebacks)
2. Output from the PyTorch diagnostic command in section 4.1
3. Information about your GPU/driver (from `nvidia-smi`)

Include that data when filing an issue or asking for help to speed up debugging.