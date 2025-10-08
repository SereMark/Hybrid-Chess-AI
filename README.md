# Hybrid Chess AI

## Prereqs
- Windows 11 x64 with latest updates and NVIDIA drivers  
- Python 3.13 x64 (`Add to PATH` during install)  
- Visual Studio 2022 Build Tools · *Desktop development with C++ workload*  
- CMake ≥ 3.21 (`winget install Kitware.CMake`)  
- Optional: `nvidia-smi` in PATH for quick GPU diagnostics

## Bootstrap
```powershell
git clone https://github.com/your-account/Hybrid-Chess-AI.git
cd Hybrid-Chess-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy psutil pybind11 python-chess
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Build the C++ Core
Run from **Developer PowerShell for VS 2022** so `cl.exe` is available:
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```
The tailored `CMakeLists.txt` emits a single `chesscore` wheel-ready module into `src/python` using AVX2, `/MP16`, and link-time code generation tuned for the 5800H.

## Train
```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src\python"
python -m main
```
Runtime defaults in `config.py`/`main.py` select 12 intra-op threads, 4 inter-op threads, enable CUDA TF32, and spin up 10 self-play workers—matching the 8C/16T CPU and RTX 3070.

Opening book from https://github.com/lichess-org/chess-openings