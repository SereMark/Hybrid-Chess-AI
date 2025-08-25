#!/usr/bin/env bash
set -euo pipefail

apt-get -yq install build-essential cmake ninja-build >/dev/null

python - <<'PY'
import subprocess, sys
def pip(*args): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])
pip("--upgrade", "pip", "wheel")
pip("numpy", "pybind11>=2.10", "psutil")
pip("--index-url", "https://download.pytorch.org/whl/cu121", "torch")
PY

cmake -S cpp -B build \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -GNinja
cmake --build build -j"$(nproc)"

export PYTHONPATH="${PWD}:${PWD}/build"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:true"

python - <<'PY'
try:
    import torch
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
PY

exec python -m hybridchess.trainer "$@"
