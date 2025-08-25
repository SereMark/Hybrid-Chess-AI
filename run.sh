#!/usr/bin/env bash
set -euo pipefail

if python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('google.colab') else 1)" >/dev/null 2>&1; then
  IS_COLAB=1
else
  IS_COLAB=0
fi

if [[ ${IS_COLAB} -eq 1 ]]; then
  apt-get -yq update >/dev/null 2>&1 || true
  apt-get -yq install build-essential cmake ninja-build >/dev/null 2>&1 || true

  python - <<'PY'
import subprocess, sys
def pip(*args): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])
pip("--upgrade", "pip", "wheel")
pip("numpy", "pybind11>=2.10", "psutil")
pip("--index-url", "https://download.pytorch.org/whl/cu121", "torch")
PY

  PYBIN=python
else
  if [[ ! -d venv ]]; then
    python3 -m venv venv
  fi
  venv/bin/python -m pip install -q --upgrade pip wheel
  venv/bin/python -m pip install -q numpy pybind11 psutil
  venv/bin/python -m pip install -q --index-url https://download.pytorch.org/whl/cu121 torch

  PYBIN=venv/bin/python
fi

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR_ARGS+=("-GNinja")
fi

cmake -S cpp -B build \
  -Dpybind11_DIR="$(${PYBIN} -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Release \
  "${GENERATOR_ARGS[@]}"
cmake --build build -j"$(nproc)"

export PYTHONPATH="${PWD}:${PWD}/build"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:true"

exec ${PYBIN} -m hybridchess.trainer "$@"
