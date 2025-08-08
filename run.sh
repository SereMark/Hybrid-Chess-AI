set -euo pipefail

VENV_DIR=${VENV_DIR:-venv}
PY=${PYTHON:-python3}

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv "$VENV_DIR"
  "$VENV_DIR"/bin/pip install --quiet --upgrade pip torch numpy pybind11
fi

echo "Building chess engine..."
mkdir -p build
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$("$VENV_DIR"/bin/python -m pybind11 --cmakedir)
cmake --build build -j"$(nproc)"

echo "Starting pipeline..."
export PYTHONPATH="$(pwd):build"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
exec "$VENV_DIR"/bin/python -m hca.pipeline
