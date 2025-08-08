set -e

VENV_DIR=${VENV_DIR:-venv}
PYTHON=${PYTHON:-python3}

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  $PYTHON -m venv "$VENV_DIR"
  "$VENV_DIR"/bin/pip install --upgrade pip --quiet
  "$VENV_DIR"/bin/pip install torch numpy pybind11 --quiet
  echo "Environment ready."
fi

echo "Building chess engine..."
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$("$VENV_DIR"/bin/python -m pybind11 --cmakedir) \
  -DCHESSAI_NATIVE_OPT=ON
cmake --build build -j"$(nproc)"

echo "Starting pipeline..."
PYTHONPATH=build "$VENV_DIR"/bin/python train.py
