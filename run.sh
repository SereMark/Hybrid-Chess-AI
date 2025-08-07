set -e
if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip --quiet
    ./venv/bin/pip install torch numpy pybind11 psutil --quiet
    echo "Environment ready."
fi
echo "Building chess engine..."
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -Dpybind11_DIR=$(../venv/bin/python -m pybind11 --cmakedir) \
      ..
cmake --build . -j$(nproc)
cd ..
echo "Starting training..."
PYTHONPATH=build ./venv/bin/python train.py
