# Testing and CI

The test suite is split across the C++ chess extension and the Python training code. I wanted the tests to cover the places where a bug can look believable for a long time: move legality, board orientation, search statistics, replay data, and restoring a training run.

## Running the full check locally

Build the native module first:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Then run the same checks used by CI:

```powershell
$env:PYTHONPATH = "$PWD\src\python"
python -m pytest
python -m ruff check src/python tests
python -m black --check src/python tests
python -m mypy
```

For a shorter test run:

```powershell
python -m pytest -q
```

## What is tested?

### Chess core

The native tests check legal move generation, FEN round-trips, checkmate, stalemate, insufficient material, castling, promotion, en passant, repetition, and position restoration after make/unmake.

These tests matter because an illegal position can still flow through self-play and produce normal-looking training data. Catching the error at the chess layer is much easier than diagnosing a network trained on corrupted games.

### MCTS

Search tests cover:

- simulation and visit-count behaviour
- the influence of policy priors
- mate-in-one selection
- tree reuse after a move
- batched evaluation
- convergence on small controlled positions

MCTS tests generally use simple evaluators so the expected behaviour is understandable without a trained network.

### Encoding and model output

Encoder tests check piece planes, history, auxiliary features, canonical orientation, and policy move indices. Model tests check output shapes and basic forward/backward behaviour.

Canonical orientation deserves special attention: black-to-move positions and their legal moves must be flipped in exactly the same way. If only one side is flipped, the code still runs but the policy labels become wrong.

### Inference

Inference tests cover request batching, cache limits, cache clearing after a model refresh, shutdown, and dtype behaviour. CPU tests also verify that unsupported half-precision paths fall back safely.

### Replay and training

The Python tests check:

- circular-buffer capacity and resizing
- recent/older sample mixing
- serialising and restoring replay state
- policy and value losses
- optimizer updates
- adaptive game and batch schedules
- self-play controls

### Checkpoints

Checkpoint tests verify metadata, latest/best model files, and resume behaviour. A useful checkpoint must restore the training process rather than only the network weights, so the saved state includes random-number generators, optimizer state, scheduler position, gradient scaler, replay/self-play state, and arena state.

## Native tests and skipped tests

Tests that require `chesscore` use `pytest.importorskip` or the shared `ensure_chesscore` fixture. This allows parts of the Python code to be inspected in environments without a C++ compiler.

There is one important consequence: a green test run with many skips is not a complete result. For the same coverage as CI, build the extension first and check the pytest summary for unexpected skips.

## CI

The GitHub Actions workflow runs on Windows Server 2022 and:

1. checks out the repository
2. installs Python 3.13
3. installs the project and development dependencies
4. builds the Release version of `chesscore`
5. runs pytest
6. runs Ruff and Black
7. runs Mypy

GPU training is not part of CI. It would be expensive, slow, and difficult to make deterministic enough for every commit. The CI job instead checks that the system builds and that its smaller behavioural tests pass.

## Adding a test

When fixing a bug, the most useful test is usually the smallest position or state that reproduces it. For chess rules, prefer a short FEN over a long played game. For training code, use tiny arrays and deterministic random seeds.

Tests should explain the behaviour through their setup and assertions. Comments are best kept for cases where the reason behind the test is not obvious from the code itself.
