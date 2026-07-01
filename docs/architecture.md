# Architecture

This project is split between C++ and Python. I chose that boundary because chess search spends most of its time generating moves, copying positions, and walking the MCTS tree. Those operations benefit from native code. The training loop changes more often and depends heavily on PyTorch, so keeping it in Python makes experiments much easier.

The result is not the simplest possible codebase, but it is a useful compromise: C++ does the repetitive low-level work, while Python connects the larger parts of the learning system.

## The training loop

```text
YAML configuration
        |
        v
     Trainer
        |
        +----> SelfPlayEngine ----> C++ chess position + MCTS
        |                                  |
        |                                  v
        |                         BatchedEvaluator
        |                                  |
        |                                  v
        |                           PyTorch ChessNet
        |
        v
   ReplayBuffer
        |
        v
 gradient updates, scheduler, and EMA
        |
        v
 arena match against the best saved model
        |
        v
 checkpoint, metrics, and optional model promotion
```

At a high level, self-play asks MCTS for a move distribution. MCTS asks the neural network to evaluate leaf positions, then sends the resulting visit counts back to Python. The completed game's result becomes the value target for every stored position.

## C++ side

### Chess position and move generation

`src/core/chess.hpp` and `src/core/chess.cpp` contain the board representation and chess rules. A position stores piece bitboards, occupancy, side to move, castling rights, en passant information, move counters, and a Zobrist hash.

Move generation first creates candidate moves and then checks legality by making and unmaking them. The fast make/unmake path is especially important for MCTS because a single search may visit the same position tree many times.

The core also handles:

- FEN loading and saving
- check, checkmate, and stalemate
- threefold repetition
- the fifty-move rule
- insufficient material
- castling, promotion, and en passant

### Monte Carlo Tree Search

`src/core/mcts.hpp` and `src/core/mcts.cpp` implement the search tree. Each node keeps a policy prior, visit count, accumulated value, and the location of its children in a shared node pool.

The tree stores child indices instead of long-lived pointers. This matters because the node pool is backed by a `std::vector`; if that vector grows, its memory can move and old pointers would no longer be safe.

Leaf positions are evaluated in batches. Crossing from C++ into Python for every leaf would add a large amount of overhead and would also underuse the GPU. The search therefore gathers several leaves, sends them to Python together, expands them, and then continues.

Virtual loss temporarily makes a path look less attractive while its leaf is waiting for evaluation. Without it, several simulations in the same batch could repeatedly choose the same unfinished path.

### Python bindings

`src/core/bindings.cpp` exposes positions, moves, results, move encoders, and MCTS through `pybind11`. It also adapts the Python evaluator callback into the batched format expected by C++.

This boundary is one of the more delicate parts of the project. The bindings try to pass compact arrays and lists rather than repeatedly converting entire Python objects.

## Python side

### Network

`src/python/network.py` defines `ChessNet`, a residual convolutional network with two outputs:

- The policy head predicts a score for every move in the 73-by-64 AlphaZero-style action space.
- The value head estimates the position from the current player's point of view, between `-1` and `1`.

The model is intentionally modest so that experiments remain possible on a laptop GPU.

### Position encoding

`src/python/encoder.py` converts a position into neural-network planes. It includes piece locations for the current position and recent history, repetition information, castling rights, move counters, and en passant state.

Black-to-move positions are flipped into the same orientation as white-to-move positions. This gives the network one consistent point of view instead of making it learn two mirrored versions of the game.

Moves are encoded into the same canonical orientation, so policy targets still line up with the flipped board.

### Batched inference

`src/python/inference.py` owns an evaluation copy of the model. It combines requests that arrive close together, encodes their positions, performs one network call, and returns policy/value results to MCTS.

It also keeps small LRU caches for repeated positions. Search often reaches the same state more than once, so avoiding duplicate encoding and inference can save useful time. The caches are cleared whenever the evaluator receives newer model weights.

### Self-play and replay

`src/python/self_play.py` runs games in worker threads. For each position it stores:

- the encoded board state
- sparse policy indices
- MCTS visit counts
- the final game result from the player's point of view

The states and values are quantised before entering the replay buffer. Policy targets remain sparse because only legal moves have non-zero visit counts.

`src/python/replay_buffer.py` is a circular buffer. Sampling mixes recent positions with older ones: recent data follows the improving model more closely, while older data helps reduce sudden forgetting.

### Training and model promotion

`src/python/train_loop.py` performs the actual optimizer steps. The loss combines policy cross-entropy, value error, and a small entropy term. The loop also supports mixed precision, gradient clipping, a learning-rate schedule, and an exponential moving average (EMA) of the weights.

`src/python/arena.py` plays the candidate model against the retained best model with colours swapped across games. The candidate is promoted only if it reaches the configured score threshold.

With a small arena, this result is noisy. I kept the mechanism because it makes improvement explicit, but a larger experiment should use more games or a more statistically careful promotion rule.

### Checkpoints and run files

`src/python/checkpoint.py` saves more than model weights. A resumable run also needs the optimizer, scheduler, gradient scaler, EMA weights, replay/self-play state, random-number generator states, iteration counters, and arena state.

Each run is kept under one directory:

```text
runs/<run-id>/
    checkpoints/
        latest.pt
        best.pt
    metrics/
        training.csv
        training.jsonl
    arena_games/
    config/
    run_info.json
```

Raw runs are ignored by Git because they grow quickly. Smaller benchmark summaries are kept in `benchmark_reports/` so the repository still includes results that can be inspected.

## Important trade-offs

- **C++ speed versus development simplicity:** the native boundary improves search performance but makes builds, debugging, and tests more involved.
- **Larger inference batches versus latency:** waiting briefly can create a better GPU batch, but waiting too long slows every MCTS simulation.
- **Fresh replay data versus stability:** recent samples match the current model, while older samples make training less volatile.
- **Search depth versus game count:** deeper MCTS produces stronger targets per move, but fewer self-play games fit into the same compute budget.
- **Reproducibility versus maximum speed:** deterministic settings are useful when debugging, but some CUDA optimisations are faster when strict determinism is disabled.

Those trade-offs are also the reason for the different YAML experiment files. They are not just presets; each one tests a different balance between search quality, exploration, throughput, and update frequency.
