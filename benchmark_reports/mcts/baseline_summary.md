# MCTS Benchmark Report — mcts_small

Generated: 2025-10-08T14:50:56.442995+00:00
Python: 3.13.8 (tags/v3.13.8:a15ae61, Oct  7 2025, 12:34:25) [MSC v.1944 64 bit (AMD64)]
Platform: Windows-11-10.0.26100-SP0

## Scenario

- Positions: 200
- Max random plies: 80
- Simulations: 256
- Max batch: 32
- Repetitions: 3
- Seed: 9001
- Description: Random playout positions (small)

## Timing

- C++ MCTS time: 0.7748 s
- Python MCTS time: 5.0756 s
- Speedup (cpp/python): 6.55x
- Positions per second (C++): 258
- Positions per second (Python): 39

# MCTS Benchmark Report — mcts_medium

Generated: 2025-10-08T14:51:58.349269+00:00
Python: 3.13.8 (tags/v3.13.8:a15ae61, Oct  7 2025, 12:34:25) [MSC v.1944 64 bit (AMD64)]
Platform: Windows-11-10.0.26100-SP0

## Scenario

- Positions: 400
- Max random plies: 120
- Simulations: 384
- Max batch: 48
- Repetitions: 3
- Seed: 42
- Description: Random playout positions (medium)

## Timing

- C++ MCTS time: 2.5072 s
- Python MCTS time: 17.6089 s
- Speedup (cpp/python): 7.02x
- Positions per second (C++): 160
- Positions per second (Python): 23

# MCTS Benchmark Report — mcts_deep

Generated: 2025-10-08T14:53:53.596847+00:00
Python: 3.13.8 (tags/v3.13.8:a15ae61, Oct  7 2025, 12:34:25) [MSC v.1944 64 bit (AMD64)]
Platform: Windows-11-10.0.26100-SP0

## Scenario

- Positions: 600
- Max random plies: 200
- Simulations: 512
- Max batch: 64
- Repetitions: 3
- Seed: 1337
- Description: Random playout positions (deep)

## Timing

- C++ MCTS time: 4.5449 s
- Python MCTS time: 32.5747 s
- Speedup (cpp/python): 7.17x
- Positions per second (C++): 132
- Positions per second (Python): 18
