# Chess Engine Benchmark Report

Generated: 2025-10-08T14:49:14.894222+00:00
Python: 3.13.8 (tags/v3.13.8:a15ae61, Oct  7 2025, 12:34:25) [MSC v.1944 64 bit (AMD64)]
Platform: Windows-11-10.0.26100-SP0

## Dataset

- Name: baseline_small
- Positions: 200
- Loops per timing run: 80
- Timing repetitions: 3
- Description: Random playout positions

## Move Statistics

- Mean legal moves: 30.20
- Median: 31.00
- Min / Max: 0 / 52
- Std Dev: 9.80
- Quartiles (Q1 / Q2 / Q3): 25.00 / 31.00 / 37.00

## Correctness

- Positions checked: 200
- Mismatches: 0

## Timing Summary

### chesscore

- Mean time: 0.2322 s
- Std Dev: 0.0014 s
- Min / Max: 0.2311 / 0.2338 s
- Mean positions per second: 68921
- Mean moves per second: 2081745

### python-chess

- Mean time: 1.5924 s
- Std Dev: 0.0214 s
- Min / Max: 1.5763 / 1.6167 s
- Mean positions per second: 10049
- Mean moves per second: 303533


# Chess Engine Benchmark Report

Generated: 2025-10-08T14:49:33.421359+00:00
Python: 3.13.8 (tags/v3.13.8:a15ae61, Oct  7 2025, 12:34:25) [MSC v.1944 64 bit (AMD64)]
Platform: Windows-11-10.0.26100-SP0

## Dataset

- Name: baseline_medium
- Positions: 400
- Loops per timing run: 100
- Timing repetitions: 3
- Description: Random playout positions

## Move Statistics

- Mean legal moves: 29.50
- Median: 30.00
- Min / Max: 0 / 54
- Std Dev: 10.94
- Quartiles (Q1 / Q2 / Q3): 24.00 / 30.00 / 36.00

## Correctness

- Positions checked: 400
- Mismatches: 0

## Timing Summary

### chesscore

- Mean time: 0.5834 s
- Std Dev: 0.0019 s
- Min / Max: 0.5816 / 0.5854 s
- Mean positions per second: 68565
- Mean moves per second: 2022676

### python-chess

- Mean time: 3.6841 s
- Std Dev: 0.1033 s
- Min / Max: 3.5722 / 3.7757 s
- Mean positions per second: 10863
- Mean moves per second: 320465


# Chess Engine Benchmark Report

Generated: 2025-10-08T14:50:10.798802+00:00
Python: 3.13.8 (tags/v3.13.8:a15ae61, Oct  7 2025, 12:34:25) [MSC v.1944 64 bit (AMD64)]
Platform: Windows-11-10.0.26100-SP0

## Dataset

- Name: baseline_deep
- Positions: 600
- Loops per timing run: 150
- Timing repetitions: 3
- Description: Random playout positions

## Move Statistics

- Mean legal moves: 26.04
- Median: 28.00
- Min / Max: 0 / 55
- Std Dev: 12.32
- Quartiles (Q1 / Q2 / Q3): 19.00 / 28.00 / 35.00

## Correctness

- Positions checked: 600
- Mismatches: 0

## Timing Summary

### chesscore

- Mean time: 1.1806 s
- Std Dev: 0.0310 s
- Min / Max: 1.1612 / 1.2164 s
- Mean positions per second: 76264
- Mean moves per second: 1985664

### python-chess

- Mean time: 7.5193 s
- Std Dev: 0.3550 s
- Min / Max: 7.2801 / 7.9272 s
- Mean positions per second: 11987
- Mean moves per second: 312090
