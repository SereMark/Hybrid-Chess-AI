# Hibrid Chess AI using CNN, MCTS and Opening Book

## Introduction

This project aims to develop a hybrid chess artificial intelligence that integrates a convolutional neural network (CNN), Monte Carlo Tree Search algorithm (MCTS), opening book, and graphical user interface (GUI). This AI is capable of evaluating and solving chess games by leveraging the advantages of deep learning techniques and search algorithms.

### Important Note

This project is part of a thesis, so the code found here is only for educational and research purposes. Commercial use is not permitted.

## Requirements

The software and libraries required to run the project are installed in the following steps:

- Python 3.12
- Anaconda

## Installation

First, clone the project and then create the environment using conda.

```bash
git clone <project-repo-url>
cd <project-location>
conda env create -f environment.yml
conda activate hybrid_chess_ai
```

After activating the environment, run the main script:

```bash
python -m src.gui.main
```

## Usage

- You can play against the AI through the interactive chess interface.
- It is possible to observe the internal workings of the AI by visualizing the MCTS search and displaying the probabilities of moves.