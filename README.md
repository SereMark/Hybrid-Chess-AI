# Hybrid Chess AI

Hybrid Chess AI is an end-to-end framework for building, training, evaluating, and deploying state‐of‑the‑art chess engines based on deep learning and reinforcement learning techniques. This project combines data preparation from raw PGN game files, supervised learning with a transformer-based architecture, self‑play reinforcement learning with Monte Carlo Tree Search (MCTS), extensive evaluation and benchmarking, hyperparameter optimization using Optuna, and finally deployment as a live Lichess bot. All of this is managed via an intuitive Streamlit dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Environment Setup](#environment-setup)
- [Usage Instructions](#usage-instructions)
  - [Launching the Dashboard](#launching-the-dashboard)
  - [Data Preparation](#data-preparation)
  - [Supervised Training](#supervised-training)
  - [Reinforcement Training](#reinforcement-training)
  - [Evaluation](#evaluation)
  - [Benchmarking](#benchmarking)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Lichess Bot Deployment](#lichess-bot-deployment)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

The Hybrid Chess AI project provides a full pipeline for developing competitive chess engines using both supervised and reinforcement learning approaches. It is designed to be modular, scalable, and highly configurable. Whether you are a researcher or a hobbyist, you can use this system to process raw chess data, experiment with various model architectures and training strategies, and finally deploy a live bot that competes on Lichess.

---

## Features

- **Data Preparation:**  
  Transform raw PGN files into a structured HDF5 dataset. Optionally, generate an opening book for enhanced performance.
  
- **Supervised Training:**  
  Train a custom transformer-based chess model using curated data. Includes advanced checkpointing and logging features.

- **Reinforcement Training:**  
  Enhance your model via self‑play combined with Monte Carlo Tree Search (MCTS) to simulate competitive games and generate additional training samples.

- **Evaluation & Benchmarking:**  
  Evaluate model performance on reserved test sets and run head-to-head benchmarks between different models.

- **Hyperparameter Optimization:**  
  Leverage Optuna for automated hyperparameter tuning with support for custom search spaces and advanced pruning strategies.

- **Lichess Bot Deployment:**  
  Seamlessly deploy your trained model as an active Lichess bot using the Berserk client, complete with support for opening books and MCTS.

- **Streamlit Dashboard:**  
  A unified dashboard for configuring, launching, and monitoring every component of the pipeline—no command‑line expertise required.

- **Experiment Tracking:**  
  Integrated support for Weights & Biases (WandB) to track experiments, visualize metrics, and log artifacts.

---

## Project Architecture

The repository is organized into modular components to ensure clarity and extensibility:

```
├── data/
│   ├── raw/                   # Raw PGN files and unprocessed data
│   └── processed/             # HDF5 dataset, indices, and opening book JSON
├── engine/                    # UCI chess engine executables (e.g., Stockfish)
├── models/
│   ├── checkpoints/           # Model checkpoints for supervised and reinforcement training
│   └── saved_models/          # Final saved model files
├── src/
│   ├── analysis/
│   │   ├── benchmark/         # Benchmarking scripts and bot utilities
│   │   └── evaluation/        # Model evaluation routines
│   ├── data_preperation/      # DataPreparationWorker – converts PGN files into datasets
│   ├── lichess_deployment/    # Lichess bot deployment and management scripts
│   ├── models/                # Model architecture definition (transformer-based chess model)
│   ├── training/
│   │   ├── hyperparameter_optimization/   # Hyperparameter tuning with Optuna
│   │   ├── reinforcement/     # Reinforcement learning routines (self-play + MCTS)
│   │   └── supervised/        # Supervised training routines
│   └── utils/                 # Utility modules (common functions, training utilities, etc...)
├── dashboard.py               # Streamlit dashboard to manage and monitor the entire pipeline
└── environment.yml            # Conda environment specification
```

All components are designed with consistent coding conventions and centralized utility functions for logging, checkpointing, and seed initialization.

---

## Environment Setup

Hybrid Chess AI is configured using a Conda environment. The environment is specified in the provided `environment.yml` file.

### Creating the Conda Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/hybrid_chess_ai.git
   cd hybrid_chess_ai
   ```

2. **Create the Conda environment:**

   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**

   ```bash
   conda activate hybrid_chess_ai
   ```

Now your environment is ready with all dependencies including PyTorch (with CUDA), NumPy, chess, Streamlit, WandB, Optuna, and more.

---

## Usage Instructions

The project is controlled via an interactive Streamlit dashboard. Below are detailed instructions for each component:

### Launching the Dashboard

Start the dashboard with the following command:

```bash
streamlit run dashboard.py
```

The dashboard will open in your web browser with separate tabs for data preparation, training, evaluation, benchmarking, hyperparameter optimization, and Lichess deployment.

---

### Data Preparation

- **Purpose:** Process raw PGN files to create an HDF5 dataset and optionally generate an opening book.
- **Configuration:**  
  - Provide paths to the raw PGN file and chess engine executable.
  - Configure filtering parameters (e.g., ELO range, game move thresholds).
  - Optionally select to generate an opening book using a secondary PGN file.
- **Output:**  
  - An HDF5 dataset containing board representations, policy targets, and value targets.
  - A JSON file for the opening book.
- **How to Run:**  
  Navigate to the **Data Preparation** tab on the dashboard, fill in the required inputs, and click **Start Data Preparation**.

---

### Supervised Training

- **Purpose:** Train the transformer-based chess model using supervised learning on the prepared dataset.
- **Configuration:**  
  - Set hyperparameters such as epochs, learning rate, batch size, optimizer type, scheduler type, and gradient clipping.
  - Provide paths to the dataset and indices files.
  - Optionally, resume training from an existing checkpoint.
- **Output:**  
  - A trained model saved to the `models/saved_models` directory.
  - Periodic checkpoints for recovery.
- **How to Run:**  
  Go to the **Supervised Trainer** tab, configure training parameters and dataset paths, then click **Start Supervised Training**.

---

### Reinforcement Training

- **Purpose:** Enhance the model using self-play and reinforcement learning with MCTS.
- **Configuration:**  
  - Set self-play parameters including the number of iterations, simulations per move, and learning rate.
  - Configure training-specific parameters such as accumulation steps and batch size.
- **Output:**  
  - Improved model checkpoints resulting from reinforcement training.
  - Logs and self-play game records.
- **How to Run:**  
  Use the **Reinforcement Trainer** tab to set the parameters and start the training process.

---

### Evaluation

- **Purpose:** Evaluate the performance of a trained model on a reserved test set.
- **Configuration:**  
  - Provide paths to the trained model, HDF5 dataset, and test indices.
  - Optionally enable WandB for logging advanced metrics.
- **Output:**  
  - Metrics including accuracy, loss statistics, confusion matrices, and advanced visualizations.
- **How to Run:**  
  In the **Evaluation** tab, set the required paths and click **Start Evaluation**.

---

### Benchmarking

- **Purpose:** Run head-to-head matches between two trained models to measure performance.
- **Configuration:**  
  - Specify paths for two model files.
  - Choose whether to use MCTS and/or an opening book for each bot.
  - Set the number of games for the benchmark.
- **Output:**  
  - Game logs, win/loss statistics, and performance trends.
- **How to Run:**  
  Navigate to the **Benchmarking** tab, enter the settings, and click **Start Benchmarking**.

---

### Hyperparameter Optimization

- **Purpose:** Use Optuna to automatically search for the best hyperparameters for supervised training.
- **Configuration:**  
  - Specify ranges for key hyperparameters such as learning rate, weight decay, batch size, epochs, momentum, and others.
  - Provide paths for the dataset and index files.
- **Output:**  
  - The best hyperparameter configuration stored in a results file.
  - Detailed logs of trial performance and metrics.
- **How to Run:**  
  Go to the **Hyperparameter Optimization** tab, configure the search ranges and dataset details, then click **Start Hyperparameter Optimization**.

---

### Lichess Bot Deployment

- **Purpose:** Deploy your trained model as a live Lichess bot.
- **Configuration:**  
  - Provide paths for the model and opening book JSON.
  - Enter your Lichess bot API token and select preferred time control and opponent rating range.
  - Configure options for using MCTS, auto‑resignation, and model‑based evaluation fallback.
- **Output:**  
  - A live bot that accepts challenges on Lichess.
  - Logging of game events and moves.
- **How to Run:**  
  In the **Lichess Deployment** tab, fill in the required fields and click **Deploy / Refresh Lichess Bot**.

---

## Dependencies

The project is managed via Conda using the provided `environment.yml`. Key dependencies include:

- **Python 3.11**  
- **PyTorch 2.5.1 with CUDA 12.4 support**  
- **NumPy & Matplotlib**  
- **python-chess & h5py**  
- **ONNX**  
- **Streamlit**  
- **Weights & Biases (wandb)**  
- **Optuna**  
- **Scikit-learn**  
- **SHAP**  
- **Berserk (via pip)**

All dependencies are installed automatically when you create the environment with Conda.

---

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **python‑chess:** Thanks for the robust chess engine interface.  
- **PyTorch & CUDA:** For the deep learning backbone.  
- **Streamlit:** For enabling rapid development of an intuitive dashboard.  
- **Optuna & WandB:** For providing excellent experiment tracking and hyperparameter tuning tools.  
- **Berserk:** For Lichess API integration.  

---

Hybrid Chess AI is a powerful tool for both academic research and competitive development in chess AI. Explore the dashboard, experiment with various settings, and build your own winning chess bot!
