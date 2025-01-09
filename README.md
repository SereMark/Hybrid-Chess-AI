# **Hybrid Chess AI: Combining Neural Networks, MCTS, and Opening Book**

## **Overview**

This repository implements a **Hybrid Chess AI** that integrates:
- A **Convolutional Neural Network (CNN)** for board evaluation (policy and value heads).
- **Monte Carlo Tree Search (MCTS)** for move selection in both self-play (reinforcement training) and live gameplay.
- An **opening book** for optimized early-game play.

A **PyQt5 Graphical User Interface (GUI)** is provided for:
- Interactive gameplay against the AI.
- Data preparation from PGN files.
- Supervised learning & reinforcement learning workflows.
- Model evaluation and performance benchmarking.
- Visualization of AI decision-making processes (loss curves, accuracy, etc.).

---

## **Key Features**

1. **Modular & Scalable Architecture**  
   - Clear separation of data preparation, training, evaluation, and inference modules.  
   - Allows easy extension or customization (e.g., plugging in different neural network architectures).

2. **Hybrid AI Approach**  
   - **Deep Learning** (via a CNN) for policy and value predictions.  
   - **Classical Chess Techniques** (MCTS + opening book) for searching the game tree effectively.

3. **Multiple Training Modes**  
   - **Supervised Training** using human-labeled game data (PGN files).  
   - **Reinforcement Learning** via self-play games, powered by MCTS.

4. **Visualization & GUI**  
   - Interactive PyQt5 interface for launching worker threads, monitoring progress, and plotting metrics.  
   - Real-time charts for loss, accuracy, confusion matrices, and other metrics.

5. **Flexible Configuration**  
   - Automatic detection of GPU for accelerated training if available.

6. **Educational & Research Focus**  
   - Demonstrates how neural networks, MCTS, and opening books can be combined in a chess engine.  
   - Designed for exploring advanced AI methods in a well-known domain.

---

## **System Requirements**

- **Git** - to clone the repository
- **Anaconda** - for isolated environment setup  

---

## **Installation and Setup**

1. **Clone the Repository**  
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create & Activate the Conda Environment**  
   ```bash
   conda env create -f environment.yml
   conda activate hybrid_chess_ai
   ```

3. **Launch the GUI Application**  
   ```bash
   python -m src.main
   ```
   This opens the PyQt5 interface where you can configure data preparation, run training, evaluate models, or make the AI play against other AI.

---

### **Key Logic**

- **`src.data_processing`**  
  - **`data_preparation_worker.py`**: Processes PGN files into HDF5 datasets for supervised training.  
  - **`opening_book_worker.py`**: Builds an opening book from PGN data.

- **`src.training`**  
  - **`supervised_training_worker.py`**: Runs supervised learning on the processed HDF5 dataset.  
  - **`reinforcement_training_worker.py`**: Performs self-play with MCTS, storing generated data for training.

- **`src.analysis`**  
  - **`evaluation_worker.py`**: Evaluates trained models using hold-out datasets.  
  - **`benchmark_worker.py`**: Conducts matchups between two engines/bots (can be CNN-based or external engines).

---

## **License**

This project is licensed under the [MIT License](LICENSE).  
Please see the `LICENSE` file for details.