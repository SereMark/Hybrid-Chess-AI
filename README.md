# **Hybrid Chess AI: Combining Neural Networks, MCTS, and Opening Book**

## **Project Overview**

This repository contains the implementation of a **Hybrid Chess AI** that merges advanced machine learning techniques with traditional chess algorithms. The AI integrates a **Convolutional Neural Network (CNN)** for board evaluation, **Monte Carlo Tree Search (MCTS)** for move selection in both self-play training and gameplay against users, and an **opening book** for optimized early-game play. 

The application is equipped with a **Graphical User Interface (GUI)** built using PyQt5, enabling interactive gameplay, model training, evaluation, and visualization of AI decision-making processes.

---

## **Key Features**

1. **Modular Design**: 
   - Structured into distinct components, such as data preparation, supervised training, self-play training, evaluation, and gameplay.
   - Scalable architecture for future enhancements and feature additions.

2. **Hybrid Approach**:
   - Combines deep learning and classical algorithms for improved decision-making.

3. **Advanced Use of MCTS**:
   - MCTS powers **AI decision-making during self-play training**, enabling reinforcement learning through simulated games.
   - **During gameplay against users**, MCTS evaluates moves dynamically, providing robust and adaptive strategies.

4. **Visualization Tools**:
   - Real-time charts and plots to monitor training, evaluation, and gameplay dynamics.

5. **Interactive GUI**:
   - User-friendly interface to train models, configure parameters, and play against the AI.

6. **Educational Focus**:
   - Developed for research and education, ideal for exploring chess AI methodologies.

---

## **System Requirements**

- **Python**: 3.12+
- **Anaconda**: For managing dependencies

---

## **Installation and Setup**

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set Up the Environment**:
   Create a conda environment with the required dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate hybrid_chess_ai
   ```

3. **Run the Application**:
   Launch the GUI:
   ```bash
   python -m src.main
   ```

---

## **Application Architecture**

### **1. Main Application**
- Initializes the main GUI window and integrates tabs for all functionalities.
- Tabs include **data preparation**, **training**, **evaluation**, and **gameplay**.

### **2. Core Modules**
- **Data Preparation**: Prepares datasets from chess game data, including PGN parsing and opening book generation.
- **Supervised Training**: Trains neural networks using labeled datasets.
- **Self-Play Training**: Implements reinforcement learning through self-play games powered by MCTS.
- **Evaluation**: Assesses model performance with test datasets.
- **Gameplay**: Allows users to play against the AI, utilizing MCTS for move selection.

### **3. Visualization**
- Provides real-time insights via charts and plots, such as:
  - Training metrics (loss, accuracy)
  - Evaluation results (confusion matrix, precision, recall)
  - Gameplay analysis (material balance, move evaluations)

### **4. Core Logic**
- **Model**: Implements a CNN architecture with policy and value heads.
- **MCTS**: Integrates Monte Carlo Tree Search for both self-play training and real-time gameplay decisions.
- **Opening Book**: Provides optimized moves for the early game.

---

## **Usage**

### **1. Data Preparation**
Prepare data for training:
- Input: Chess PGN files
- Output: Processed datasets and opening books

### **2. Training**
Two modes of training are available:
- **Supervised Training**: Train on pre-labeled datasets.
- **Self-Play Training**: Reinforcement learning through self-play using MCTS for move selection.

### **3. Evaluation**
Evaluate trained models using test datasets:
- Metrics include accuracy, precision, recall, and confusion matrix visualization.

### **4. Gameplay**
Play chess against the AI:
- Interactive chessboard with move highlighting and real-time AI evaluations powered by MCTS.

---

## **License and Disclaimer**

- **License**: This project is licensed under a custom license for **non-commercial use only**.
- **Disclaimer**: The code in this repository may not be used for commercial purposes or academic publications without explicit permission. Refer to the [LICENSE](LICENSE) file for detailed terms.
