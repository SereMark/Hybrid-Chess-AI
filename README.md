# **Hybrid Chess AI using CNN, MCTS, and Opening Book**

## **Overview**

This project is the implementation of a **hybrid chess artificial intelligence** designed to combine advanced deep learning methods and classical search algorithms. The AI utilizes a **Convolutional Neural Network (CNN)** for board evaluation, **Monte Carlo Tree Search (MCTS)** for move selection, and an **opening book** for optimized early-game play. It also features a **Graphical User Interface (GUI)** for interactive gameplay.

### **Purpose**
This project is developed as part of a thesis and serves as an educational and research tool to explore the intersection of neural networks and traditional chess AI techniques.

### **Disclaimer**
The code in this repository is for **non-commercial use only**.  
It **may not be used, reproduced, or distributed for any commercial purposes**, and it **may not be used in academic works, such as theses or research papers, without prior written consent from the author**. Please refer to the [LICENSE](LICENSE) file for details.

---

## **Features**
- **AI Gameplay**: Play against the hybrid chess AI and test its strategic capabilities.
- **Visualization Tools**: Observe the MCTS search process and the probabilities of candidate moves.
- **User-Friendly GUI**: Intuitive interface for playing, analyzing, and interacting with the AI.

---

## **System Requirements**
To run this project, you need:
- **Python 3.12**
- **Anaconda** (for dependency management)
- A modern GPU (recommended for optimal performance with neural networks)

---

## **Installation**

Follow these steps to set up the environment and run the project:

1. **Clone the Repository**:
   ```bash
   git clone <project-repo-url>
   cd <project-location>
   ```

2. **Create and Activate the Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate hybrid_chess_ai
   ```

3. **Run the Application**:
   Launch the main script to start the GUI:
   ```bash
   python -m src.gui.main
   ```

---

## **Usage**

### **Playing Chess**
- Launch the GUI to play chess against the AI. The interface supports interactive gameplay, showing move suggestions and AI thinking processes.

### **Visualizing AI Behavior**
- View the internal operations of the AI, such as:
  - **MCTS Search Tree**: Explore how the AI evaluates potential moves.
  - **Move Probabilities**: See how the CNN assigns probabilities to possible moves.

---

## **License**
This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode), with the following **Additional Terms**:

1. **Academic Integrity**: This code may not be used, presented, or published in any academic work, including theses or research papers, without prior written consent from the author.
2. **No Warranty or Liability**: The code is provided "as is," with no guarantees of functionality or reliability.

For more details, see the [LICENSE](LICENSE) file.
