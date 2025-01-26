import os
import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime
from src.data_processing.data_preparation.data_preparation_worker import DataPreparationWorker
from src.data_processing.opening_book.opening_book_worker import OpeningBookWorker
from src.training.supervised.supervised_training_worker import SupervisedWorker
from src.training.reinforcement.reinforcement_training_worker import ReinforcementWorker
from src.analysis.evaluation.evaluation_worker import EvaluationWorker
from src.analysis.benchmark.benchmark_worker import BenchmarkWorker

st.set_page_config(page_title="Chess AI Management Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.section-header {font-size:24px;font-weight:bold;margin-bottom:10px;}
.stButton>button {width:100%;}
.stCheckbox > div > label {font-size:16px;}
.badge {padding:0.5em 1em;border-radius:0.5em;color:white;font-weight:bold;font-size:14px;}
.badge-completed {background-color:#28a745;}
.badge-in-progress {background-color:#ffc107;}
.badge-failed {background-color:#dc3545;}
.log-output {background-color:#2c2c2c;color:#ffffff;padding:1em;border-radius:0.5em;height:500px;overflow-y:scroll;white-space:pre-wrap;font-family:monospace;font-size:14px;}
.log-output::-webkit-scrollbar {width:8px;}
.log-output::-webkit-scrollbar-track {background:#1e1e1e;}
.log-output::-webkit-scrollbar-thumb {background-color:#555;border-radius:4px;border:2px solid #2c2c2c;}
</style>
""", unsafe_allow_html=True)

if 'logs' not in st.session_state:
    st.session_state['logs'] = {}

def validate_file_path(file_path, file_type="file"):
    return os.path.isfile(file_path) if file_type == "file" else os.path.isdir(file_path) if file_type == "directory" else False

def show_validation_message(is_valid, message):
    st.markdown(f"✅ **{message}**" if is_valid else f"⚠️ **{message}**")

def add_log(process, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state['logs'].setdefault(process, []).append(log_entry)

def run_data_preparation_worker():
    st.header("📂 Data Preparation")
    with st.expander("🛠️ Configure Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            raw_pgn_file = st.text_input("Path to Raw PGN File:", value="data/raw/lichess_db_standard_rated_2024-12.pgn", placeholder="e.g., data/raw/lichess_db_standard_rated_2024-12.pgn", help="Enter the path to the raw PGN file for data preparation.")
            if raw_pgn_file:
                is_valid = validate_file_path(raw_pgn_file)
                show_validation_message(is_valid, "Valid PGN file path." if is_valid else "Invalid PGN file path.")
        with col2:
            engine_path = st.text_input("Path to Chess Engine:", value="engine/stockfish/stockfish-windows-x86-64-avx2.exe", placeholder="e.g., engine/stockfish/stockfish-windows-x86-64-avx2.exe", help="Enter the path to the chess engine executable.")
            if engine_path:
                is_valid = validate_file_path(engine_path)
                show_validation_message(is_valid, "Valid Engine path." if is_valid else "Invalid Engine path.")
            max_games = st.slider("Max Games to Process:", 100, 5000, 1000, help="Set the maximum number of games to process.")
            min_elo = st.slider("Minimum Player ELO:", 0, 3000, 1600, help="Set the minimum ELO rating of players to include.")
        col3, col4 = st.columns(2)
        with col3:
            batch_size = st.number_input("Batch Size:", 1, 10000, 128, step=1, help="Number of games to process in each batch.")
        with col4:
            engine_depth = st.slider("Engine Depth:", 1, 30, 20, help="Set the search depth for the chess engine.")
            engine_threads = st.slider("Engine Threads:", 1, 8, 4, help="Set the number of threads for the chess engine.")
            engine_hash = st.slider("Engine Hash (MB):", 128, 2048, 512, step=128, help="Set the hash size (MB) for the chess engine.")
    submitted = st.button("Start Data Preparation 🏁")
    if submitted:
        if validate_file_path(raw_pgn_file) and validate_file_path(engine_path):
            add_log("Data Preparation", "Data preparation started.")
            progress = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()
            def progress_callback(percent):
                progress.progress(int(percent))
            def status_callback(message):
                status_text.text(message)
                add_log("Data Preparation", message)
                log_container.markdown("<div class='log-output'>" + "\n".join(st.session_state['logs']['Data Preparation']) + "</div>", unsafe_allow_html=True)
            try:
                worker = DataPreparationWorker(raw_pgn_file, max_games, min_elo, batch_size, engine_path, engine_depth, engine_threads, engine_hash, progress_callback, status_callback)
                result = worker.run()
                if result:
                    progress.progress(100)
                    status_text.text("🎉 Data Preparation Completed!")
                    add_log("Data Preparation", "Data preparation completed successfully.")
                else:
                    status_text.text("⚠️ Processing failed. Please check your inputs.")
                    add_log("Data Preparation", "Data preparation failed.")
            except Exception as e:
                status_text.text(f"⚠️ An unexpected error occurred: {e}")
                add_log("Data Preparation", f"Error: {e}")
        else:
            st.error("⚠️ Invalid PGN file or engine path. Please check and try again.")

def run_opening_book_worker():
    st.header("📖 Opening Book Generator")
    with st.expander("🛠️ Configure Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            pgn_file_path = st.text_input("Path to PGN File:", value="data/raw/lichess_db_standard_rated_2024-12.pgn", placeholder="e.g., data/raw/lichess_db_standard_rated_2024-12.pgn", help="Enter the path to the PGN file for generating the opening book.")
            if pgn_file_path:
                is_valid = validate_file_path(pgn_file_path)
                show_validation_message(is_valid, "Valid PGN file path." if is_valid else "Invalid PGN file path.")
        with col2:
            max_games = st.slider("Max Games to Process:", 100, 5000, 1000, help="Set the maximum number of games to process.")
            min_elo = st.slider("Minimum Player ELO:", 0, 3000, 1600, help="Set the minimum ELO rating of players to include.")
        col3, _ = st.columns(2)
        with col3:
            max_opening_moves = st.slider("Max Opening Moves:", 1, 30, 10, help="Set the maximum number of opening moves to include.")
    submitted = st.button("Start Opening Book Generation 🏁")
    if submitted:
        if validate_file_path(pgn_file_path):
            add_log("Opening Book Generation", "Opening book generation started.")
            progress = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()
            def progress_callback(percent):
                progress.progress(int(percent))
            def status_callback(message):
                status_text.text(message)
                add_log("Opening Book Generation", message)
                log_container.markdown("<div class='log-output'>" + "\n".join(st.session_state['logs']['Opening Book Generation']) + "</div>", unsafe_allow_html=True)
            try:
                worker = OpeningBookWorker(pgn_file_path, max_games, min_elo, max_opening_moves, progress_callback, status_callback)
                result = worker.run()
                if result:
                    progress.progress(100)
                    status_text.text("🎉 Opening Book Generation Completed!")
                    add_log("Opening Book Generation", "Opening book generation completed successfully.")
                else:
                    status_text.text("⚠️ Processing failed. Please check your inputs.")
                    add_log("Opening Book Generation", "Opening book generation failed.")
            except Exception as e:
                status_text.text(f"⚠️ An unexpected error occurred: {e}")
                add_log("Opening Book Generation", f"Error: {e}")
        else:
            st.error("⚠️ Invalid PGN file path. Please check and try again.")

def run_supervised_training_worker():
    st.header("🎓 Supervised Trainer")
    with st.expander("🛠️ Configure Training Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs:", 1, 1000, 10, step=1, help="Number of training epochs.")
            batch_size = st.number_input("Batch Size:", 1, 10000, 128, step=1, help="Number of samples per training batch.")
            learning_rate = st.text_input("Learning Rate:", "0.0001", placeholder="e.g., 0.0001", help="Set the learning rate for the optimizer.")
            weight_decay = st.text_input("Weight Decay:", "0.0001", placeholder="e.g., 0.0001", help="Set the weight decay (L2 penalty) for the optimizer.")
            optimizer_type = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"], help="Choose the optimizer for training.")
        with col2:
            scheduler_type = st.selectbox("Scheduler Type:", ["cosineannealingwarmrestarts", "cosineannealing", "steplr", "exponentiallr", "onelr", "none"], help="Choose the learning rate scheduler.")
            num_workers = st.number_input("Number of Worker Threads:", 1, 32, 4, step=1, help="Number of worker threads for data loading.")
            random_seed = st.number_input("Random Seed:", 0, 100000, 42, step=1, help="Set the random seed for reproducibility.")
    with st.expander("📁 Dataset Details", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            dataset_path = st.text_input("Path to Dataset:", "data/processed/dataset.h5", placeholder="e.g., data/processed/dataset.h5", help="Enter the path to the processed dataset.")
            if dataset_path:
                is_valid = validate_file_path(dataset_path)
                show_validation_message(is_valid, "Valid Dataset path." if is_valid else "Invalid Dataset path.")
        with col4:
            train_indices_path = st.text_input("Path to Train Indices:", "data/processed/train_indices.npy", placeholder="e.g., data/processed/train_indices.npy", help="Enter the path to the training indices.")
            if train_indices_path:
                is_valid = validate_file_path(train_indices_path)
                show_validation_message(is_valid, "Valid Train Indices path." if is_valid else "Invalid Train Indices path.")
            val_indices_path = st.text_input("Path to Validation Indices:", "data/processed/val_indices.npy", placeholder="e.g., data/processed/val_indices.npy", help="Enter the path to the validation indices.")
            if val_indices_path:
                is_valid = validate_file_path(val_indices_path)
                show_validation_message(is_valid, "Valid Validation Indices path." if is_valid else "Invalid Validation Indices path.")
    with st.expander("🔗 Model Options", expanded=True):
        col5, col6 = st.columns(2)
        with col5:
            model_path = st.text_input("Path to Pretrained Model (optional):", "", placeholder="e.g., models/pretrained_model.pth", help="Enter the path to a pretrained model if available.")
            if model_path:
                is_valid = validate_file_path(model_path)
                show_validation_message(is_valid, "Valid Model path." if is_valid else "Invalid Model path.")
        with col6:
            save_checkpoints = st.checkbox("Save Checkpoints", True, help="Enable to save model checkpoints during training.")
            checkpoint_type = None
            checkpoint_interval = None
            if save_checkpoints:
                checkpoint_type = st.selectbox("Checkpoint Type:", ["epoch", "time"], help="Type of checkpoint interval.")
                if checkpoint_type == "epoch":
                    checkpoint_interval = st.number_input("Checkpoint Interval (Epochs):", 1, 1000, 5, step=1, help="Interval for saving checkpoints based on epochs.")
                else:
                    checkpoint_interval = st.number_input("Checkpoint Interval (Minutes):", 1, 10000, 60, step=1, help="Interval for saving checkpoints based on time.")
    submitted = st.button("Start Supervised Training 🏁")
    if submitted:
        required_files = [dataset_path, train_indices_path, val_indices_path]
        if model_path:
            required_files.append(model_path)
        missing_files = [f for f in required_files if f and not validate_file_path(f)]
        if not missing_files:
            add_log("Supervised Training", "Supervised training started.")
            progress = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()
            def progress_callback(percent):
                progress.progress(int(percent))
            def status_callback(message):
                status_text.text(message)
                add_log("Supervised Training", message)
                log_container.markdown("<div class='log-output'>" + "\n".join(st.session_state['logs']['Supervised Training']) + "</div>", unsafe_allow_html=True)
            try:
                lr = float(learning_rate)
                wd = float(weight_decay)
                worker = SupervisedWorker(
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    learning_rate=lr,
                    weight_decay=wd,
                    save_checkpoints=save_checkpoints,
                    checkpoint_interval=int(checkpoint_interval) if save_checkpoints else None,
                    dataset_path=dataset_path,
                    train_indices_path=train_indices_path,
                    val_indices_path=val_indices_path,
                    model_path=model_path if model_path else None,
                    checkpoint_type=checkpoint_type if save_checkpoints else None,
                    optimizer_type=optimizer_type,
                    scheduler_type=scheduler_type,
                    num_workers=int(num_workers),
                    random_seed=int(random_seed),
                    progress_callback=progress_callback,
                    status_callback=status_callback
                )
                result = worker.run()
                if result:
                    progress.progress(100)
                    status_text.text("🎉 Supervised Training Completed!")
                    add_log("Supervised Training", "Supervised training completed successfully.")
                else:
                    status_text.text("⚠️ Supervised Training failed. Please check the logs.")
                    add_log("Supervised Training", "Supervised training failed.")
            except ValueError:
                st.error("⚠️ Learning Rate and Weight Decay must be valid numbers.")
            except Exception as e:
                status_text.text(f"⚠️ An unexpected error occurred: {e}")
                add_log("Supervised Training", f"Error: {e}")
        else:
            st.error(f"⚠️ The following required files are missing: {', '.join(missing_files)}. Please check the paths and try again.")

def run_reinforcement_training_worker():
    st.header("🛡️ Reinforcement Trainer")
    with st.expander("🛠️ Configure Training Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            num_iterations = st.number_input("Number of Iterations:", 1, 1000, 10, step=1, help="Number of reinforcement training iterations.")
            num_games_per_iteration = st.number_input("Games per Iteration:", 1, 10000, 1000, step=100, help="Number of games to play in each iteration.")
            simulations = st.number_input("Simulations per Move:", 1, 10000, 100, step=10, help="Number of MCTS simulations per move.")
            c_puct = st.number_input("C_PUCT:", 0.0, 10.0, 1.4, step=0.1, format="%.1f", help="Exploration constant for MCTS.")
            temperature = st.number_input("Temperature:", 0.0, 10.0, 1.0, step=0.1, format="%.1f", help="Temperature parameter for move selection.")
        with col2:
            num_epochs = st.number_input("Epochs per Iteration:", 1, 1000, 5, step=1, help="Number of training epochs per iteration.")
            batch_size = st.number_input("Batch Size:", 1, 10000, 128, step=1, help="Number of samples per training batch.")
            num_threads = st.number_input("Number of Threads:", 1, 32, 4, step=1, help="Number of threads for parallel processing.")
            learning_rate = st.text_input("Learning Rate:", "0.0001", placeholder="e.g., 0.0001", help="Set the learning rate for the optimizer.")
            weight_decay = st.text_input("Weight Decay:", "0.0001", placeholder="e.g., 0.0001", help="Set the weight decay (L2 penalty) for the optimizer.")
            optimizer_type = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"], help="Choose the optimizer for training.")
            scheduler_type = st.selectbox("Scheduler Type:", ["cosineannealingwarmrestarts", "cosineannealing", "steplr", "exponentiallr", "onelr", "none"], help="Choose the learning rate scheduler.")
    with st.expander("🔗 Model Options", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            model_path = st.text_input("Path to Existing Model (optional):", "", placeholder="e.g., models/supervised_model.pth", help="Enter the path to an existing model if available.")
            if model_path:
                is_valid = validate_file_path(model_path)
                show_validation_message(is_valid, "Valid Model path." if is_valid else "Invalid Model path.")
        with col4:
            save_checkpoints = st.checkbox("Save Checkpoints", True, help="Enable to save model checkpoints during training.")
            checkpoint_type = None
            checkpoint_interval = None
            if save_checkpoints:
                checkpoint_type = st.selectbox("Checkpoint Type:", ["iteration", "time"], help="Type of checkpoint interval.")
                if checkpoint_type == "iteration":
                    checkpoint_interval = st.number_input("Checkpoint Interval (Iterations):", 1, 1000, 5, step=1, help="Interval for saving checkpoints based on iterations.")
                else:
                    checkpoint_interval = st.number_input("Checkpoint Interval (Minutes):", 1, 10000, 60, step=1, help="Interval for saving checkpoints based on time.")
    random_seed = st.number_input("Random Seed:", 0, 100000, 42, step=1, help="Set the random seed for reproducibility.")
    submitted = st.button("Start Reinforcement Training 🏁")
    if submitted:
        if model_path and not validate_file_path(model_path):
            st.error("⚠️ Model path is invalid or the file does not exist. Please check and try again.")
        else:
            add_log("Reinforcement Training", "Reinforcement training started.")
            progress = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()
            def progress_callback(percent):
                progress.progress(int(percent))
            def status_callback(message):
                status_text.text(message)
                add_log("Reinforcement Training", message)
                log_container.markdown("<div class='log-output'>" + "\n".join(st.session_state['logs']['Reinforcement Training']) + "</div>", unsafe_allow_html=True)
            try:
                lr = float(learning_rate)
                wd = float(weight_decay)
                worker = ReinforcementWorker(
                    model_path=model_path if model_path else None,
                    num_iterations=int(num_iterations),
                    num_games_per_iteration=int(num_games_per_iteration),
                    simulations=int(simulations),
                    c_puct=float(c_puct),
                    temperature=float(temperature),
                    num_epochs=int(num_epochs),
                    batch_size=int(batch_size),
                    num_threads=int(num_threads),
                    save_checkpoints=save_checkpoints,
                    checkpoint_interval=int(checkpoint_interval) if save_checkpoints else None,
                    checkpoint_type=checkpoint_type if save_checkpoints else None,
                    random_seed=int(random_seed),
                    optimizer_type=optimizer_type,
                    learning_rate=lr,
                    weight_decay=wd,
                    scheduler_type=scheduler_type,
                    progress_callback=progress_callback,
                    status_callback=status_callback
                )
                result = worker.run()
                if result:
                    progress.progress(100)
                    status_text.text("🎉 Reinforcement Training Completed!")
                    add_log("Reinforcement Training", "Reinforcement training completed successfully.")
                else:
                    status_text.text("⚠️ Reinforcement Training failed. Please check the logs.")
                    add_log("Reinforcement Training", "Reinforcement training failed.")
            except ValueError:
                st.error("⚠️ Learning Rate and Weight Decay must be valid numbers.")
            except Exception as e:
                status_text.text(f"⚠️ An unexpected error occurred: {e}")
                add_log("Reinforcement Training", f"Error: {e}")

def run_evaluation_worker():
    st.header("📈 Evaluation")
    with st.expander("🛠️ Configure Evaluation Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            model_path = st.text_input("Path to Trained Model:", "models/saved_models/supervised_model.pth", placeholder="e.g., models/saved_models/supervised_model.pth", help="Enter the path to the trained model for evaluation.")
            if model_path:
                is_valid = validate_file_path(model_path)
                show_validation_message(is_valid, "Valid Model path." if is_valid else "Invalid Model path.")
        with col2:
            dataset_indices_path = st.text_input("Path to Dataset Indices:", "data/processed/test_indices.npy", placeholder="e.g., data/processed/test_indices.npy", help="Enter the path to the dataset indices for evaluation.")
            if dataset_indices_path:
                is_valid = validate_file_path(dataset_indices_path)
                show_validation_message(is_valid, "Valid Dataset Indices path." if is_valid else "Invalid Dataset Indices path.")
        h5_file_path = st.text_input("Path to H5 Dataset:", "data/processed/dataset.h5", placeholder="e.g., data/processed/dataset.h5", help="Enter the path to the H5 dataset file.")
        if h5_file_path:
            is_valid = validate_file_path(h5_file_path)
            show_validation_message(is_valid, "Valid H5 Dataset path." if is_valid else "Invalid H5 Dataset path.")
    submitted = st.button("Start Evaluation 🏁")
    if submitted:
        required_files = [model_path, dataset_indices_path, h5_file_path]
        missing_files = [f for f in required_files if f and not validate_file_path(f)]
        if not missing_files:
            add_log("Evaluation", "Evaluation started.")
            progress = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()
            def progress_callback(percent):
                progress.progress(int(percent))
            def status_callback(message):
                status_text.text(message)
                add_log("Evaluation", message)
                log_container.markdown("<div class='log-output'>" + "\n".join(st.session_state['logs']['Evaluation']) + "</div>", unsafe_allow_html=True)
            try:
                worker = EvaluationWorker(model_path, dataset_indices_path, h5_file_path, progress_callback, status_callback)
                metrics = worker.run()
                if metrics:
                    progress.progress(100)
                    status_text.text("🎉 Evaluation Completed!")
                    add_log("Evaluation", "Evaluation completed successfully.")
                else:
                    status_text.text("⚠️ Evaluation failed. Please check the logs.")
                    add_log("Evaluation", "Evaluation failed.")
            except Exception as e:
                status_text.text(f"⚠️ An unexpected error occurred: {e}")
                add_log("Evaluation", f"Error: {e}")
        else:
            st.error(f"⚠️ The following required files are missing: {', '.join(missing_files)}. Please check the paths and try again.")

def run_benchmark_worker():
    st.header("🏆 Benchmarking")
    with st.expander("🛠️ Configure Benchmarking Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            bot1_path = st.text_input("Path to Bot1 Model:", "models\saved_models\supervised_model.pth", placeholder="e.g., models\saved_models\supervised_model.pth", help="Enter the path to Bot1's model.")
            if bot1_path:
                is_valid = validate_file_path(bot1_path)
                show_validation_message(is_valid, "Valid Bot1 model path." if is_valid else "Invalid Bot1 model path.")
            bot1_use_mcts = st.checkbox("Bot1 Use MCTS", True, help="Enable MCTS for Bot1.")
            bot1_use_opening_book = st.checkbox("Bot1 Use Opening Book", True, help="Enable Opening Book for Bot1.")
        with col2:
            bot2_path = st.text_input("Path to Bot2 Model:", "models\saved_models\supervised_model.pth", placeholder="e.g., models\saved_models\supervised_model.pth", help="Enter the path to Bot2's model.")
            if bot2_path:
                is_valid = validate_file_path(bot2_path)
                show_validation_message(is_valid, "Valid Bot2 model path." if is_valid else "Invalid Bot2 model path.")
            bot2_use_mcts = st.checkbox("Bot2 Use MCTS", True, help="Enable MCTS for Bot2.")
            bot2_use_opening_book = st.checkbox("Bot2 Use Opening Book", True, help="Enable Opening Book for Bot2.")
        num_games = st.number_input("Number of Games:", 1, 10000, 100, step=1, help="Total number of games to be played in the benchmark.")
    submitted = st.button("Start Benchmarking 🏁")
    if submitted:
        required_files = [bot1_path, bot2_path]
        missing_files = [f for f in required_files if f and not validate_file_path(f)]
        if not missing_files:
            add_log("Benchmarking", "Benchmarking started.")
            progress = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()
            def progress_callback(percent):
                progress.progress(int(percent))
            def status_callback(message):
                status_text.text(message)
                add_log("Benchmarking", message)
                log_container.markdown("<div class='log-output'>" + "\n".join(st.session_state['logs']['Benchmarking']) + "</div>", unsafe_allow_html=True)
            try:
                worker = BenchmarkWorker(bot1_path, bot2_path, int(num_games), bot1_use_mcts, bot1_use_opening_book, bot2_use_mcts, bot2_use_opening_book, progress_callback, status_callback)
                metrics = worker.run()
                if metrics:
                    progress.progress(100)
                    status_text.text("🎉 Benchmarking Completed!")
                    add_log("Benchmarking", "Benchmarking completed successfully.")
                    if "results" in metrics and isinstance(metrics["results"], dict):
                        results_df = pd.DataFrame(list(metrics["results"].items()), columns=["Bot", "Wins"])
                        st.bar_chart(results_df.set_index("Bot"))
                else:
                    status_text.text("⚠️ Benchmarking failed. Please check the logs.")
                    add_log("Benchmarking", "Benchmarking failed.")
            except Exception as e:
                status_text.text(f"⚠️ An unexpected error occurred: {e}")
                add_log("Benchmarking", f"Error: {e}")
        else:
            st.error(f"⚠️ The following required files are missing: {', '.join(missing_files)}. Please check the paths and try again.")

if __name__ == "__main__":
    st.sidebar.title("🔧 Navigation")
    app_mode = st.sidebar.radio("Choose the section:", [
        "Data Preparation", 
        "Opening Book Generator", 
        "Supervised Trainer", 
        "Reinforcement Trainer", 
        "Evaluation", 
        "Benchmarking"
    ])
    if app_mode == "Data Preparation":
        run_data_preparation_worker()
    elif app_mode == "Opening Book Generator":
        run_opening_book_worker()
    elif app_mode == "Supervised Trainer":
        run_supervised_training_worker()
    elif app_mode == "Reinforcement Trainer":
        run_reinforcement_training_worker()
    elif app_mode == "Evaluation":
        run_evaluation_worker()
    elif app_mode == "Benchmarking":
        run_benchmark_worker()
    else:
        st.error("Unknown section selected!")