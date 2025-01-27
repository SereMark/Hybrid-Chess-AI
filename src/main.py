import os, warnings, streamlit as st
from src.data_processing.data_preparation.data_preparation_worker import DataPreparationWorker
from src.data_processing.opening_book.opening_book_worker import OpeningBookWorker
from src.training.supervised.supervised_training_worker import SupervisedWorker
from src.training.reinforcement.reinforcement_training_worker import ReinforcementWorker
from src.analysis.evaluation.evaluation_worker import EvaluationWorker
from src.analysis.benchmark.benchmark_worker import BenchmarkWorker
warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext!")

st.set_page_config(page_title="Chess AI Management Dashboard", layout="wide", initial_sidebar_state="expanded")

def validate_path(path, type_="file"):
    return os.path.isfile(path) if type_ == "file" else os.path.isdir(path) if type_ == "directory" else False

def input_with_validation(label, default, help_text, type_="file"):
    path = st.text_input(label, default, help=help_text)
    if path:
        valid = validate_path(path, type_)
        st.markdown(f"✅ **Valid {label.split()[2]} path.**" if valid else f"⚠️ **Invalid {label.split()[2]} path.**")
    return path

def execute_worker(create_worker):
    progress = st.progress(0)
    status = st.empty()
    try:
        worker = create_worker(lambda p: progress.progress(int(p)), lambda m: status.text(m))
        status.text("🚀 Started!")
        result = worker.run()
        if result:
            progress.progress(100)
            status.text("🎉 Completed!")
        else:
            status.text("⚠️ Failed.")
    except Exception as e:
        status.text(f"⚠️ Error: {e}")

def run_data_preparation_worker():
    st.header("📂 Data Preparation")
    with st.expander("🛠️ Configure Parameters", True):
        col1, col2 = st.columns(2)
        with col1:
            raw_pgn = input_with_validation("Path to Raw PGN File:", "data/raw/lichess_db_standard_rated_2024-12.pgn", "Path to raw PGN file.", "file")
        with col2:
            engine = input_with_validation("Path to Chess Engine:", "engine/stockfish/stockfish-windows-x86-64-avx2.exe", "Path to chess engine.", "file")
            max_games = st.slider("Max Games:", 100, 5000, 1000)
            min_elo = st.slider("Min ELO:", 0, 3000, 1600)
        col3, col4 = st.columns(2)
        with col3:
            batch_size = st.number_input("Batch Size:", 1, 10000, 128, step=1)
        with col4:
            engine_depth = st.slider("Engine Depth:", 1, 30, 20)
            engine_threads = st.slider("Engine Threads:", 1, 8, 4)
            engine_hash = st.slider("Engine Hash (MB):", 128, 2048, 512, step=128)
    if st.button("Start Data Preparation 🏁"):
        if validate_path(raw_pgn, "file") and validate_path(engine, "file"):
            execute_worker(lambda pc, sc: DataPreparationWorker(raw_pgn, max_games, min_elo, batch_size, engine, engine_depth, engine_threads, engine_hash, pc, sc))
        else:
            st.error("⚠️ Invalid PGN file or engine path.")

def run_opening_book_worker():
    st.header("📖 Opening Book Generator")
    with st.expander("🛠️ Configure Parameters", True):
        col1, col2 = st.columns(2)
        with col1:
            pgn = input_with_validation("Path to PGN File:", "data/raw/lichess_db_standard_rated_2024-12.pgn", "Path to PGN file.", "file")
        with col2:
            max_games = st.slider("Max Games:", 100, 5000, 1000)
            min_elo = st.slider("Min ELO:", 0, 3000, 1600)
            max_moves = st.slider("Max Opening Moves:", 1, 30, 10)
    if st.button("Start Opening Book Generation 🏁"):
        if validate_path(pgn, "file"):
            execute_worker(lambda pc, sc: OpeningBookWorker(pgn, max_games, min_elo, max_moves, pc, sc))
        else:
            st.error("⚠️ Invalid PGN file path.")

def run_supervised_training_worker():
    st.header("🎓 Supervised Trainer")
    with st.expander("🛠️ Configure Training Parameters", True):
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs:", 1, 1000, 10)
            batch_size = st.number_input("Batch Size:", 1, 10000, 128)
            accumulation_steps = st.number_input("Accumulation Steps:", 1, 100, 3)
            lr = st.text_input("Learning Rate:", "0.0001")
            wd = st.text_input("Weight Decay:", "0.0001")
            optimizer = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"])
        with col2:
            scheduler = st.selectbox("Scheduler Type:", ["cosineannealingwarmrestarts", "cosineannealing", "steplr", "exponentiallr", "onelr", "none"])
            num_workers = st.number_input("Worker Threads:", 1, 32, 4)
            random_seed = st.number_input("Random Seed:", 0, 100000, 42)
            policy_weight = st.number_input("Policy Weight:", 0.0, 10.0, 1.0, step=0.1)
            value_weight = st.number_input("Value Weight:", 0.0, 10.0, 2.0, step=0.1)
    with st.expander("📁 Dataset Details", True):
        col3, col4 = st.columns(2)
        with col3:
            dataset = input_with_validation("Path to Dataset:", "data/processed/dataset.h5", "Path to dataset.", "file")
        with col4:
            train_idx = input_with_validation("Path to Train Indices:", "data/processed/train_indices.npy", "Path to train indices.", "file")
            val_idx = input_with_validation("Path to Validation Indices:", "data/processed/val_indices.npy", "Path to validation indices.", "file")
    with st.expander("🔗 Model Options", True):
        col5, col6 = st.columns(2)
        with col5:
            model = input_with_validation("Path to Existing Model (optional):", "", "Path to pretrained model.", "file")
        with col6:
            chkpt_interval = st.number_input("Checkpoint Interval (Epochs):", 0, 100, 1, help="Set the interval (in epochs) for saving checkpoints. Leave it empty or set it to 0 for no checkpoints.")
    if st.button("Start Supervised Training 🏁"):
        required = [dataset, train_idx, val_idx] + ([model] if model else [])
        missing = [f for f in required if not validate_path(f, "file")]
        if not missing:
            try:
                lr_val, wd_val = float(lr), float(wd)
                execute_worker(lambda pc, sc: SupervisedWorker(int(epochs), int(batch_size), lr_val, wd_val, int(chkpt_interval) if chkpt_interval else 0, dataset, train_idx, val_idx, model if model else None,
                                                               optimizer, scheduler, accumulation_steps, int(num_workers), int(random_seed), policy_weight, value_weight, pc, sc))
            except ValueError:
                st.error("⚠️ Learning Rate and Weight Decay must be valid numbers.")
        else:
            st.error(f"⚠️ Missing files: {', '.join(missing)}.")

def run_reinforcement_training_worker():
    st.header("🛡️ Reinforcement Trainer")
    with st.expander("🛠️ Configure Training Parameters", True):
        col1, col2 = st.columns(2)
        with col1:
            num_iter = st.number_input("Number of Iterations:", 1, 1000, 10)
            games_per_iter = st.number_input("Games per Iteration:", 1, 10000, 1000, step=100)
            simulations = st.number_input("Simulations per Move:", 1, 10000, 100, step=10)
            c_puct = st.number_input("C_PUCT:", 0.0, 10.0, 1.4, step=0.1)
            temperature = st.number_input("Temperature:", 0.0, 10.0, 1.0, step=0.1)
        with col2:
            epochs = st.number_input("Epochs per Iteration:", 1, 1000, 5)
            batch_size = st.number_input("Batch Size:", 1, 10000, 128)
            accumulation_steps = st.number_input("Accumulation Steps:", 1, 100, 3)
            num_threads = st.number_input("Number of Threads:", 1, 32, 4)
            lr = st.text_input("Learning Rate:", "0.0001")
            wd = st.text_input("Weight Decay:", "0.0001")
            optimizer = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"])
            scheduler = st.selectbox("Scheduler Type:", ["cosineannealingwarmrestarts", "cosineannealing", "steplr", "exponentiallr", "onelr", "none"])
            num_workers = st.number_input("Worker Threads:", 1, 32, 4)
            policy_weight = st.number_input("Policy Weight:", 0.0, 10.0, 1.0, step=0.1)
            value_weight = st.number_input("Value Weight:", 0.0, 10.0, 2.0, step=0.1)
    with st.expander("🔗 Model Options", True):
        col3, col4 = st.columns(2)
        with col3:
            model = input_with_validation("Path to Pretrained Model (optional):", "", "Path to existing model.", "file")
        with col4:
            chkpt_interval = st.number_input("Checkpoint Interval (Iterations):", 0, 100, 1, help="Set the interval (in iterations) for saving checkpoints. Leave it empty or set it to 0 for no checkpoints.")
    random_seed = st.number_input("Random Seed:", 0, 100000, 42)
    if st.button("Start Reinforcement Training 🏁"):
        if model and not validate_path(model, "file"):
            st.error("⚠️ Invalid model path.")
        else:
            try:
                lr_val, wd_val = float(lr), float(wd)
                execute_worker(lambda pc, sc: ReinforcementWorker(model if model else None, int(num_iter), int(games_per_iter), int(simulations), float(c_puct), float(temperature), int(epochs), int(batch_size),
                                                                  int(num_threads), int(chkpt_interval), int(random_seed), optimizer, lr_val, wd_val, scheduler, accumulation_steps, int(num_workers), policy_weight, value_weight, pc, sc))
            except ValueError:
                st.error("⚠️ Learning Rate and Weight Decay must be valid numbers.")

def run_evaluation_worker():
    st.header("📈 Evaluation")
    with st.expander("🛠️ Configure Evaluation Parameters", True):
        col1, col2 = st.columns(2)
        with col1:
            model = input_with_validation("Path to Trained Model:", "models/saved_models/supervised_model.pth", "Path to trained model.", "file")
        with col2:
            dataset_idx = input_with_validation("Path to Dataset Indices:", "data/processed/test_indices.npy", "Path to dataset indices.", "file")
        h5_path = input_with_validation("Path to H5 Dataset:", "data/processed/dataset.h5", "Path to H5 dataset.", "file")
    if st.button("Start Evaluation 🏁"):
        required = [model, dataset_idx, h5_path]
        missing = [f for f in required if not validate_path(f, "file")]
        if not missing:
            execute_worker(lambda pc, sc: EvaluationWorker(model, dataset_idx, h5_path, pc, sc))
        else:
            st.error(f"⚠️ Missing files: {', '.join(missing)}.")

def run_benchmark_worker():
    st.header("🏆 Benchmarking")
    with st.expander("🛠️ Configure Benchmarking Parameters", True):
        col1, col2 = st.columns(2)
        with col1:
            bot1 = input_with_validation("Path to Bot1 Model:", "models/saved_models/supervised_model.pth", "Path to Bot1's model.", "file")
            bot1_mcts = st.checkbox("Bot1 Use MCTS", True)
            bot1_open = st.checkbox("Bot1 Use Opening Book", True)
        with col2:
            bot2 = input_with_validation("Path to Bot2 Model:", "models/saved_models/supervised_model.pth", "Path to Bot2's model.", "file")
            bot2_mcts = st.checkbox("Bot2 Use MCTS", True)
            bot2_open = st.checkbox("Bot2 Use Opening Book", True)
        num_games = st.number_input("Number of Games:", 1, 10000, 100)
    if st.button("Start Benchmarking 🏁"):
        required = [bot1, bot2]
        missing = [f for f in required if not validate_path(f, "file")]
        if not missing:
            execute_worker(lambda pc, sc: BenchmarkWorker(bot1, bot2, int(num_games), bot1_mcts, bot1_open, bot2_mcts, bot2_open, pc, sc))
        else:
            st.error(f"⚠️ Missing files: {', '.join(missing)}.")

sections = {
    "Data Preparation": run_data_preparation_worker,
    "Opening Book Generator": run_opening_book_worker,
    "Supervised Trainer": run_supervised_training_worker,
    "Reinforcement Trainer": run_reinforcement_training_worker,
    "Evaluation": run_evaluation_worker,
    "Benchmarking": run_benchmark_worker
}

if __name__ == "__main__":
    st.sidebar.title("🔧 Navigation")
    section = st.sidebar.radio("Choose the section:", list(sections.keys()))
    sections[section]()