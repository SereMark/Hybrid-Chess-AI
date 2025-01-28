import os, warnings, streamlit as st
from src.data_processing.data_preparation.data_preparation_worker import DataPreparationWorker
from src.data_processing.opening_book.opening_book_worker import OpeningBookWorker
from src.training.Hyperparameter_Optimization.hyperparameter_optimization_worker import HyperparameterOptimizationWorker
from src.training.supervised.supervised_training_worker import SupervisedWorker
from src.training.reinforcement.reinforcement_training_worker import ReinforcementWorker
from src.analysis.evaluation.evaluation_worker import EvaluationWorker
from src.analysis.benchmark.benchmark_worker import BenchmarkWorker
warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext!")

st.set_page_config(page_title="Chess AI Management Dashboard", layout="wide", initial_sidebar_state="expanded")

def validate_path(path, type_="file"):
    return os.path.isfile(path) if type_ == "file" else os.path.isdir(path) if type_ == "directory" else False

def input_with_validation(label, default, type_="file"):
    path = st.text_input(label, default)
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
            raw_pgn = input_with_validation("Path to Raw PGN File:", "data/raw/lichess_db_standard_rated_2024-12.pgn", "file")
        with col2:
            engine = input_with_validation("Path to Chess Engine:", "engine/stockfish/stockfish-windows-x86-64-avx2.exe", "file")
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
            pgn = input_with_validation("Path to PGN File:", "data/raw/lichess_db_standard_rated_2024-12.pgn", "file")
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
            lr = st.number_input("Learning Rate:", 1e-6, 1.0, 0.0001, format="%.6f")
            wd = st.number_input("Weight Decay:", 0.0, 1.0, 0.0001, format="%.6f")
            optimizer = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"])
            momentum = st.number_input("Momentum:", 0.0, 1.0, 0.9, step=0.1) if optimizer in ["sgd", "rmsprop"] else 0.0
        with col2:
            scheduler = st.selectbox("Scheduler Type:", ["cosineannealingwarmrestarts", "step", "linear", "onecycle"])
            num_workers = st.number_input("Worker Threads:", 1, 32, 4)
            random_seed = st.number_input("Random Seed:", 0, 100000, 42)
            policy_weight = st.number_input("Policy Weight:", 0.0, 10.0, 1.0, step=0.1)
            value_weight = st.number_input("Value Weight:", 0.0, 10.0, 2.0, step=0.1)
            grad_clip = st.number_input("Gradient Clip:", 0.0, 10.0, 0.1, step=0.1)
    with st.expander("📁 Dataset Details", True):
        col3, col4 = st.columns(2)
        with col3:
            dataset = input_with_validation("Path to Dataset:", "data/processed/dataset.h5", "file")
        with col4:
            train_idx = input_with_validation("Path to Train Indices:", "data/processed/train_indices.npy", "file")
            val_idx = input_with_validation("Path to Validation Indices:", "data/processed/val_indices.npy", "file")
    with st.expander("🔗 Model Options", True):
        col5, col6 = st.columns(2)
        with col5:
            model = input_with_validation("Path to Existing Model (optional):", "", "file")
        with col6:
            chkpt_interval = st.number_input("Checkpoint Interval (Epochs):", 0, 100, 1, help="Set the interval (in epochs) for saving checkpoints. Set it to 0 for no checkpoints.")
    if st.button("Start Supervised Training 🏁"):
        if scheduler == "onecycle" and optimizer not in ["sgd", "rmsprop"]:
            st.error("⚠️ onecycle scheduler is only compatible with optimizers supporting momentum (e.g., sgd or rmsprop).")
            return
        required = [dataset, train_idx, val_idx] + ([model] if model else [])
        missing = [f for f in required if not validate_path(f, "file")]
        if not missing:
            try:
                lr_val, wd_val = float(lr), float(wd)
                execute_worker(lambda pc, sc: SupervisedWorker(int(epochs), int(batch_size), lr_val, wd_val, int(chkpt_interval) if chkpt_interval else 0, dataset, train_idx, val_idx, model if model else None,
                                                               optimizer, scheduler, accumulation_steps, int(num_workers), int(random_seed), policy_weight, value_weight, grad_clip, momentum, pc, sc))
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
            lr = st.number_input("Learning Rate:", 1e-6, 1.0, 0.0001, format="%.6f")
            wd = st.number_input("Weight Decay:", 0.0, 1.0, 0.0001, format="%.6f")
            optimizer = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"])
            momentum = st.number_input("Momentum:", 0.0, 1.0, 0.9, step=0.1) if optimizer in ["sgd", "rmsprop"] else 0.0
            scheduler = st.selectbox("Scheduler Type:", ["cosineannealingwarmrestarts", "step", "linear", "onecycle"])
            num_workers = st.number_input("Worker Threads:", 1, 32, 4)
            policy_weight = st.number_input("Policy Weight:", 0.0, 10.0, 1.0, step=0.1)
            value_weight = st.number_input("Value Weight:", 0.0, 10.0, 2.0, step=0.1)
            grad_clip = st.number_input("Gradient Clip:", 0.0, 10.0, 0.1, step=0.1)
    with st.expander("🔗 Model Options", True):
        col3, col4 = st.columns(2)
        with col3:
            model = input_with_validation("Path to Pretrained Model (optional):", "", "file")
        with col4:
            chkpt_interval = st.number_input("Checkpoint Interval (Iterations):", 0, 100, 1, help="Set the interval (in iterations) for saving checkpoints. Set it to 0 for no checkpoints.")
    random_seed = st.number_input("Random Seed:", 0, 100000, 42)
    if st.button("Start Reinforcement Training 🏁"):
        if scheduler == "onecycle" and optimizer not in ["sgd", "rmsprop"]:
            st.error("⚠️ onecycle scheduler is only compatible with optimizers supporting momentum (e.g., sgd or rmsprop).")
            return
        if model and not validate_path(model, "file"):
            st.error("⚠️ Invalid model path.")
            return
        try:
            lr_val, wd_val = float(lr), float(wd)
            execute_worker(lambda pc, sc: ReinforcementWorker(model if model else None, int(num_iter), int(games_per_iter), int(simulations), float(c_puct), float(temperature), int(epochs), int(batch_size), int(num_threads), int(chkpt_interval),
                                                                int(random_seed), optimizer, lr_val, wd_val, scheduler, accumulation_steps, int(num_workers), policy_weight, value_weight, grad_clip, momentum, pc, sc))
        except ValueError:
            st.error("⚠️ Learning Rate and Weight Decay must be valid numbers.")

def run_evaluation_worker():
    st.header("📈 Evaluation")
    with st.expander("🛠️ Configure Evaluation Parameters", True):
        col1, col2 = st.columns(2)
        with col1:
            model = input_with_validation("Path to Trained Model:", "models/saved_models/supervised_model.pth", "file")
        with col2:
            dataset_idx = input_with_validation("Path to Dataset Indices:", "data/processed/test_indices.npy", "file")
            h5_path = input_with_validation("Path to H5 Dataset:", "data/processed/dataset.h5", "file")
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
            bot1 = input_with_validation("Path to Bot1 Model:", "models/saved_models/supervised_model.pth", "file")
            bot1_mcts = st.checkbox("Bot1 Use MCTS", True)
            bot1_open = st.checkbox("Bot1 Use Opening Book", True)
        with col2:
            bot2 = input_with_validation("Path to Bot2 Model:", "models/saved_models/supervised_model.pth", "file")
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

def run_hyperparameter_optimization_worker():
    st.header("🔍 Hyperparameter Optimization")
    with st.expander("🛠️ General Configuration", True):
        col1, col2 = st.columns(2)
        with col1:
            num_trials = st.number_input("Number of Trials:", min_value=1, max_value=1000, value=100, step=1)
        with col2:
            timeout = st.number_input("Timeout (seconds):", min_value=10, max_value=86400, value=3600, step=10)
    with st.expander("📁 Dataset Details", True):
        col1, col2 = st.columns(2)
        with col1:
            dataset_path = input_with_validation("Path to Dataset:", "data/processed/dataset.h5", "file")
        with col2:
            train_indices_path = input_with_validation("Path to Train Indices:", "data/processed/train_indices.npy", "file")
            val_indices_path = input_with_validation("Path to Validation Indices:", "data/processed/val_indices.npy", "file")
    with st.expander("⚙️ Hardware & Random Seed", True):
        col3, col4 = st.columns(2)
        with col3:
            num_workers = st.number_input("Number of Worker Threads:", min_value=1, max_value=32, value=8, step=1)
        with col4:
            random_seed = st.number_input("Random Seed:", min_value=0, max_value=100000, value=42, step=1)
    with st.expander("⚙️ Hyperparameter Settings", True):
        st.subheader("Learning Parameters")
        col1, col2 = st.columns(2)
        with col1:
            lr_min = st.number_input("Learning Rate (Min):", min_value=1e-7, max_value=1.0, value=1e-4, format="%.1e")
        with col2:
            lr_max = st.number_input("Learning Rate (Max):", min_value=1e-7, max_value=1.0, value=1e-2, format="%.1e")
        col3, col4 = st.columns(2)
        with col3:
            wd_min = st.number_input("Weight Decay (Min):", min_value=1e-7, max_value=1.0, value=1e-5, format="%.1e")
        with col4:
            wd_max = st.number_input("Weight Decay (Max):", min_value=1e-7, max_value=1.0, value=1e-2, format="%.1e")
        st.subheader("Policy and Value Weights")
        col1, col2 = st.columns(2)
        with col1:
            pw_min = st.number_input("Policy Weight (Min):", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
        with col2:
            pw_max = st.number_input("Policy Weight (Max):", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        col3, col4 = st.columns(2)
        with col3:
            vw_min = st.number_input("Value Weight (Min):", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
        with col4:
            vw_max = st.number_input("Value Weight (Max):", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        st.subheader("Training Parameters")
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.multiselect("Batch Sizes:", [16, 32, 64, 128], default=[32, 64, 128])
        with col2:
            epochs_min = st.number_input("Epochs (Min):", min_value=1, max_value=500, value=10, step=1)
            epochs_max = st.number_input("Epochs (Max):", min_value=1, max_value=500, value=50, step=1)
        st.subheader("Optimizer & Scheduler")
        col1, col2 = st.columns(2)
        with col1:
            optimizer = st.multiselect("Optimizers:", ["adamw", "sgd", "adam", "rmsprop"], default=["adamw", "adam", "sgd"])
        with col2:
            scheduler = st.multiselect("Schedulers:", ["cosineannealingwarmrestarts", "step", "linear", "onecycle"], default=["cosineannealingwarmrestarts", "linear", "onecycle"])
        st.subheader("Regularization Parameters")
        col1, col2 = st.columns(2)
        with col1:
            grad_clip_min = st.slider("Gradient Clipping (Min):", 0.0, 5.0, 0.1, 0.1)
        with col2:
            grad_clip_max = st.slider("Gradient Clipping (Max):", 0.0, 5.0, 1.0, 0.1)
        st.subheader("Momentum & Accumulation Steps")
        col1, col2 = st.columns(2)
        with col1:
            momentum_min = st.slider("Momentum (Min):", 0.5, 0.99, 0.9, 0.01) if any(opt in ["sgd", "rmsprop"] for opt in optimizer) else None
        with col2:
            momentum_max = st.slider("Momentum (Max):", 0.5, 0.99, 0.99, 0.01) if any(opt in ["sgd", "rmsprop"] for opt in optimizer) else None
        col3, col4 = st.columns(2)
        with col3:
            accumulation_steps_min = st.number_input("Accumulation Steps (Min):", min_value=1, max_value=64, value=2, step=1)
        with col4:
            accumulation_steps_max = st.number_input("Accumulation Steps (Max):", min_value=1, max_value=64, value=16, step=1)
    if st.button("Start Optimization 🏁"):
        required_paths = [dataset_path, train_indices_path, val_indices_path]
        missing = [p for p in required_paths if not validate_path(p, "file")]
        if not missing:
            execute_worker(lambda pc, sc: HyperparameterOptimizationWorker(num_trials, timeout, dataset_path, train_indices_path, val_indices_path, num_workers, random_seed, lr_min, lr_max, wd_min, wd_max, batch_size, epochs_min, epochs_max, optimizer, scheduler,
                                                                           grad_clip_min, grad_clip_max, momentum_min, momentum_max, accumulation_steps_min, accumulation_steps_max, pw_min, pw_max, vw_min, vw_max, pc, sc))
        else:
            st.error(f"⚠️ Missing files: {', '.join(missing)}.")

sections = {
    "Data Preparation": run_data_preparation_worker,
    "Opening Book Generator": run_opening_book_worker,
    "Supervised Trainer": run_supervised_training_worker,
    "Reinforcement Trainer": run_reinforcement_training_worker,
    "Evaluation": run_evaluation_worker,
    "Benchmarking": run_benchmark_worker,
    "Hyperparameter Optimization": run_hyperparameter_optimization_worker
}

if __name__ == "__main__":
    st.sidebar.title("🔧 Navigation")
    sections[st.sidebar.radio("Choose the section:", list(sections.keys()))]()