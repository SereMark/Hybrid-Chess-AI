import os, warnings, streamlit as st
from src.data_preperation.data_preparation_worker import DataPreparationWorker
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
    progress, status = st.progress(0), st.empty()
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
        raw_pgn = input_with_validation("Path to Raw PGN File:", "data/raw/lichess_db_standard_rated_2024-12.pgn", "file")
        engine = input_with_validation("Path to Chess Engine:", "engine/stockfish/stockfish-windows-x86-64-avx2.exe", "file")
        generate_book = st.checkbox("Generate Opening Book", True)
        if generate_book:
            pgn = input_with_validation("Path to PGN File:", "data/raw/lichess_db_standard_rated_2024-12.pgn", "file")
            max_opening_moves = st.slider("Max Opening Moves:", 1, 30, 25)
        else:
            pgn, max_opening_moves = None, 0
        col1, col2 = st.columns(2)
        with col1:
            max_games = st.slider("Max Games:", 100, 20000, 10000)
        with col2:
            min_elo = st.slider("Min ELO:", 0, 3000, 1600)
        col3, col4 = st.columns(2)
        with col3:
            engine_depth = st.slider("Engine Depth:", 1, 30, 20)
        with col4:
            engine_threads = st.slider("Engine Threads:", 1, 8, 4)
        engine_hash = st.slider("Engine Hash (MB):", 128, 4096, 2048, step=128)
        batch_size = st.number_input("Batch Size:", 1, 10000, 512, step=1)
    if st.button("Start Data Preparation 🏁"):
        if generate_book:
            paths = [raw_pgn, engine, pgn]
        else:
            paths = [raw_pgn, engine]
        if all(validate_path(p, "file") for p in paths):
            execute_worker(lambda pc, sc: DataPreparationWorker(raw_pgn, max_games, min_elo, batch_size, engine, engine_depth, engine_threads, engine_hash, pgn or "", max_opening_moves, pc, sc))
        else:
            st.error("⚠️ Invalid file paths.")

def run_supervised_training_worker():
    st.header("🎓 Supervised Trainer")
    with st.expander("🛠️ Configure Training Parameters", True):
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs:", 1, 1000, 50)
            accumulation_steps = st.number_input("Accumulation Steps:", 1, 100, 8)
            wd = st.number_input("Weight Decay:", 0.0, 1.0, 0.0001, format="%.6f")
            model = input_with_validation("Path to Existing Model (optional):", "", "file")
        with col2:
            batch_size = st.number_input("Batch Size:", 1, 10000, 512, step=1)
            lr = st.number_input("Learning Rate:", 1e-6, 1.0, 0.0001, format="%.6f")
            optimizer = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"], index=0)
            chkpt_interval = st.number_input("Checkpoint Interval (Epochs):", 0, 100, 10, help="Set the interval (in epochs) for saving checkpoints. Set it to 0 for no checkpoints.")
        if optimizer in ["sgd", "rmsprop"]:
            momentum = st.number_input("Momentum:", 0.0, 1.0, 0.85, step=0.05)
        else:
            momentum = 0.0
        col3, col4 = st.columns(2)
        with col3:
            scheduler = st.selectbox("Scheduler Type:", ["cosineannealingwarmrestarts", "step", "linear", "onecycle"], index=0)
        with col4:
            num_workers = st.number_input("Number of Dataloader Workers:", 1, 32, 16)
        random_seed = st.number_input("Random Seed:", 0, 100000, 12345)
        policy_weight = st.number_input("Policy Weight:", 0.0, 10.0, 2.0, step=0.1)
        value_weight = st.number_input("Value Weight:", 0.0, 10.0, 3.0, step=0.1)
        grad_clip = st.number_input("Gradient Clip:", 0.0, 10.0, 2.0, step=0.1)
    with st.expander("📁 Dataset Details", True):
        dataset = input_with_validation("Path to Dataset:", "data/processed/dataset.h5", "file")
        train_idx = input_with_validation("Path to Train Indices:", "data/processed/train_indices.npy", "file")
        val_idx = input_with_validation("Path to Validation Indices:", "data/processed/val_indices.npy", "file")
    with st.expander("🔗 Model Options", True):
        wandb_flag = st.checkbox("Use Weights & Biases", True)
    if st.button("Start Supervised Training 🏁"):
        missing = [p for p in [dataset, train_idx, val_idx] if not validate_path(p, "file")] + ([model] if model and not validate_path(model, "file") else [])
        if scheduler == "onecycle" and optimizer not in ["sgd", "rmsprop"]:
            st.error("⚠️ onecycle scheduler is only compatible with optimizers supporting momentum (e.g., sgd or rmsprop).")
        elif any(lr <= 0 or wd < 0 for lr, wd in [(lr, wd)]) or any(v < 0 for v in [epochs, batch_size, accumulation_steps, num_workers, random_seed, policy_weight, value_weight, grad_clip]):
            st.error("⚠️ Invalid numeric values.")
        elif missing:
            st.error(f"⚠️ Missing files: {', '.join(missing)}.")
        elif not batch_size:
            st.error("⚠️ At least one batch size must be selected.")
        else:
            try:
                execute_worker(lambda pc, sc: SupervisedWorker(int(epochs), int(batch_size), float(lr), float(wd), int(chkpt_interval) if chkpt_interval else 0, dataset, train_idx, val_idx, model or None, optimizer, scheduler, accumulation_steps, int(num_workers), int(random_seed), float(policy_weight), float(value_weight), float(grad_clip), float(momentum), wandb_flag, pc, sc))
            except ValueError:
                st.error("⚠️ Invalid input values.")

def run_reinforcement_training_worker():
    st.header("🛡️ Reinforcement Trainer")
    with st.expander("🛠️ Configure Training Parameters", True):
        col1, col2 = st.columns(2)
        with col1:
            num_iter = st.number_input("Number of Iterations:", 1, 1000, 200)
            simulations = st.number_input("Simulations per Move:", 1, 10000, 1000, step=50)
            c_puct = st.number_input("C_PUCT:", 0.0, 10.0, 1.6, step=0.1)
            epochs = st.number_input("Epochs per Iteration:", 1, 1000, 20)
            accumulation_steps = st.number_input("Accumulation Steps:", 1, 100, 8)
            lr = st.number_input("Learning Rate:", 1e-6, 1.0, 0.0001, format="%.6f")
            optimizer = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"], index=0)
            scheduler = st.selectbox("Scheduler Type:", ["cosineannealingwarmrestarts", "step", "linear", "onecycle"], index=0)
            policy_weight = st.number_input("Policy Weight:", 0.0, 10.0, 2.0, step=0.1)
            grad_clip = st.number_input("Gradient Clip:", 0.0, 10.0, 2.0, step=0.1)
        with col2:
            games_per_iter = st.number_input("Games per Iteration:", 1, 10000, 3000, step=100)
            temperature = st.number_input("Temperature:", 0.0, 10.0, 0.6, step=0.1)
            num_threads = st.number_input("Number of Self-Play Threads:", 1, 32, 16)
            batch_size = st.number_input("Batch Size:", 1, 10000, 512)
            num_workers = st.number_input("Number of Dataloader Workers:", 1, 32, 16)
            wd = st.number_input("Weight Decay:", 0.0, 1.0, 0.0001, format="%.6f")
            chkpt_interval = st.number_input("Checkpoint Interval (Iterations):", 0, 100, 10, help="Set the interval (in iterations) for saving checkpoints. Set it to 0 for no checkpoints.")
            random_seed = st.number_input("Random Seed:", 0, 100000, 12345)
            value_weight = st.number_input("Value Weight:", 0.0, 10.0, 3.0, step=0.1)
        if optimizer in ["sgd", "rmsprop"]:
            momentum = st.number_input("Momentum:", 0.0, 1.0, 0.85, step=0.05)
        else:
            momentum = 0.0
    with st.expander("🔗 Model Options", True):
        col1, col2 = st.columns(2)
        with col1:
            model = input_with_validation("Path to Pretrained Model (optional):", "", "file")
        with col2:
            wandb_flag = st.checkbox("Use Weights & Biases", True)
    if st.button("Start Reinforcement Training 🏁"):
        missing = [model] if model and not validate_path(model, "file") else []
        bounds_checks = [
            (lr, 1e-6, 1.0),
            (wd, 0.0, 1.0),
            (c_puct, 0.0, 10.0),
            (temperature, 0.0, 10.0)
        ]
        if scheduler == "onecycle" and optimizer not in ["sgd", "rmsprop"]:
            st.error("⚠️ onecycle scheduler is only compatible with optimizers supporting momentum (e.g., sgd or rmsprop).")
        elif any(v < mn or v > mx for v, mn, mx in bounds_checks) or any(v < 0 for v in [num_iter, games_per_iter, simulations, epochs, batch_size, accumulation_steps, num_threads, num_workers, policy_weight, value_weight, grad_clip]):
            st.error("⚠️ One or more parameters are out of bounds or negative.")
        elif missing:
            st.error(f"⚠️ Missing files: {model}.")
        else:
            try:
                execute_worker(lambda pc, sc: ReinforcementWorker(
                    model or None, int(num_iter), int(games_per_iter), int(simulations), float(c_puct), float(temperature),
                    int(epochs), int(batch_size), int(num_threads), int(chkpt_interval), int(random_seed), optimizer,
                    float(lr), float(wd), scheduler, accumulation_steps, int(num_workers),
                    float(policy_weight), float(value_weight), float(grad_clip), float(momentum), wandb_flag, pc, sc
                ))
            except ValueError:
                st.error("⚠️ Invalid input values.")

def run_evaluation_worker():
    st.header("📈 Evaluation")
    with st.expander("🛠️ Configure Evaluation Parameters", True):
        model = input_with_validation("Path to Trained Model:", "models/saved_models/supervised_model.pth", "file")
        dataset_idx = input_with_validation("Path to Dataset Indices:", "data/processed/test_indices.npy", "file")
        h5_path = input_with_validation("Path to H5 Dataset:", "data/processed/dataset.h5", "file")
    if st.button("Start Evaluation 🏁"):
        missing = [p for p in [model, dataset_idx, h5_path] if not validate_path(p, "file")]
        if missing:
            st.error(f"⚠️ Missing files: {', '.join(missing)}.")
        else:
            execute_worker(lambda pc, sc: EvaluationWorker(model, dataset_idx, h5_path, pc, sc))

def run_benchmark_worker():
    st.header("🏆 Benchmarking")
    with st.expander("🛠️ Configure Benchmarking Parameters", True):
        col1, col2 = st.columns(2)
        with col1:
            bot1 = input_with_validation("Path to Bot1 Model:", "models/saved_models/supervised_model.pth", "file")
            bot1_mcts = st.checkbox("Bot1 Use MCTS", True)
        with col2:
            bot2 = input_with_validation("Path to Bot2 Model:", "models/saved_models/supervised_model.pth", "file")
            bot2_mcts = st.checkbox("Bot2 Use MCTS", True)
        col3, col4 = st.columns(2)
        with col3:
            bot1_open = st.checkbox("Bot1 Use Opening Book", True)
        with col4:
            bot2_open = st.checkbox("Bot2 Use Opening Book", True)
        num_games = st.number_input("Number of Games:", 1, 10000, 100)
    if st.button("Start Benchmarking 🏁"):
        missing = [p for p in [bot1, bot2] if not validate_path(p, "file")]
        if missing:
            st.error(f"⚠️ Missing files: {', '.join(missing)}.")
        else:
            execute_worker(lambda pc, sc: BenchmarkWorker(bot1, bot2, int(num_games), bot1_mcts, bot1_open, bot2_mcts, bot2_open, pc, sc))

def run_hyperparameter_optimization_worker():
    st.header("🔍 Hyperparameter Optimization")
    with st.expander("🛠️ General Configuration", True):
        col1, col2 = st.columns(2)
        with col1:
            num_trials = st.number_input("Number of Trials:", 1, 1000, 200, step=1)
            dataset_path = input_with_validation("Path to Dataset:", "data/processed/dataset.h5", "file")
            n_jobs = st.number_input("Number of Optuna Jobs:", 1, 16, 8, step=1)
            num_workers = st.number_input("Number of Dataloader Workers:", 1, 16, 1, step=1)
        with col2:
            timeout = st.number_input("Timeout (seconds):", 10, 86400, 7200, step=10)
            train_indices_path = input_with_validation("Path to Train Indices:", "data/processed/train_indices.npy", "file")
            random_seed = st.number_input("Random Seed:", 0, 100000, 12345, step=1)
    with st.expander("⚙️ Hyperparameter Settings", True):
        col1, col2 = st.columns(2)
        with col1:
            lr_min = st.number_input("Learning Rate (Min):", 1e-7, 1.0, 1e-5, format="%.1e")
            wd_min = st.number_input("Weight Decay (Min):", 1e-7, 1.0, 1e-6, format="%.1e")
            pw_min = st.number_input("Policy Weight (Min):", 0.0, 10.0, 1.0, step=0.1)
            vw_min = st.number_input("Value Weight (Min):", 0.0, 10.0, 1.0, step=0.1)
            epochs_min = st.number_input("Epochs (Min):", 1, 500, 20, step=1)
            optimizer = st.multiselect("Optimizers:", ["adamw", "sgd", "adam", "rmsprop"], default=["adamw", "adam"])
            grad_clip_min = st.slider("Gradient Clipping (Min):", 0.0, 5.0, 0.5, 0.1)
            momentum_min = st.slider("Momentum (Min):", 0.5, 0.99, 0.85, 0.01) if any(opt in ["sgd", "rmsprop"] for opt in optimizer) else 0.0
            accumulation_steps_min = st.number_input("Accumulation Steps (Min):", 1, 64, 4, step=1)
        with col2:
            lr_max = st.number_input("Learning Rate (Max):", 1e-7, 1.0, 1e-3, format="%.1e")
            wd_max = st.number_input("Weight Decay (Max):", 1e-7, 1.0, 1e-3, format="%.1e")
            pw_max = st.number_input("Policy Weight (Max):", 0.0, 10.0, 3.0, step=0.1)
            vw_max = st.number_input("Value Weight (Max):", 0.0, 10.0, 3.0, step=0.1)
            epochs_max = st.number_input("Epochs (Max):", 1, 500, 100, step=1)
            scheduler = st.multiselect("Schedulers:", ["cosineannealingwarmrestarts", "step", "linear", "onecycle"], default=["cosineannealingwarmrestarts", "onecycle"])
            grad_clip_max = st.slider("Gradient Clipping (Max):", 0.0, 5.0, 2.0, 0.1)
            momentum_max = st.slider("Momentum (Max):", 0.5, 0.99, 0.95, 0.01) if any(opt in ["sgd", "rmsprop"] for opt in optimizer) else 0.0
            accumulation_steps_max = st.number_input("Accumulation Steps (Max):", 1, 64, 8, step=1)
        batch_size = st.multiselect("Batch Sizes:", [16, 32, 64, 128, 256], default=[32, 64, 128, 256])
    with st.expander("📁 Dataset Details", True):
        val_indices_path = input_with_validation("Path to Validation Indices:", "data/processed/val_indices.npy", "file")
    if st.button("Start Optimization 🏁"):
        missing = [p for p in [dataset_path, train_indices_path, val_indices_path] if not validate_path(p, "file")]
        conditions = [lr_min > lr_max, wd_min > wd_max, pw_min > pw_max, vw_min > vw_max]
        if any(conditions + ([momentum_min > momentum_max] if any(opt in ["sgd", "rmsprop"] for opt in optimizer) else [])):
            st.error("⚠️ Minimum values cannot exceed maximum values.")
        elif not batch_size:
            st.error("⚠️ At least one batch size must be selected.")
        elif not optimizer:
            st.error("⚠️ At least one optimizer must be selected.")
        elif not scheduler:
            st.error("⚠️ At least one scheduler must be selected.")
        elif missing:
            st.error(f"⚠️ Missing files: {', '.join(missing)}.")
        else:
            try:
                execute_worker(lambda pc, sc: HyperparameterOptimizationWorker(
                    num_trials, timeout, dataset_path, train_indices_path, val_indices_path, n_jobs, num_workers, random_seed,
                    lr_min, lr_max, wd_min, wd_max, batch_size, epochs_min, epochs_max, optimizer, scheduler,
                    grad_clip_min, grad_clip_max, momentum_min, momentum_max, accumulation_steps_min, accumulation_steps_max,
                    pw_min, pw_max, vw_min, vw_max, pc, sc
                ))
            except ValueError:
                st.error("⚠️ Invalid input values.")

sections = {
    "Data Preparation": run_data_preparation_worker,
    "Supervised Trainer": run_supervised_training_worker,
    "Reinforcement Trainer": run_reinforcement_training_worker,
    "Evaluation": run_evaluation_worker,
    "Benchmarking": run_benchmark_worker,
    "Hyperparameter Optimization": run_hyperparameter_optimization_worker
}

if __name__ == "__main__":
    st.sidebar.title("🔧 Navigation")
    sections[st.sidebar.radio("Choose the section:", list(sections.keys()))]()