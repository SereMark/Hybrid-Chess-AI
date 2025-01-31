import os, streamlit as st
from src.data_preperation.data_preparation_worker import DataPreparationWorker
from src.training.hyperparameter_optimization.hyperparameter_optimization_worker import HyperparameterOptimizationWorker
from src.training.supervised.supervised_training_worker import SupervisedWorker
from src.training.reinforcement.reinforcement_training_worker import ReinforcementWorker
from src.analysis.evaluation.evaluation_worker import EvaluationWorker
from src.analysis.benchmark.benchmark_worker import BenchmarkWorker
from src.lichess_deployment.lichess_bot_deployment_worker import LichessBotDeploymentWorker

st.set_page_config(page_title="Chess AI Management Dashboard", page_icon="♟️", layout="wide")

st.markdown(
    """
    <style>
    /* Section header styling */
    .section-header {
         font-size: 1.2rem;
         font-weight: 600;
         border-bottom: 1px solid #ccc;
         padding-bottom: 0.5rem;
         margin-top: 1rem;
         margin-bottom: 1rem;
    }
    /* Button styling */
    .stButton>button {
         margin-top: 1rem;
         padding: 0.5rem 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def validate_path(p, path_type="file"):
    if not p:
        return False
    if path_type == "file":
        return os.path.isfile(p)
    elif path_type == "directory":
        return os.path.isdir(p)
    return False

def input_v(label, default="", path_type="file", help_text=None, key=None):
    if key is None:
        key = label.replace(" ", "_").lower() + "_key"
    value = st.text_input(label, default, help=help_text, key=key)
    if value:
        message = "✅ Valid Path" if validate_path(value, path_type) else "⚠️ Invalid Path"
        st.markdown(f"<span>{message}</span>", unsafe_allow_html=True)
    return value

def run_worker(button_label, required_files, worker_fn):
    if st.button(button_label):
        missing = [file_path for file_path in required_files if not validate_path(file_path, "file")]
        if missing:
            st.error(f"⚠️ Missing or invalid files: {', '.join(missing)}.")
            return
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            worker = worker_fn(
                lambda p: progress_bar.progress(int(p)),
                lambda m: status_text.text(m)
            )
            status_text.text("🚀 Started!")
            if worker.run():
                progress_bar.progress(100)
                status_text.text("🎉 Completed!")
                st.success("Process completed successfully!")
                st.balloons()
            else:
                status_text.text("⚠️ Failed.")
                st.error("Process failed. Check your inputs and try again.")
        except Exception as e:
            status_text.text(f"⚠️ Error: {e}")
            st.error(f"An error occurred: {e}")

def dp_ui():
    st.subheader("Data Preparation")
    with st.container():
        st.markdown("<div class='section-header'>Basic Settings</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            raw_pgn = input_v("Raw PGN File Path:", "data/raw/lichess_db_standard_rated_2024-12.pgn",
                                "file", "Path to raw PGN.", "dp_raw_pgn")
            engine_path = input_v("Chess Engine Path:", "engine/stockfish/stockfish-windows-x86-64-avx2.exe",
                                    "file", "Engine executable.", "dp_engine")
            generate_book = st.checkbox("Generate Opening Book", True, key="dp_generate_book")
            if generate_book:
                book_pgn = input_v("Opening Book PGN File:", "data/raw/lichess_db_standard_rated_2024-12.pgn",
                                    "file", key="dp_opening_pgn")
                max_opening_moves = st.slider("Max Opening Moves:", 1, 30, 20, key="dp_max_opening_moves")
            else:
                book_pgn = ""
                max_opening_moves = 0
        with col2:
            max_games = st.slider("Max Games:", 100, 20000, 5000, key="dp_max_games")
            elo_range = st.slider("ELO Range:", 800, 2800, (1400, 2400), 50, key="dp_elo_range")
            engine_depth = st.slider("Engine Depth:", 1, 30, 20, key="dp_engine_depth")
            engine_threads = st.slider("Engine Threads:", 1, 8, 4, key="dp_engine_threads")
            engine_hash = st.slider("Engine Hash (MB):", 128, 4096, 2048, 128, key="dp_engine_hash")
            batch_size = st.number_input("Batch Size:", 1, 10000, 256, 1, key="dp_batch_size")
            wandb_flag = st.checkbox("Enable Weights & Biases", True, key="dp_wandb")

        with st.expander("Advanced Filtering"):
            colA, colB = st.columns(2)
            with colA:
                skip_moves = st.slider("Skip games (moves):", 0, 500, (5, 150), 1, key="dp_skip_moves")
            with colB:
                use_time_analysis = st.checkbox("Time-based Analysis", False, key="dp_use_time_analysis")
                analysis_time = st.slider("Analysis Time (s):", 0.1, 5.0, 0.5, 0.1, key="dp_analysis_time") if use_time_analysis else None

        required_files = [raw_pgn, engine_path] + ([book_pgn] if generate_book else [])
        run_worker("Start Data Preparation", required_files,
                   lambda prog, stat: DataPreparationWorker(
                       raw_pgn=raw_pgn,
                       max_games=max_games,
                       min_elo=elo_range[0],
                       max_elo=elo_range[1],
                       batch_size=batch_size,
                       engine_path=engine_path,
                       engine_depth=engine_depth,
                       engine_threads=engine_threads,
                       engine_hash=engine_hash,
                       pgn_file=book_pgn,
                       max_opening_moves=max_opening_moves,
                       wandb_flag=wandb_flag,
                       progress_callback=prog,
                       status_callback=stat,
                       skip_min_moves=skip_moves[0],
                       skip_max_moves=skip_moves[1],
                       use_time_analysis=use_time_analysis,
                       analysis_time=analysis_time
                   )
        )

def sup_ui():
    st.subheader("Supervised Trainer")
    with st.container():
        st.markdown("<div class='section-header'>Model & Dataset Settings</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.number_input("Epochs:", 1, 1000, 50, key="sup_epochs")
            weight_decay = st.number_input("Weight Decay:", 0.0, 1.0, 0.01, format="%.6f", key="sup_wd")
            learning_rate = st.number_input("Learning Rate:", 1e-6, 1.0, 0.001, format="%.6f", key="sup_lr")
            checkpoint_interval = st.number_input("Checkpoint Interval:", 0, 100, 10, key="sup_chkpt_int")
            policy_weight = st.number_input("Policy Weight:", 0.0, 10.0, 1.0, 0.1, key="sup_policy_weight")
            dataset_path = input_v("Path to Dataset:", "data/processed/dataset.h5", "file", key="sup_dataset_path")
        with col2:
            accumulation_steps = st.number_input("Accumulation Steps:", 1, 100, 4, key="sup_accum_steps")
            batch_size = st.number_input("Batch Size:", 1, 10000, 256, key="sup_batch_size")
            optimizer_type = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"],
                                          index=0, key="sup_optimizer")
            scheduler_type = st.selectbox("Scheduler Type:",
                                          ["cosineannealingwarmrestarts", "step", "linear", "onecycle"],
                                          index=0, key="sup_scheduler")
            value_weight = st.number_input("Value Weight:", 0.0, 10.0, 1.0, 0.1, key="sup_value_weight")
            train_indices_path = input_v("Path to Train Indices:", "data/processed/train_indices.npy",
                                         "file", key="sup_train_idx")
        with col3:
            num_workers = st.number_input("Dataloader Workers:", 1, 32, 8, key="sup_num_workers")
            random_seed = st.number_input("Random Seed:", 0, 100000, 42, key="sup_random_seed")
            grad_clip = st.number_input("Gradient Clip:", 0.0, 10.0, 1.0, 0.1, key="sup_grad_clip")
            momentum = st.number_input("Momentum:", 0.0, 1.0, 0.9, 0.05, key="sup_momentum") if optimizer_type in ["sgd", "rmsprop"] else 0.0
            val_indices_path = input_v("Path to Validation Indices:", "data/processed/val_indices.npy",
                                       "file", key="sup_val_idx")
            existing_model = input_v("Existing Model (optional):", "", "file", key="sup_model_path")

        st.markdown("<div class='section-header'>Additional Options</div>", unsafe_allow_html=True)
        colA, colB = st.columns(2)
        with colA:
            wandb_flag = st.checkbox("Use Weights & Biases", True, key="sup_wandb")
        with colB:
            use_early_stopping = st.checkbox("Use Early Stopping", False, key="sup_earlystop_checkbox")
            early_stopping_patience = st.number_input("Early Stopping Patience:", 1, 50, 5, key="sup_es_patience") if use_early_stopping else 0

        required_files = [dataset_path, train_indices_path, val_indices_path] + ([existing_model] if existing_model else [])
        run_worker("Start Supervised Training", required_files,
                   lambda prog, stat: SupervisedWorker(
                       epochs=int(epochs),
                       batch_size=int(batch_size),
                       lr=float(learning_rate),
                       weight_decay=float(weight_decay),
                       checkpoint_interval=int(checkpoint_interval),
                       dataset_path=dataset_path,
                       train_indices_path=train_indices_path,
                       val_indices_path=val_indices_path,
                       model_path=existing_model if existing_model else None,
                       optimizer=optimizer_type,
                       scheduler=scheduler_type,
                       accumulation_steps=int(accumulation_steps),
                       num_workers=int(num_workers),
                       random_seed=int(random_seed),
                       policy_weight=float(policy_weight),
                       value_weight=float(value_weight),
                       grad_clip=float(grad_clip),
                       momentum=float(momentum),
                       wandb_flag=wandb_flag,
                       use_early_stopping=use_early_stopping,
                       early_stopping_patience=int(early_stopping_patience),
                       progress_callback=prog,
                       status_callback=stat
                   )
        )

def rein_ui():
    st.subheader("Reinforcement Trainer")
    with st.container():
        st.markdown("<div class='section-header'>Self-Play & Training Settings</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            iterations = st.number_input("Iterations:", 1, 1000, 200, key="rein_num_iter")
            simulations = st.number_input("Simulations per Move:", 1, 10000, 500, 50, key="rein_simulations")
            accumulation_steps = st.number_input("Accumulation Steps:", 1, 100, 4, key="rein_accum_steps")
        with col2:
            c_puct = st.number_input("C_PUCT:", 0.0, 10.0, 1.0, 0.1, key="rein_c_puct")
            epochs = st.number_input("Epochs per Iteration:", 1, 1000, 10, key="rein_epochs")
            learning_rate = st.number_input("Learning Rate:", 1e-6, 1.0, 1e-4, format="%.6f", key="rein_lr")
        with col3:
            optimizer_type = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"],
                                          index=0, key="rein_optimizer")
            scheduler_type = st.selectbox("Scheduler Type:", ["linear", "cosineannealingwarmrestarts", "step", "onecycle"],
                                          index=0, key="rein_scheduler")
            policy_weight = st.number_input("Policy Weight:", 0.0, 10.0, 2.0, 0.1, key="rein_policy_weight")

        st.markdown("<div class='section-header'>Additional Self-Play Options</div>", unsafe_allow_html=True)
        colA, colB, colC = st.columns(3)
        with colA:
            games_per_iteration = st.number_input("Games per Iteration:", 1, 10000, 1000, 50, key="rein_games_per_iter")
            batch_size = st.number_input("Batch Size:", 1, 10000, 256, key="rein_batch_size")
        with colB:
            temperature = st.number_input("Temperature:", 0.0, 10.0, 1.0, 0.1, key="rein_temperature")
            dataloader_workers = st.number_input("Dataloader Workers:", 1, 32, 4, key="rein_num_workers")
        with colC:
            selfplay_threads = st.number_input("Self-Play Threads:", 1, 32, 4, key="rein_num_threads")
            weight_decay = st.number_input("Weight Decay:", 0.0, 1.0, 0.0001, format="%.6f", key="rein_wd")
        colD, colE, colF = st.columns(3)
        with colD:
            checkpoint_interval = st.number_input("Checkpoint Interval:", 0, 100, 5, key="rein_chkpt_interval")
        with colE:
            random_seed = st.number_input("Random Seed:", 0, 100000, 12345, key="rein_random_seed")
        with colF:
            value_weight = st.number_input("Value Weight:", 0.0, 10.0, 1.0, 0.1, key="rein_value_weight")
            momentum = st.number_input("Momentum:", 0.0, 1.0, 0.85, 0.05, key="rein_momentum") if optimizer_type in ["sgd", "rmsprop"] else 0.0

        pretrained_model = input_v("Pretrained Model (optional):", "", "file", key="rein_model_path")
        wandb_flag = st.checkbox("Use Weights & Biases", True, key="rein_wandb")
        required_files = [pretrained_model] if pretrained_model else []
        run_worker("Start Reinforcement Training", required_files,
                   lambda prog, stat: ReinforcementWorker(
                       model_path=pretrained_model if pretrained_model else None,
                       num_iterations=int(iterations),
                       num_games_per_iteration=int(games_per_iteration),
                       simulations_per_move=int(simulations),
                       c_puct=float(c_puct),
                       temperature=float(temperature),
                       epochs_per_iteration=int(epochs),
                       batch_size=int(batch_size),
                       num_selfplay_threads=int(selfplay_threads),
                       checkpoint_interval=int(checkpoint_interval),
                       random_seed=int(random_seed),
                       optimizer_type=optimizer_type,
                       learning_rate=float(learning_rate),
                       weight_decay=float(weight_decay),
                       scheduler_type=scheduler_type,
                       accumulation_steps=int(accumulation_steps),
                       num_workers=int(dataloader_workers),
                       policy_weight=float(policy_weight),
                       value_weight=float(value_weight),
                       grad_clip=1.0,
                       momentum=float(momentum),
                       wandb_flag=wandb_flag,
                       progress_callback=prog,
                       status_callback=stat
                   )
        )

def eval_ui():
    st.subheader("Evaluation")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            model_path = input_v("Trained Model Path:", "models/saved_models/supervised_model.pth", "file", key="eval_model_path")
        with col2:
            test_indices = input_v("Test Indices Path:", "data/processed/test_indices.npy", "file", key="eval_dataset_idx")
        with col3:
            h5_dataset = input_v("H5 Dataset Path:", "data/processed/dataset.h5", "file", key="eval_h5_path")
        wandb_flag = st.checkbox("Enable Weights & Biases", True, key="eval_wandb")
        required_files = [model_path, test_indices, h5_dataset]
        run_worker("Start Evaluation", required_files,
                   lambda prog, stat: EvaluationWorker(
                       model_path=model_path,
                       indices_path=test_indices,
                       h5_path=h5_dataset,
                       wandb_flag=wandb_flag,
                       progress_callback=prog,
                       status_callback=stat
                   )
        )

def bench_ui():
    st.subheader("Benchmarking")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            bot1_model = input_v("Bot1 Model Path:", "models/saved_models/supervised_model.pth", "file", key="bench_bot1")
            bot2_model = input_v("Bot2 Model Path:", "models/saved_models/supervised_model.pth", "file", key="bench_bot2")
            num_games = st.number_input("Number of Games:", 1, 10000, 100, key="bench_num_games")
        with col2:
            bot1_use_mcts = st.checkbox("Bot1 Use MCTS", True, key="bench_bot1_mcts")
            bot1_use_opening_book = st.checkbox("Bot1 Use Opening Book", True, key="bench_bot1_open")
            bot2_use_mcts = st.checkbox("Bot2 Use MCTS", True, key="bench_bot2_mcts")
            bot2_use_opening_book = st.checkbox("Bot2 Use Opening Book", True, key="bench_bot2_open")
            wandb_flag = st.checkbox("Enable Weights & Biases", True, key="bench_wandb")
        required_files = [bot1_model, bot2_model]
        run_worker("Start Benchmarking", required_files,
                   lambda prog, stat: BenchmarkWorker(
                       bot1_path=bot1_model,
                       bot2_path=bot2_model,
                       num_games=int(num_games),
                       bot1_use_mcts=bot1_use_mcts,
                       bot1_use_opening_book=bot1_use_opening_book,
                       bot2_use_mcts=bot2_use_mcts,
                       bot2_use_opening_book=bot2_use_opening_book,
                       wandb_flag=wandb_flag,
                       progress_callback=prog,
                       status_callback=stat
                   )
        )

def hypo_ui():
    st.subheader("Hyperparameter Optimization")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            num_trials = st.number_input("Trials:", 1, 1000, 200, key="hopt_num_trials")
            n_jobs = st.number_input("Optuna Jobs:", 1, 16, 8, key="hopt_n_jobs")
        with col2:
            dataset_path = input_v("Dataset Path:", "data/processed/dataset.h5", "file", key="hopt_dataset_path")
            train_indices = input_v("Train Indices Path:", "data/processed/train_indices.npy", "file", key="hopt_train_indices")
        with col3:
            timeout = st.number_input("Timeout (s):", 10, 86400, 7200, key="hopt_timeout")
            random_seed = st.number_input("Random Seed:", 0, 100000, 12345, key="hopt_random_seed")
        
        with st.expander("Hyperparameter Settings", expanded=True):
            colA, colB, colC = st.columns(3)
            with colA:
                lr_range = st.slider("LR Range:", 1e-7, 1.0, (1e-5, 1e-3), format="%.1e", key="hopt_lr_range")
                wd_range = st.slider("WD Range:", 1e-7, 1.0, (1e-6, 1e-3), format="%.1e", key="hopt_wd_range")
                policy_weight_range = st.slider("Policy Weight Range:", 0.0, 10.0, (1.0, 2.0), 0.1, key="hopt_pw_range")
            with colB:
                value_weight_range = st.slider("Value Weight Range:", 0.0, 10.0, (1.0, 2.0), 0.1, key="hopt_vw_range")
                epochs_range = st.slider("Epochs Range:", 1, 500, (20, 100), key="hopt_epochs_range")
                grad_clip_range = st.slider("Grad Clip Range:", 0.0, 5.0, (0.5, 2.0), 0.1, key="hopt_grad_clip_range")
            with colC:
                accum_range = st.slider("Accumulation Range:", 1, 64, (4, 8), key="hopt_accum_steps_range")
                scheduler_opts = st.multiselect("Schedulers:", ["cosineannealingwarmrestarts", "step", "linear", "onecycle"],
                                                default=["cosineannealingwarmrestarts", "onecycle"], key="hopt_scheduler_opts")
                batch_sizes = st.multiselect("Batch Sizes:", [64, 128, 256],
                                             default=[64, 128, 256], key="hopt_batch_sizes")
                optimizer_opts = st.multiselect("Optimizers:", ["adamw", "sgd", "adam", "rmsprop"],
                                                default=["adamw", "adam"], key="hopt_optimizer_opts")
                momentum_range = st.slider("Momentum Range:", 0.5, 0.99, (0.85, 0.95), 0.01, key="hopt_momentum_range") if any(opt in ["sgd", "rmsprop"] for opt in optimizer_opts) else (0.0, 0.0)
        
        num_workers = st.number_input("Dataloader Workers:", 1, 16, 1, key="hopt_num_workers")
        
        with st.expander("Dataset Details"):
            val_indices = input_v("Validation Indices Path:", "data/processed/val_indices.npy", "file", key="hopt_val_indices")
        
        required_files = [dataset_path, train_indices, val_indices]
        run_worker("Start Hyperparameter Optimization", required_files,
                   lambda prog, stat: HyperparameterOptimizationWorker(
                       num_trials=int(num_trials),
                       timeout=int(timeout),
                       dataset_path=dataset_path,
                       train_indices_path=train_indices,
                       val_indices_path=val_indices,
                       n_jobs=int(n_jobs),
                       num_workers=int(num_workers),
                       random_seed=int(random_seed),
                       lr_min=float(lr_range[0]),
                       lr_max=float(lr_range[1]),
                       wd_min=float(wd_range[0]),
                       wd_max=float(wd_range[1]),
                       batch_size_options=batch_sizes,
                       epochs_min=int(epochs_range[0]),
                       epochs_max=int(epochs_range[1]),
                       optimizer_options=optimizer_opts,
                       scheduler_options=scheduler_opts,
                       grad_clip_min=float(grad_clip_range[0]),
                       grad_clip_max=float(grad_clip_range[1]),
                       momentum_min=float(momentum_range[0]),
                       momentum_max=float(momentum_range[1]),
                       accumulation_steps_min=int(accum_range[0]),
                       accumulation_steps_max=int(accum_range[1]),
                       policy_weight_min=float(policy_weight_range[0]),
                       policy_weight_max=float(policy_weight_range[1]),
                       value_weight_min=float(value_weight_range[0]),
                       value_weight_max=float(value_weight_range[1]),
                       progress_callback=prog,
                       status_callback=stat
                   )
        )

def lich_ui():
    st.subheader("Lichess Bot Deployment")
    with st.container():
        st.markdown("<div class='section-header'>Deployment Settings</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            model_path = input_v("Model Path:", "models/saved_models/chess_model.pth", "file", key="lichess_model_path")
            opening_book = input_v("Opening Book JSON Path:", "data/openings/book.json", "file",
                                    "Path to opening book JSON.", "lichess_opening_book_path")
        with col2:
            lich_token = st.text_input("Lichess Bot API Token:", "", type="password",
                                       help="Your bot token (requires 'bot:play').", key="lichess_token")
            time_control = st.selectbox("Preferred Time Control:",
                                        ["1+0 (Bullet)", "3+2 (Blitz)", "5+0 (Blitz)", "15+10 (Rapid)", "Classical"],
                                        index=1, key="lichess_time_control")
            rating_range = st.slider("Opponent Rating Range:", 800, 3000, (1200, 2400), 50, key="lichess_rating_range")
            use_mcts = st.checkbox("Use MCTS", True, key="lichess_use_mcts")
        
        cloud_provider = st.selectbox("Cloud Provider:", ["AWS", "Google Cloud", "Azure", "Other"],
                                      index=0, key="lichess_cloud_provider")
        st.write("Ensure your cloud credentials are set up.")
        required_files = [model_path, opening_book]

        def deployment_worker(prog, stat):
            return LichessBotDeploymentWorker(
                model_path=model_path,
                opening_book_path=opening_book,
                lichess_token=lich_token,
                time_control=time_control,
                rating_range=rating_range,
                use_mcts=use_mcts,
                cloud_provider=cloud_provider,
                progress_callback=prog,
                status_callback=stat
            )

        if st.button("Deploy / Refresh Lichess Bot"):
            if not lich_token.strip():
                st.error("⚠️ Lichess token is required.")
            else:
                run_worker("Deploy / Refresh Lichess Bot", required_files, deployment_worker)

if __name__ == "__main__":
    tabs = st.tabs([
        "Data Preparation",
        "Supervised Trainer",
        "Reinforcement Trainer",
        "Evaluation",
        "Benchmarking",
        "Hyperparameter Optimization",
        "Lichess Deployment"
    ])

    with tabs[0]: dp_ui()
    with tabs[1]: sup_ui()
    with tabs[2]: rein_ui()
    with tabs[3]: eval_ui()
    with tabs[4]: bench_ui()
    with tabs[5]: hypo_ui()
    with tabs[6]: lich_ui()