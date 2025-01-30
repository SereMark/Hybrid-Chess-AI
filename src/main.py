import os
import warnings
import streamlit as st

from src.data_preperation.data_preparation_worker import DataPreparationWorker
from src.training.hyperparameter_optimization.hyperparameter_optimization_worker import HyperparameterOptimizationWorker
from src.training.supervised.supervised_training_worker import SupervisedWorker
from src.training.reinforcement.reinforcement_training_worker import ReinforcementWorker
from src.analysis.evaluation.evaluation_worker import EvaluationWorker
from src.analysis.benchmark.benchmark_worker import BenchmarkWorker
from src.lichess_deployment.lichess_bot_deployment_worker import LichessBotDeploymentWorker

warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext!")

st.set_page_config(
    page_title="Chess AI Management Dashboard",
    page_icon="♟️",
    layout="wide"
)

def validate_path(path: str, path_type: str = "file") -> bool:
    if not path:
        return False
    if path_type == "file":
        return os.path.isfile(path)
    elif path_type == "directory":
        return os.path.isdir(path)
    return False

def input_with_validation(
    label: str,
    default_value: str = "",
    path_type: str = "file",
    help_text: str = None,
    key: str = None,
) -> str:
    unique_key = key or (label + "_key").replace(" ", "_").lower()
    path = st.text_input(label, default_value, help=help_text, key=unique_key)

    if path:
        if validate_path(path, path_type):
            st.markdown(f"✅ **Valid Path**")
        else:
            st.markdown(f"⚠️ **Invalid Path**")
    return path

def execute_worker(create_worker):
    progress = st.progress(0)
    status = st.empty()
    try:
        worker = create_worker(
            lambda p: progress.progress(int(p)),
            lambda m: status.text(m)
        )
        status.text("🚀 Started!")
        result = worker.run()
        if result:
            progress.progress(100)
            status.text("🎉 Completed!")
            st.balloons()
        else:
            status.text("⚠️ Failed.")
    except Exception as e:
        status.text(f"⚠️ Error: {e}")

def data_preparation_tab():
    st.subheader("Data Preparation")
    st.write("Process PGN data, optionally generate an opening book, and prepare a dataset for training.")

    raw_pgn = input_with_validation(
        label="Raw PGN File Path:",
        default_value="data/raw/lichess_db_standard_rated_2024-12.pgn",
        path_type="file",
        help_text="Path to your raw PGN file.",
        key="dp_raw_pgn"
    )
    engine = input_with_validation(
        label="Chess Engine Path:",
        default_value="engine/stockfish/stockfish-windows-x86-64-avx2.exe",
        path_type="file",
        help_text="Path to your chess engine executable (e.g., Stockfish).",
        key="dp_engine"
    )
    generate_book = st.checkbox("Generate Opening Book", value=True, key="dp_generate_book")
    wandb_flag = st.checkbox("Enable Weights & Biases", value=True, key="dp_wandb_flag")

    if generate_book:
        pgn = input_with_validation(
            label="PGN File For Opening Book:",
            default_value="data/raw/lichess_db_standard_rated_2024-12.pgn",
            path_type="file",
            key="dp_opening_pgn"
        )
        max_opening_moves = st.slider("Max Opening Moves:", 1, 30, 25, key="dp_max_opening_moves")
    else:
        pgn, max_opening_moves = "", 0

    col1, col2 = st.columns(2)
    with col1:
        max_games = st.slider("Max Games:", 100, 20000, 10000, key="dp_max_games")
    with col2:
        min_elo = st.slider("ELO Range:", 0, 3000, (1600, 2400), key="dp_elo_range", step=50)

    col3, col4 = st.columns(2)
    with col3:
        engine_depth = st.slider("Engine Depth:", 1, 30, 20, key="dp_engine_depth")
    with col4:
        engine_threads = st.slider("Engine Threads:", 1, 8, 4, key="dp_engine_threads")

    engine_hash = st.slider("Engine Hash (MB):", 128, 4096, 2048, step=128, key="dp_engine_hash")
    batch_size = st.number_input("Batch Size:", 1, 10000, 512, step=1, key="dp_batch_size")

    with st.expander("Advanced Filtering"):
        skip_moves_range = st.slider(
            "Skip games with X moves",
            min_value=0,
            max_value=500,
            value=(5, 200),
            step=1,
            key="dp_skip_moves"
        )
        skip_min_moves, skip_max_moves = skip_moves_range
        use_time_analysis = st.checkbox("Use Time-based Engine Analysis", value=False, key="dp_use_time_analysis")
        analysis_time = st.slider("Analysis Time per Move (seconds)", 0.1, 5.0, 0.5, step=0.1, key="dp_analysis_time") if use_time_analysis else None

    if st.button("Start Data Preparation", key="dp_start_button"):
        paths_to_check = [raw_pgn, engine] + ([pgn] if generate_book else [])
        if all(validate_path(path, "file") for path in paths_to_check):
            min_elo_val, max_elo_val = min_elo
            execute_worker(lambda pc, sc: DataPreparationWorker(
                raw_pgn=raw_pgn,
                max_games=max_games,
                min_elo=min_elo_val,
                max_elo=max_elo_val,
                batch_size=batch_size,
                engine_path=engine,
                engine_depth=engine_depth,
                engine_threads=engine_threads,
                engine_hash=engine_hash,
                pgn_file=pgn,
                max_opening_moves=max_opening_moves,
                wandb_flag=wandb_flag,
                progress_callback=pc,
                status_callback=sc,
                skip_min_moves=skip_min_moves,
                skip_max_moves=skip_max_moves,
                use_time_analysis=use_time_analysis,
                analysis_time=analysis_time
            ))
        else:
            st.error("⚠️ Invalid file paths. Please check your inputs.")

def supervised_training_tab():
    st.subheader("Supervised Trainer")
    st.write("Train your model using supervised learning on a prepared dataset.")

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Epochs:", 1, 1000, 50, key="sup_epochs")
        accumulation_steps = st.number_input("Accumulation Steps:", 1, 100, 8, key="sup_accum_steps")
        wd = st.number_input("Weight Decay:", 0.0, 1.0, 0.0001, format="%.6f", key="sup_wd")
        model = input_with_validation(
            label="Existing Model (optional):",
            default_value="",
            path_type="file",
            help_text="Resume or transfer-learn.",
            key="sup_model_path"
        )
    with col2:
        batch_size = st.number_input("Batch Size:", 1, 10000, 512, step=1, key="sup_batch_size")
        lr = st.number_input("Learning Rate:", 1e-6, 1.0, 0.0001, format="%.6f", key="sup_lr")
        optimizer = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"], index=0, key="sup_optimizer")
        chkpt_interval = st.number_input("Checkpoint Interval (Epochs):", 0, 100, 10, key="sup_chkpt_int")

    if optimizer in ["sgd", "rmsprop"]:
        momentum = st.number_input("Momentum:", 0.0, 1.0, 0.85, step=0.05, key="sup_momentum")
    else:
        momentum = 0.0

    col3, col4 = st.columns(2)
    with col3:
        scheduler = st.selectbox(
            "Scheduler Type:",
            ["cosineannealingwarmrestarts", "step", "linear", "onecycle"], 
            index=0,
            key="sup_scheduler"
        )
    with col4:
        num_workers = st.number_input("Number of Dataloader Workers:", 1, 32, 16, key="sup_num_workers")

    random_seed = st.number_input("Random Seed:", 0, 100000, 12345, key="sup_random_seed")
    policy_weight = st.number_input("Policy Weight:", 0.0, 10.0, 2.0, step=0.1, key="sup_policy_weight")
    value_weight = st.number_input("Value Weight:", 0.0, 10.0, 3.0, step=0.1, key="sup_value_weight")
    grad_clip = st.number_input("Gradient Clip:", 0.0, 10.0, 2.0, step=0.1, key="sup_grad_clip")

    st.markdown("---")
    st.markdown("#### Dataset Details")
    dataset = input_with_validation(
        label="Path to Dataset:",
        default_value="data/processed/dataset.h5",
        path_type="file",
        key="sup_dataset_path"
    )
    train_idx = input_with_validation(
        label="Path to Train Indices:",
        default_value="data/processed/train_indices.npy",
        path_type="file",
        key="sup_train_idx"
    )
    val_idx = input_with_validation(
        label="Path to Validation Indices:",
        default_value="data/processed/val_indices.npy",
        path_type="file",
        key="sup_val_idx"
    )

    st.markdown("#### Additional Options")
    wandb_flag = st.checkbox("Use Weights & Biases", True, key="sup_wandb")
    use_early_stopping = st.checkbox("Use Early Stopping", value=False, key="sup_earlystop_checkbox")
    if use_early_stopping:
        early_stopping_patience = st.number_input("Early Stopping Patience", 1, 50, 5, key="sup_es_patience")
    else:
        early_stopping_patience = 0

    if st.button("Start Supervised Training", key="sup_start_button"):
        required_paths = [dataset, train_idx, val_idx]
        missing = [p for p in required_paths if not validate_path(p, "file")]
        if model and not validate_path(model, "file"):
            missing.append(model)

        if scheduler == "onecycle" and optimizer not in ["sgd", "rmsprop"]:
            st.error("⚠️ 'onecycle' scheduler is only compatible with momentum-based optimizers (SGD/RMSProp).")
        elif missing:
            st.error(f"⚠️ Missing or invalid files: {', '.join(missing)}.")
        elif batch_size < 1:
            st.error("⚠️ Batch size must be >= 1.")
        else:
            execute_worker(lambda pc, sc: SupervisedWorker(
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                weight_decay=float(wd),
                checkpoint_interval=int(chkpt_interval),
                dataset_path=dataset,
                train_indices_path=train_idx,
                val_indices_path=val_idx,
                model_path=model or None,
                optimizer=optimizer,
                scheduler=scheduler,
                accumulation_steps=accumulation_steps,
                num_workers=int(num_workers),
                random_seed=int(random_seed),
                policy_weight=float(policy_weight),
                value_weight=float(value_weight),
                grad_clip=float(grad_clip),
                momentum=float(momentum),
                wandb_flag=wandb_flag,
                use_early_stopping=use_early_stopping,
                early_stopping_patience=int(early_stopping_patience),
                progress_callback=pc,
                status_callback=sc
            ))

def reinforcement_training_tab():
    st.subheader("Reinforcement Trainer")
    st.write("Configure and start reinforcement learning for your chess model.")

    col1, col2 = st.columns(2)
    with col1:
        num_iter = st.number_input("Number of Iterations:", 1, 1000, 200, key="rein_num_iter")
        simulations = st.number_input("Simulations per Move:", 1, 10000, 1000, step=50, key="rein_simulations")
        c_puct = st.number_input("C_PUCT:", 0.0, 10.0, 1.6, step=0.1, key="rein_c_puct")
        epochs = st.number_input("Epochs per Iteration:", 1, 1000, 20, key="rein_epochs")
        accumulation_steps = st.number_input("Accumulation Steps:", 1, 100, 8, key="rein_accum_steps")
        lr = st.number_input("Learning Rate:", 1e-6, 1.0, 0.0001, format="%.6f", key="rein_lr")
        optimizer = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"], index=0, key="rein_optimizer")
        scheduler = st.selectbox("Scheduler Type:", ["cosineannealingwarmrestarts", "step", "linear", "onecycle"], index=0, key="rein_scheduler")
        policy_weight = st.number_input("Policy Weight:", 0.0, 10.0, 2.0, step=0.1, key="rein_policy_weight")
        grad_clip = st.number_input("Gradient Clip:", 0.0, 10.0, 2.0, step=0.1, key="rein_grad_clip")

    with col2:
        games_per_iter = st.number_input("Games per Iteration:", 1, 10000, 3000, step=100, key="rein_games_per_iter")
        temperature = st.number_input("Temperature:", 0.0, 10.0, 0.6, step=0.1, key="rein_temperature")
        num_threads = st.number_input("Number of Self-Play Threads:", 1, 32, 16, key="rein_num_threads")
        batch_size = st.number_input("Batch Size:", 1, 10000, 512, key="rein_batch_size")
        num_workers = st.number_input("Number of Dataloader Workers:", 1, 32, 16, key="rein_num_workers")
        wd = st.number_input("Weight Decay:", 0.0, 1.0, 0.0001, format="%.6f", key="rein_wd")
        chkpt_interval = st.number_input("Checkpoint Interval (Iterations):", 0, 100, 10, key="rein_chkpt_interval")
        random_seed = st.number_input("Random Seed:", 0, 100000, 12345, key="rein_random_seed")
        value_weight = st.number_input("Value Weight:", 0.0, 10.0, 3.0, step=0.1, key="rein_value_weight")

    if optimizer in ["sgd", "rmsprop"]:
        momentum = st.number_input("Momentum:", 0.0, 1.0, 0.85, step=0.05, key="rein_momentum")
    else:
        momentum = 0.0

    st.markdown("---")
    st.markdown("#### Model & Logging")
    model = input_with_validation(
        label="Pretrained Model (optional):",
        default_value="",
        path_type="file",
        key="rein_model_path"
    )
    wandb_flag = st.checkbox("Use Weights & Biases", True, key="rein_wandb")

    if st.button("Start Reinforcement Training", key="rein_start_button"):
        missing = []
        if model and not validate_path(model, "file"):
            missing.append(model)

        if scheduler == "onecycle" and optimizer not in ["sgd", "rmsprop"]:
            st.error("⚠️ 'onecycle' scheduler is only compatible with momentum-based optimizers (SGD/RMSProp).")
        elif missing:
            st.error(f"⚠️ Missing files: {', '.join(missing)}.")
        else:
            execute_worker(lambda pc, sc: ReinforcementWorker(
                model_path=model or None,
                num_iterations=int(num_iter),
                num_games_per_iteration=int(games_per_iter),
                simulations_per_move=int(simulations),
                c_puct=float(c_puct),
                temperature=float(temperature),
                epochs_per_iteration=int(epochs),
                batch_size=int(batch_size),
                num_selfplay_threads=int(num_threads),
                checkpoint_interval=int(chkpt_interval),
                random_seed=int(random_seed),
                optimizer_type=optimizer,
                learning_rate=float(lr),
                weight_decay=float(wd),
                scheduler_type=scheduler,
                accumulation_steps=int(accumulation_steps),
                num_workers=int(num_workers),
                policy_weight=float(policy_weight),
                value_weight=float(value_weight),
                grad_clip=float(grad_clip),
                momentum=float(momentum),
                wandb_flag=wandb_flag,
                progress_callback=pc,
                status_callback=sc
            ))

def evaluation_tab():
    st.subheader("Evaluation")
    st.write("Evaluate a trained model on a reserved test set.")

    model = input_with_validation(
        label="Trained Model Path:",
        default_value="models/saved_models/supervised_model.pth",
        path_type="file",
        key="eval_model_path"
    )
    dataset_idx = input_with_validation(
        label="Test Indices Path:",
        default_value="data/processed/test_indices.npy",
        path_type="file",
        key="eval_dataset_idx"
    )
    h5_path = input_with_validation(
        label="H5 Dataset Path:",
        default_value="data/processed/dataset.h5",
        path_type="file",
        key="eval_h5_path"
    )
    wandb_flag = st.checkbox("Enable Weights & Biases", True, key="eval_wandb")

    if st.button("Start Evaluation", key="eval_start_button"):
        missing = [p for p in [model, dataset_idx, h5_path] if not validate_path(p, "file")]
        if missing:
            st.error(f"⚠️ Missing files: {', '.join(missing)}.")
        else:
            execute_worker(lambda pc, sc: EvaluationWorker(
                model_path=model,
                indices_path=dataset_idx,
                h5_path=h5_path,
                wandb_flag=wandb_flag,
                progress_callback=pc,
                status_callback=sc
            ))

def benchmarking_tab():
    st.subheader("Benchmarking")
    st.write("Pit two trained bot models against each other and gather performance statistics.")

    col1, col2 = st.columns(2)
    with col1:
        bot1 = input_with_validation(
            label="Bot1 Model Path:",
            default_value="models/saved_models/supervised_model.pth",
            path_type="file",
            key="bench_bot1"
        )
        bot1_mcts = st.checkbox("Bot1 Use MCTS", True, key="bench_bot1_mcts")
        bot1_open = st.checkbox("Bot1 Use Opening Book", True, key="bench_bot1_open")
    with col2:
        bot2 = input_with_validation(
            label="Bot2 Model Path:",
            default_value="models/saved_models/supervised_model.pth",
            path_type="file",
            key="bench_bot2"
        )
        bot2_mcts = st.checkbox("Bot2 Use MCTS", True, key="bench_bot2_mcts")
        bot2_open = st.checkbox("Bot2 Use Opening Book", True, key="bench_bot2_open")

    num_games = st.number_input("Number of Games:", 1, 10000, 100, step=1, key="bench_num_games")
    wandb_flag = st.checkbox("Enable Weights & Biases", True, key="bench_wandb")

    if st.button("Start Benchmarking", key="bench_start_button"):
        missing = [p for p in [bot1, bot2] if not validate_path(p, "file")]
        if missing:
            st.error(f"⚠️ Missing files: {', '.join(missing)}.")
        else:
            execute_worker(lambda pc, sc: BenchmarkWorker(
                bot1_path=bot1,
                bot2_path=bot2,
                num_games=int(num_games),
                bot1_use_mcts=bot1_mcts,
                bot1_use_opening_book=bot1_open,
                bot2_use_mcts=bot2_mcts,
                bot2_use_opening_book=bot2_open,
                wandb_flag=wandb_flag,
                progress_callback=pc,
                status_callback=sc
            ))

def hyperparameter_optimization_tab():
    st.subheader("Hyperparameter Optimization")
    st.write("Use Optuna to find the best hyperparameters for your supervised training.")

    with st.expander("General Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            num_trials = st.number_input("Number of Trials:", 1, 1000, 200, step=1, key="hopt_num_trials")
            dataset_path = input_with_validation(
                label="Path to Dataset:",
                default_value="data/processed/dataset.h5",
                path_type="file",
                key="hopt_dataset_path"
            )
            n_jobs = st.number_input("Number of Optuna Jobs:", 1, 16, 8, step=1, key="hopt_n_jobs")
            num_workers = st.number_input("Number of Dataloader Workers:", 1, 16, 1, step=1, key="hopt_num_workers")

        with col2:
            timeout = st.number_input("Timeout (seconds):", 10, 86400, 7200, step=10, key="hopt_timeout")
            train_indices_path = input_with_validation(
                label="Path to Train Indices:",
                default_value="data/processed/train_indices.npy",
                path_type="file",
                key="hopt_train_indices"
            )
            random_seed = st.number_input("Random Seed:", 0, 100000, 12345, step=1, key="hopt_random_seed")

    with st.expander("Hyperparameter Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            lr_range = st.slider(
                "Learning Rate Range:",
                1e-7,
                1.0,
                (1e-5, 1e-3),
                format="%.1e",
                key="hopt_lr_range"
            )
            wd_range = st.slider(
                "Weight Decay Range:",
                1e-7,
                1.0,
                (1e-6, 1e-3),
                format="%.1e",
                key="hopt_wd_range"
            )
            pw_range = st.slider(
                "Policy Weight Range:",
                0.0,
                10.0,
                (1.0, 3.0),
                step=0.1,
                key="hopt_pw_range"
            )
            vw_range = st.slider(
                "Value Weight Range:",
                0.0,
                10.0,
                (1.0, 3.0),
                step=0.1,
                key="hopt_vw_range"
            )
            epochs_range = st.slider(
                "Epochs Range:",
                1,
                500,
                (20, 100),
                key="hopt_epochs_range"
            )
            grad_clip_range = st.slider(
                "Gradient Clip Range:",
                0.0,
                5.0,
                (0.5, 2.0),
                0.1,
                key="hopt_grad_clip_range"
            )
            accumulation_steps_range = st.slider(
                "Accumulation Steps Range:",
                1,
                64,
                (4, 8),
                key="hopt_accum_steps_range"
            )
        
        with col2:
            scheduler_opts = st.multiselect(
                "Schedulers:",
                ["cosineannealingwarmrestarts", "step", "linear", "onecycle"],
                default=["cosineannealingwarmrestarts", "onecycle"],
                key="hopt_scheduler_opts"
            )
            batch_size_options = st.multiselect(
                "Batch Sizes:",
                [16, 32, 64, 128, 256],
                default=[32, 64, 128, 256],
                key="hopt_batch_sizes"
            )

            optimizer_opts = st.multiselect(
                "Optimizers:",
                ["adamw", "sgd", "adam", "rmsprop"],
                default=["adamw", "adam"],
                key="hopt_optimizer_opts"
            )

            if any(opt in ["sgd", "rmsprop"] for opt in optimizer_opts):
                momentum_range = st.slider(
                    "Momentum Range:",
                    0.5,
                    0.99,
                    (0.85, 0.95),
                    0.01,
                    key="hopt_momentum_range"
                )
            else:
                momentum_range = (0.0, 0.0)

    with st.expander("Dataset Details", expanded=False):
        val_indices_path = input_with_validation(
            label="Path to Validation Indices:",
            default_value="data/processed/val_indices.npy",
            path_type="file",
            key="hopt_val_indices"
        )

    if st.button("Start Hyperparameter Optimization", key="hopt_start_button"):
        required_paths = [dataset_path, train_indices_path, val_indices_path]
        missing = [p for p in required_paths if not validate_path(p, "file")]

        lr_min, lr_max = lr_range
        wd_min, wd_max = wd_range
        pw_min, pw_max = pw_range
        vw_min, vw_max = vw_range
        epochs_min, epochs_max = epochs_range
        grad_clip_min, grad_clip_max = grad_clip_range
        accumulation_steps_min, accumulation_steps_max = accumulation_steps_range

        if any(opt in ["sgd", "rmsprop"] for opt in optimizer_opts):
            momentum_min, momentum_max = momentum_range
        else:
            momentum_min, momentum_max = 0.0, 0.0

        if lr_min > lr_max or wd_min > wd_max or pw_min > pw_max or vw_min > vw_max:
            st.error("⚠️ Min values cannot exceed max values (LR/WD/PW/VW).")
            return

        if any(opt in ["sgd", "rmsprop"] for opt in optimizer_opts) and (momentum_min > momentum_max):
            st.error("⚠️ 'momentum_min' cannot exceed 'momentum_max'.")
            return

        if not batch_size_options:
            st.error("⚠️ At least one batch size must be selected.")
            return

        if not optimizer_opts:
            st.error("⚠️ At least one optimizer must be selected.")
            return

        if not scheduler_opts:
            st.error("⚠️ At least one scheduler must be selected.")
            return

        if missing:
            st.error(f"⚠️ Missing files: {', '.join(missing)}.")
            return

        execute_worker(lambda pc, sc: HyperparameterOptimizationWorker(
            num_trials=int(num_trials),
            timeout=int(timeout),
            dataset_path=dataset_path,
            train_indices_path=train_indices_path,
            val_indices_path=val_indices_path,
            n_jobs=int(n_jobs),
            num_workers=int(num_workers),
            random_seed=int(random_seed),
            lr_min=float(lr_min),
            lr_max=float(lr_max),
            wd_min=float(wd_min),
            wd_max=float(wd_max),
            batch_size_options=batch_size_options,
            epochs_min=int(epochs_min),
            epochs_max=int(epochs_max),
            optimizer_options=optimizer_opts,
            scheduler_options=scheduler_opts,
            grad_clip_min=float(grad_clip_min),
            grad_clip_max=float(grad_clip_max),
            momentum_min=float(momentum_min),
            momentum_max=float(momentum_max),
            accumulation_steps_min=int(accumulation_steps_min),
            accumulation_steps_max=int(accumulation_steps_max),
            policy_weight_min=float(pw_min),
            policy_weight_max=float(pw_max),
            value_weight_min=float(vw_min),
            value_weight_max=float(vw_max),
            progress_callback=pc,
            status_callback=sc
        ))

def lichess_deployment_tab():
    st.subheader("Lichess Bot Deployment")
    st.write(
        "Configure and deploy your Lichess bot to the cloud, integrating your AI model, opening book, "
        "and MCTS logic. Provide your Lichess bot token, cloud settings, and engine paths below."
    )

    st.markdown("### Chess Engine & Model Integration")
    model_path = input_with_validation(
        label="Model Path:",
        default_value="models/saved_models/chess_model.pth",
        path_type="file",
        help_text="Path to your trained model file.",
        key="lichess_model_path"
    )
    opening_book_path = input_with_validation(
        label="Opening Book JSON Path:",
        default_value="data/openings/book.json",
        path_type="file",
        help_text="Path to your opening book JSON.",
        key="lichess_opening_book_path"
    )

    st.markdown("### Lichess Bot Settings")
    lichess_token = st.text_input(
        "Lichess Bot API Token:",
        value="",
        type="password",
        help="Your Lichess bot account token. Must have the 'bot:play' scope.",
        key="lichess_token"
    )
    time_control = st.selectbox(
        "Preferred Time Control:",
        ["1+0 (Bullet)", "3+2 (Blitz)", "5+0 (Blitz)", "15+10 (Rapid)", "Classical"],
        index=1,
        key="lichess_time_control"
    )
    rating_min, rating_max = st.slider(
        "Opponent Rating Range:",
        min_value=800,
        max_value=3000,
        value=(1200, 2400),
        step=50,
        key="lichess_rating_range"
    )
    use_mcts = st.checkbox("Use MCTS in Bot Play", value=True, key="lichess_use_mcts")

    st.markdown("### Cloud Deployment")
    cloud_provider = st.selectbox(
        "Cloud Provider:",
        ["AWS", "Google Cloud", "Azure", "Other"],
        index=0,
        key="lichess_cloud_provider"
    )
    st.write(
        "Select your desired cloud provider where the bot engine will be hosted. "
        "Make sure you have proper credentials set up."
    )

    st.markdown("### Deploy or Refresh")
    st.write(
        "Click the button below to upload/refresh your bot deployment in the cloud and authorize the bot "
        "to play on Lichess using the provided token."
    )

    if st.button("Deploy / Refresh Lichess Bot", key="lichess_deploy_button"):
        if not validate_path(model_path, "file"):
            st.error("⚠️ Invalid model file path.")
            return
        if not validate_path(opening_book_path, "file"):
            st.error("⚠️ Invalid opening book path.")
            return
        if not lichess_token.strip():
            st.error("⚠️ Lichess token is required.")
            return

        execute_worker(
            lambda progress_cb, status_cb: LichessBotDeploymentWorker(
                model_path=model_path,
                opening_book_path=opening_book_path,
                lichess_token=lichess_token,
                time_control=time_control,
                rating_range=(rating_min, rating_max),
                use_mcts=use_mcts,
                cloud_provider=cloud_provider,
                progress_callback=progress_cb,
                status_callback=status_cb
            )
        )

def main():
    tabs = st.tabs([
        "Data Preparation",
        "Supervised Trainer",
        "Reinforcement Trainer",
        "Evaluation",
        "Benchmarking",
        "Hyperparameter Optimization",
        "Lichess Deployment"
    ])

    with tabs[0]:
        data_preparation_tab()
    with tabs[1]:
        supervised_training_tab()
    with tabs[2]:
        reinforcement_training_tab()
    with tabs[3]:
        evaluation_tab()
    with tabs[4]:
        benchmarking_tab()
    with tabs[5]:
        hyperparameter_optimization_tab()
    with tabs[6]:
        lichess_deployment_tab()


if __name__ == "__main__":
    main()