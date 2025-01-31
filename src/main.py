import os, streamlit as st
from src.data_preperation.data_preparation_worker import DataPreparationWorker
from src.training.hyperparameter_optimization.hyperparameter_optimization_worker import HyperparameterOptimizationWorker
from src.training.supervised.supervised_training_worker import SupervisedWorker
from src.training.reinforcement.reinforcement_training_worker import ReinforcementWorker
from src.analysis.evaluation.evaluation_worker import EvaluationWorker
from src.analysis.benchmark.benchmark_worker import BenchmarkWorker
from src.lichess_deployment.lichess_bot_deployment_worker import LichessBotDeploymentWorker

st.set_page_config(page_title="Chess AI Management Dashboard", page_icon="♟️", layout="wide")

def validate_path(p, t="file"):
    if not p: return False
    return os.path.isfile(p) if t=="file" else os.path.isdir(p) if t=="directory" else False

def input_v(label, default="", p_type="file", help_text=None, key=None):
    key = key or label.replace(" ", "_").lower()+"_key"
    p = st.text_input(label, default, help=help_text, key=key)
    if p: st.markdown(f"<span>{'✅ Valid Path' if validate_path(p, p_type) else '⚠️ Invalid Path'}</span>", unsafe_allow_html=True)
    return p

def run_worker(btn, req, worker_fn, add_check=None, add_err=""):
    if st.button(btn):
        missing=[p for p in req if not validate_path(p, "file")]
        if add_check and not add_check():
            st.error(add_err)
        elif missing:
            st.error(f"⚠️ Missing or invalid files: {', '.join(missing)}.")
        else:
            with st.spinner("Processing..."):
                prog=st.progress(0); stat=st.empty()
                try:
                    worker=worker_fn(lambda p: prog.progress(int(p)), lambda m: stat.text(m))
                    stat.text("🚀 Started!")
                    if worker.run():
                        prog.progress(100); stat.text("🎉 Completed!")
                        st.success("Process completed successfully!"); st.balloons()
                    else:
                        stat.text("⚠️ Failed."); st.error("Process failed. Check your inputs and try again.")
                except Exception as e:
                    stat.text(f"⚠️ Error: {e}"); st.error(f"An error occurred: {e}")

def dp_ui():
    st.subheader("Data Preparation")
    c1, c2 = st.columns(2)
    with c1:
        raw = input_v("Raw PGN File Path:", "data/raw/lichess_db_standard_rated_2024-12.pgn", "file", "Path to raw PGN.", "dp_raw_pgn")
        eng = input_v("Chess Engine Path:", "engine/stockfish/stockfish-windows-x86-64-avx2.exe", "file", "Engine executable.", "dp_engine")
        gen = st.checkbox("Generate Opening Book", True, key="dp_generate_book")
        if gen:
            book = input_v("Opening Book PGN File:", "data/raw/lichess_db_standard_rated_2024-12.pgn", "file", key="dp_opening_pgn")
            max_moves = st.slider("Max Opening Moves:", 1, 30, 20, key="dp_max_opening_moves")
        else:
            book, max_moves = "", 0
    with c2:
        max_games = st.slider("Max Games:", 100, 20000, 5000, key="dp_max_games")
        elo = st.slider("ELO Range:", 800, 2800, (1400,2400), 50, key="dp_elo_range")
        depth = st.slider("Engine Depth:", 1, 30, 20, key="dp_engine_depth")
        threads = st.slider("Engine Threads:", 1, 8, 4, key="dp_engine_threads")
        hash_mb = st.slider("Engine Hash (MB):", 128, 4096, 2048, 128, key="dp_engine_hash")
        bs = st.number_input("Batch Size:", 1, 10000, 256, 1, key="dp_batch_size")
        wandb = st.checkbox("Enable Weights & Biases", True, key="dp_wandb")
    with st.expander("Advanced Filtering"):
        ca, cb = st.columns(2)
        with ca: skip = st.slider("Skip games (moves):", 0, 500, (5,150), 1, key="dp_skip_moves")
        with cb:
            use_time = st.checkbox("Time-based Analysis", False, key="dp_use_time_analysis")
            analysis = st.slider("Analysis Time (s):", 0.1, 5.0, 0.5, 0.1, key="dp_analysis_time") if use_time else None
    reqs=[raw, eng]+([book] if gen else [])
    run_worker("Start Data Preparation", reqs, lambda pc, sc: DataPreparationWorker(
         raw_pgn=raw, max_games=max_games, min_elo=elo[0], max_elo=elo[1], batch_size=bs, engine_path=eng,
         engine_depth=depth, engine_threads=threads, engine_hash=hash_mb, pgn_file=book, max_opening_moves=max_moves,
         wandb_flag=wandb, progress_callback=pc, status_callback=sc, skip_min_moves=skip[0], skip_max_moves=skip[1],
         use_time_analysis=use_time, analysis_time=analysis))

def sup_ui():
    st.subheader("Supervised Trainer")
    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.number_input("Epochs:", 1, 1000, 50, key="sup_epochs")
        wd = st.number_input("Weight Decay:", 0.0, 1.0, 0.01, format="%.6f", key="sup_wd")
        lr = st.number_input("Learning Rate:", 1e-6, 1.0, 0.001, format="%.6f", key="sup_lr")
        chkpt = st.number_input("Checkpoint Interval:", 0, 100, 10, key="sup_chkpt_int")
        pw = st.number_input("Policy Weight:", 0.0, 10.0, 1.0, 0.1, key="sup_policy_weight")
        ds = input_v("Path to Dataset:", "data/processed/dataset.h5", "file", key="sup_dataset_path")
    with c2:
        accum = st.number_input("Accumulation Steps:", 1, 100, 4, key="sup_accum_steps")
        bs = st.number_input("Batch Size:", 1, 10000, 256, key="sup_batch_size")
        opt = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"], index=0, key="sup_optimizer")
        sch = st.selectbox("Scheduler Type:", ["cosineannealingwarmrestarts", "step", "linear", "onecycle"], index=0, key="sup_scheduler")
        vw = st.number_input("Value Weight:", 0.0, 10.0, 1.0, 0.1, key="sup_value_weight")
        ti = input_v("Path to Train Indices:", "data/processed/train_indices.npy", "file", key="sup_train_idx")
    with c3:
        n_workers = st.number_input("Dataloader Workers:", 1, 32, 8, key="sup_num_workers")
        seed = st.number_input("Random Seed:", 0, 100000, 42, key="sup_random_seed")
        gc = st.number_input("Gradient Clip:", 0.0, 10.0, 1.0, 0.1, key="sup_grad_clip")
        mom = st.number_input("Momentum:", 0.0, 1.0, 0.9, 0.05, key="sup_momentum") if opt in ["sgd", "rmsprop"] else 0.0
        vi = input_v("Path to Validation Indices:", "data/processed/val_indices.npy", "file", key="sup_val_idx")
        model = input_v("Existing Model (optional):", "", "file", key="sup_model_path")
    cc1, cc2 = st.columns(2)
    with cc1: wandb = st.checkbox("Use Weights & Biases", True, key="sup_wandb")
    with cc2:
        use_es = st.checkbox("Use Early Stopping", False, key="sup_earlystop_checkbox")
        es_pat = st.number_input("Early Stopping Patience:", 1, 50, 5, key="sup_es_patience") if use_es else 0
    reqs=[ds, ti, vi]+([model] if model else [])
    run_worker("Start Supervised Training", reqs, lambda pc, sc: SupervisedWorker(
         epochs=int(epochs), batch_size=int(bs), lr=float(lr), weight_decay=float(wd), checkpoint_interval=int(chkpt),
         dataset_path=ds, train_indices_path=ti, val_indices_path=vi, model_path=model or None,
         optimizer=opt, scheduler=sch, accumulation_steps=int(accum), num_workers=int(n_workers), random_seed=int(seed),
         policy_weight=float(pw), value_weight=float(vw), grad_clip=float(gc), momentum=float(mom),
         wandb_flag=wandb, use_early_stopping=use_es, early_stopping_patience=int(es_pat),
         progress_callback=pc, status_callback=sc))

def rein_ui():
    st.subheader("Reinforcement Trainer")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_iter = st.number_input("Iterations:", 1, 1000, 200, key="rein_num_iter")
        sims = st.number_input("Simulations per Move:", 1, 10000, 500, 50, key="rein_simulations")
        accum = st.number_input("Accumulation Steps:", 1, 100, 4, key="rein_accum_steps")
    with c2:
        cpuct = st.number_input("C_PUCT:", 0.0, 10.0, 1.0, 0.1, key="rein_c_puct")
        epochs = st.number_input("Epochs per Iteration:", 1, 1000, 10, key="rein_epochs")
        lr = st.number_input("Learning Rate:", 1e-6, 1.0, 1e-4, format="%.6f", key="rein_lr")
    with c3:
        opt = st.selectbox("Optimizer Type:", ["adamw", "sgd", "adam", "rmsprop"], index=0, key="rein_optimizer")
        sch = st.selectbox("Scheduler Type:", ["linear", "cosineannealingwarmrestarts", "step", "onecycle"], index=0, key="rein_scheduler")
        pw = st.number_input("Policy Weight:", 0.0, 10.0, 2.0, 0.1, key="rein_policy_weight")
    d1, d2, d3 = st.columns(3)
    with d1:
        games = st.number_input("Games per Iteration:", 1, 10000, 1000, 50, key="rein_games_per_iter")
        bs = st.number_input("Batch Size:", 1, 10000, 256, key="rein_batch_size")
    with d2:
        temp = st.number_input("Temperature:", 0.0, 10.0, 1.0, 0.1, key="rein_temperature")
        n_workers = st.number_input("Dataloader Workers:", 1, 32, 4, key="rein_num_workers")
    with d3:
        threads = st.number_input("Self-Play Threads:", 1, 32, 4, key="rein_num_threads")
        wd = st.number_input("Weight Decay:", 0.0, 1.0, 0.0001, format="%.6f", key="rein_wd")
    e1, e2, e3 = st.columns(3)
    with e1: chkpt = st.number_input("Checkpoint Interval:", 0, 100, 5, key="rein_chkpt_interval")
    with e2: seed = st.number_input("Random Seed:", 0, 100000, 12345, key="rein_random_seed")
    with e3:
        vw = st.number_input("Value Weight:", 0.0, 10.0, 1.0, 0.1, key="rein_value_weight")
        mom = st.number_input("Momentum:", 0.0, 1.0, 0.85, 0.05, key="rein_momentum") if opt in ["sgd", "rmsprop"] else 0.0
    mod = input_v("Pretrained Model (optional):", "", "file", key="rein_model_path")
    wandb = st.checkbox("Use Weights & Biases", True, key="rein_wandb")
    reqs = [mod] if mod else []
    run_worker("Start Reinforcement Training", reqs, lambda pc, sc: ReinforcementWorker(
         model_path=mod or None, num_iterations=int(n_iter), num_games_per_iteration=int(games),
         simulations_per_move=int(sims), c_puct=float(cpuct), temperature=float(temp),
         epochs_per_iteration=int(epochs), batch_size=int(bs), num_selfplay_threads=int(threads),
         checkpoint_interval=int(chkpt), random_seed=int(seed), optimizer_type=opt, learning_rate=float(lr),
         weight_decay=float(wd), scheduler_type=sch, accumulation_steps=int(accum), num_workers=int(n_workers),
         policy_weight=float(pw), value_weight=float(vw), grad_clip=1.0, momentum=float(mom),
         wandb_flag=wandb, progress_callback=pc, status_callback=sc))

def eval_ui():
    st.subheader("Evaluation")
    c1, c2, c3 = st.columns(3)
    with c1: mod = input_v("Trained Model Path:", "models/saved_models/supervised_model.pth", "file", key="eval_model_path")
    with c2: ti = input_v("Test Indices Path:", "data/processed/test_indices.npy", "file", key="eval_dataset_idx")
    with c3: h5 = input_v("H5 Dataset Path:", "data/processed/dataset.h5", "file", key="eval_h5_path")
    wandb = st.checkbox("Enable Weights & Biases", True, key="eval_wandb")
    run_worker("Start Evaluation", [mod, ti, h5], lambda pc, sc: EvaluationWorker(
         model_path=mod, indices_path=ti, h5_path=h5, wandb_flag=wandb, progress_callback=pc, status_callback=sc))

def bench_ui():
    st.subheader("Benchmarking")
    c1, c2 = st.columns(2)
    with c1:
        bot1 = input_v("Bot1 Model Path:", "models/saved_models/supervised_model.pth", "file", key="bench_bot1")
        bot2 = input_v("Bot2 Model Path:", "models/saved_models/supervised_model.pth", "file", key="bench_bot2")
        num_games = st.number_input("Number of Games:", 1, 10000, 100, key="bench_num_games")
    with c2:
        bot1_mcts = st.checkbox("Bot1 Use MCTS", True, key="bench_bot1_mcts")
        bot1_open = st.checkbox("Bot1 Use Opening Book", True, key="bench_bot1_open")
        bot2_mcts = st.checkbox("Bot2 Use MCTS", True, key="bench_bot2_mcts")
        bot2_open = st.checkbox("Bot2 Use Opening Book", True, key="bench_bot2_open")
        wandb = st.checkbox("Enable Weights & Biases", True, key="bench_wandb")
    run_worker("Start Benchmarking", [bot1, bot2], lambda pc, sc: BenchmarkWorker(
         bot1_path=bot1, bot2_path=bot2, num_games=int(num_games), bot1_use_mcts=bot1_mcts,
         bot1_use_opening_book=bot1_open, bot2_use_mcts=bot2_mcts, bot2_use_opening_book=bot2_open,
         wandb_flag=wandb, progress_callback=pc, status_callback=sc))

def hypo_ui():
    st.subheader("Hyperparameter Optimization")
    c1, c2, c3 = st.columns(3)
    with c1:
        num_trials = st.number_input("Trials:", 1, 1000, 200, key="hopt_num_trials")
        n_jobs = st.number_input("Optuna Jobs:", 1, 16, 8, key="hopt_n_jobs")
    with c2:
        dataset = input_v("Dataset Path:", "data/processed/dataset.h5", "file", key="hopt_dataset_path")
        train_idx = input_v("Train Indices Path:", "data/processed/train_indices.npy", "file", key="hopt_train_indices")
    with c3:
        timeout = st.number_input("Timeout (s):", 10, 86400, 7200, key="hopt_timeout")
        seed = st.number_input("Random Seed:", 0, 100000, 12345, key="hopt_random_seed")
    with st.expander("Hyperparameter Settings", expanded=True):
        d1, d2, d3 = st.columns(3)
        lr_range = d1.slider("LR Range:", 1e-7, 1.0, (1e-5,1e-3), format="%.1e", key="hopt_lr_range")
        wd_range = d1.slider("WD Range:", 1e-7, 1.0, (1e-6,1e-3), format="%.1e", key="hopt_wd_range")
        pw_range = d1.slider("Policy Weight Range:", 0.0, 10.0, (1.0,2.0), 0.1, key="hopt_pw_range")
        vw_range = d2.slider("Value Weight Range:", 0.0, 10.0, (1.0,2.0), 0.1, key="hopt_vw_range")
        epochs_range = d2.slider("Epochs Range:", 1, 500, (20,100), key="hopt_epochs_range")
        grad_clip_range = d2.slider("Grad Clip Range:", 0.0, 5.0, (0.5,2.0), 0.1, key="hopt_grad_clip_range")
        accum_range = d3.slider("Accumulation Range:", 1, 64, (4,8), key="hopt_accum_steps_range")
        scheduler_opts = d3.multiselect("Schedulers:", ["cosineannealingwarmrestarts","step","linear","onecycle"],
                                         default=["cosineannealingwarmrestarts","onecycle"], key="hopt_scheduler_opts")
        batch_sizes = d3.multiselect("Batch Sizes:", [64,128,256], default=[64,128,256], key="hopt_batch_sizes")
        optimizer_opts = d3.multiselect("Optimizers:", ["adamw","sgd","adam","rmsprop"],
                                         default=["adamw","adam"], key="hopt_optimizer_opts")
        momentum_range = d3.slider("Momentum Range:", 0.5, 0.99, (0.85,0.95), 0.01, key="hopt_momentum_range") if any(opt in ["sgd","rmsprop"] for opt in optimizer_opts) else (0.0,0.0)
    num_workers = st.number_input("Dataloader Workers:", 1, 16, 1, key="hopt_num_workers")
    with st.expander("Dataset Details"):
        val_idx = input_v("Validation Indices Path:", "data/processed/val_indices.npy", "file", key="hopt_val_indices")
    req=[dataset, train_idx, val_idx]
    run_worker("Start Hyperparameter Optimization", req, lambda pc, sc: HyperparameterOptimizationWorker(
         num_trials=int(num_trials), timeout=int(timeout), dataset_path=dataset, train_indices_path=train_idx,
         val_indices_path=val_idx, n_jobs=int(n_jobs), num_workers=int(num_workers), random_seed=int(seed),
         lr_min=float(lr_range[0]), lr_max=float(lr_range[1]), wd_min=float(wd_range[0]), wd_max=float(wd_range[1]),
         batch_size_options=batch_sizes, epochs_min=int(epochs_range[0]), epochs_max=int(epochs_range[1]),
         optimizer_options=optimizer_opts, scheduler_options=scheduler_opts, grad_clip_min=float(grad_clip_range[0]),
         grad_clip_max=float(grad_clip_range[1]), momentum_min=float(momentum_range[0]), momentum_max=float(momentum_range[1]),
         accumulation_steps_min=int(accum_range[0]), accumulation_steps_max=int(accum_range[1]),
         policy_weight_min=float(pw_range[0]), policy_weight_max=float(pw_range[1]),
         value_weight_min=float(vw_range[0]), value_weight_max=float(vw_range[1]),
         progress_callback=pc, status_callback=sc))

def lich_ui():
    st.subheader("Lichess Bot Deployment")
    c1, c2 = st.columns(2)
    with c1:
        mod = input_v("Model Path:", "models/saved_models/chess_model.pth", "file", key="lichess_model_path")
        book = input_v("Opening Book JSON Path:", "data/openings/book.json", "file", "Path to opening book JSON.", "lichess_opening_book_path")
    with c2:
        token = st.text_input("Lichess Bot API Token:", "", type="password", help="Your bot token (requires 'bot:play').", key="lichess_token")
        tc = st.selectbox("Preferred Time Control:", ["1+0 (Bullet)","3+2 (Blitz)","5+0 (Blitz)","15+10 (Rapid)","Classical"], index=1, key="lichess_time_control")
        rr = st.slider("Opponent Rating Range:", 800, 3000, (1200,2400), 50, key="lichess_rating_range")
        mcts = st.checkbox("Use MCTS", True, key="lichess_use_mcts")
    cloud = st.selectbox("Cloud Provider:", ["AWS","Google Cloud","Azure","Other"], index=0, key="lichess_cloud_provider")
    st.write("Ensure your cloud credentials are set up.")
    run_worker("Deploy / Refresh Lichess Bot", [mod, book], lambda: bool(token.strip()), "⚠️ Lichess token is required.", lambda pc, sc: LichessBotDeploymentWorker(
         model_path=mod, opening_book_path=book, lichess_token=token, time_control=tc, rating_range=rr, use_mcts=mcts,
         cloud_provider=cloud, progress_callback=pc, status_callback=sc))

if __name__=="__main__":
    tabs = st.tabs(["Data Preparation","Supervised Trainer","Reinforcement Trainer","Evaluation","Benchmarking","Hyperparameter Optimization","Lichess Deployment"])
    with tabs[0]: dp_ui()
    with tabs[1]: sup_ui()
    with tabs[2]: rein_ui()
    with tabs[3]: eval_ui()
    with tabs[4]: bench_ui()
    with tabs[5]: hypo_ui()
    with tabs[6]: lich_ui()