from src.training.supervised.supervised_training_worker import SupervisedWorker
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
import optuna
import os

class HyperparameterOptimizationWorker:
    def __init__(self, num_trials, timeout, dataset_path, train_indices_path,
                 val_indices_path, n_jobs, num_workers, random_seed,
                 lr_min, lr_max, wd_min, wd_max, batch_size_options,
                 epochs_min, epochs_max, optimizer_options, scheduler_options,
                 grad_clip_min, grad_clip_max, momentum_min, momentum_max,
                 accumulation_steps_min, accumulation_steps_max,
                 policy_weight_min, policy_weight_max, value_weight_min, value_weight_max,
                 progress_callback=None, status_callback=None):
        self.num_trials = num_trials
        self.timeout = timeout
        self.dataset_path = dataset_path
        self.train_indices_path = train_indices_path
        self.val_indices_path = val_indices_path
        self.n_jobs = n_jobs
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.wd_min = wd_min
        self.wd_max = wd_max
        self.batch_size_options = batch_size_options
        self.epochs_min = epochs_min
        self.epochs_max = epochs_max
        self.optimizer_options = optimizer_options
        self.scheduler_options = scheduler_options
        self.grad_clip_min = grad_clip_min
        self.grad_clip_max = grad_clip_max
        self.momentum_min = momentum_min
        self.momentum_max = momentum_max
        self.accumulation_steps_min = accumulation_steps_min
        self.accumulation_steps_max = accumulation_steps_max
        self.policy_weight_min = policy_weight_min
        self.policy_weight_max = policy_weight_max
        self.value_weight_min = value_weight_min
        self.value_weight_max = value_weight_max
        self.progress_callback = progress_callback or (lambda x: None)
        self.status_callback = status_callback or (lambda x: None)
        self.db_name = "optimization_results.db"
        self.study_name = "HPO_Study"

    def objective(self, trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", self.lr_min, self.lr_max, log=True)
        wd = trial.suggest_float("weight_decay", self.wd_min, self.wd_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", self.batch_size_options)
        epochs = trial.suggest_int("epochs", self.epochs_min, self.epochs_max)
        optimizer = trial.suggest_categorical("optimizer", self.optimizer_options)
        scheduler = trial.suggest_categorical("scheduler", self.scheduler_options)
        grad_clip = trial.suggest_float("grad_clip", self.grad_clip_min, self.grad_clip_max)
        accumulation_steps = trial.suggest_int("accumulation_steps", self.accumulation_steps_min, self.accumulation_steps_max)
        policy_weight = trial.suggest_float("policy_weight", self.policy_weight_min, self.policy_weight_max)
        value_weight = trial.suggest_float("value_weight", self.value_weight_min, self.value_weight_max)
        if optimizer in ["sgd", "rmsprop"]:
            momentum = trial.suggest_float("momentum", self.momentum_min, self.momentum_max)
        else:
            momentum = 0.0
        worker = SupervisedWorker(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=wd,
            checkpoint_interval=0,
            dataset_path=self.dataset_path,
            train_indices_path=self.train_indices_path,
            val_indices_path=self.val_indices_path,
            model_path=None,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
            num_workers=self.num_workers,
            random_seed=self.random_seed,
            policy_weight=policy_weight,
            value_weight=value_weight,
            grad_clip=grad_clip,
            momentum=momentum,
            wandb_flag=False,
            use_early_stopping=False,
            early_stopping_patience=5,
            progress_callback=None,
            status_callback=None
        )
        results = worker.run()
        best_val_loss = results.get("best_composite_loss", float("inf"))
        trial.set_user_attr("learning_rate", lr)
        trial.set_user_attr("weight_decay", wd)
        trial.set_user_attr("batch_size", batch_size)
        trial.set_user_attr("epochs", epochs)
        trial.set_user_attr("optimizer", optimizer)
        trial.set_user_attr("scheduler", scheduler)
        trial.set_user_attr("grad_clip", grad_clip)
        trial.set_user_attr("accumulation_steps", accumulation_steps)
        trial.set_user_attr("policy_weight", policy_weight)
        trial.set_user_attr("value_weight", value_weight)
        trial.set_user_attr("momentum", momentum)
        trial.set_user_attr("training_time", results.get("training_time", 0.0))
        return best_val_loss

    def run(self) -> bool:
        try:
            optuna.delete_study(study_name=self.study_name, storage=f"sqlite:///{self.db_name}")
        except Exception:
            pass
        study = optuna.create_study(
            study_name=self.study_name,
            storage=f"sqlite:///{self.db_name}",
            direction="minimize",
            sampler=TPESampler(seed=self.random_seed),
            pruner=MedianPruner(),
            load_if_exists=False
        )
        def trial_callback(study: optuna.Study, trial: optuna.Trial):
            completed_trials = len([t for t in study.get_trials(deepcopy=False)
                                     if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
            progress_percent = (completed_trials / self.num_trials) * 100
            self.progress_callback(progress_percent)
            current_best = study.best_value if study.best_value is not None else float('inf')
            self.status_callback(f"Trial {completed_trials}/{self.num_trials} ended. Current Best Loss: {current_best:.4f}")
        try:
            study.optimize(
                self.objective,
                n_trials=self.num_trials,
                timeout=self.timeout if self.timeout > 0 else None,
                n_jobs=self.n_jobs,
                callbacks=[trial_callback]
            )
        except Exception as e:
            self.status_callback(f"Optimization failed: {e}")
            return False
        best_trial = study.best_trial
        best_loss = study.best_value
        self.status_callback(f"✅ Hyperparameter Optimization finished. Best Loss: {best_loss:.5f}, Trial {best_trial.number} with parameters: {best_trial.params}")
        results_dir = "hyperopt_results"
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "best_trial.txt"), "w") as f:
            f.write(f"Best Value (Val Loss) = {best_loss:.5f}\n")
            f.write(f"Best Trial = {best_trial.number}\n")
            for k, v in best_trial.params.items():
                f.write(f"{k}: {v}\n")
            f.write("\nUser Attributes:\n")
            for k, v in best_trial.user_attrs.items():
                f.write(f"{k}: {v}\n")
        return True