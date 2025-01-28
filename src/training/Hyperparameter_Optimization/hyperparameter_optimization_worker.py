from src.training.supervised.supervised_training_worker import SupervisedWorker
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
import optuna

class HyperparameterOptimizationWorker:
    def __init__(self, num_trials, timeout, dataset_path, train_indices_path, val_indices_path, n_jobs, num_workers, random_seed, lr_min, lr_max, wd_min, wd_max, batch_size, epochs_min, epochs_max, optimizer, scheduler, grad_clip_min, grad_clip_max, momentum_min, momentum_max, accumulation_steps_min, accumulation_steps_max, policy_weight_min, policy_weight_max, value_weight_min, value_weight_max, progress_callback, status_callback):
        self.num_trials, self.timeout = num_trials, timeout
        self.dataset_path, self.train_indices_path, self.val_indices_path = dataset_path, train_indices_path, val_indices_path
        self.n_jobs, self.num_workers, self.random_seed = n_jobs, num_workers, random_seed
        self.lr_min, self.lr_max = lr_min, lr_max
        self.wd_min, self.wd_max = wd_min, wd_max
        self.batch_size = batch_size
        self.epochs_min, self.epochs_max = epochs_min, epochs_max
        self.optimizer, self.scheduler = optimizer, scheduler
        self.grad_clip_min, self.grad_clip_max = grad_clip_min, grad_clip_max
        self.momentum_min, self.momentum_max = momentum_min, momentum_max
        self.accumulation_steps_min, self.accumulation_steps_max = accumulation_steps_min, accumulation_steps_max
        self.policy_weight_min, self.policy_weight_max = policy_weight_min, policy_weight_max
        self.value_weight_min, self.value_weight_max = value_weight_min, value_weight_max
        self.progress_callback, self.status_callback = progress_callback, status_callback
        self.current_trial = 0

    def objective(self, trial):
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', self.lr_min, self.lr_max),
            'weight_decay': trial.suggest_loguniform('weight_decay', self.wd_min, self.wd_max),
            'batch_size': trial.suggest_categorical('batch_size', self.batch_size),
            'epochs': trial.suggest_int('epochs', self.epochs_min, self.epochs_max),
            'optimizer': trial.suggest_categorical('optimizer', self.optimizer),
            'scheduler': trial.suggest_categorical('scheduler', self.scheduler),
            'grad_clip': trial.suggest_uniform('grad_clip', self.grad_clip_min, self.grad_clip_max),
            'accumulation_steps': trial.suggest_int('accumulation_steps', self.accumulation_steps_min, self.accumulation_steps_max),
            'policy_weight': trial.suggest_uniform('policy_weight', self.policy_weight_min, self.policy_weight_max),
            'value_weight': trial.suggest_uniform('value_weight', self.value_weight_min, self.value_weight_max)
        }
        params['momentum'] = trial.suggest_uniform('momentum', self.momentum_min, self.momentum_max) if params['optimizer'] in ['sgd', 'rmsprop'] else 0.0
        worker = SupervisedWorker(params['epochs'], params['batch_size'], params['learning_rate'], params['weight_decay'], 0, self.dataset_path, self.train_indices_path, self.val_indices_path,
                                  None, params['optimizer'], params['scheduler'], params['accumulation_steps'], self.num_workers, self.random_seed, params['policy_weight'], params['value_weight'],
                                  params['grad_clip'], params['momentum'], None, None)
        result = worker.run()
        val_metric = result.get('metric', float('inf'))
        for key, value in params.items():
            trial.set_user_attr(key, value)
        for key in ['val_loss', 'val_accuracy', 'best_epoch', 'training_time']:
            trial.set_user_attr(key, result.get(key, 0.0 if 'loss' in key or 'time' in key else 0))
        return val_metric

    def run(self):
        try: optuna.delete_study(study_name="HPO_Study", storage="sqlite:///optimization_results.db")
        except KeyError: pass
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=self.random_seed), pruner=MedianPruner(), storage="sqlite:///optimization_results.db", study_name="HPO_Study", load_if_exists=False)
        def trial_callback(study, trial):
            completed_trials = len([t for t in study.get_trials(deepcopy=False) if t.state in {TrialState.COMPLETE, TrialState.PRUNED}])
            self.progress_callback((completed_trials / self.num_trials) * 100)
            best_val_loss = study.best_value if study.best_value is not None else float('inf')
            self.status_callback(f"Trial {completed_trials}/{self.num_trials} completed. Best val_loss: {best_val_loss:.4f}")
        try:
            study.optimize(self.objective, n_trials=self.num_trials, timeout=self.timeout, callbacks=[trial_callback], n_jobs=self.n_jobs)
            return True
        except Exception as e:
            self.status_callback(f"Optimization failed: {e}")
            return False