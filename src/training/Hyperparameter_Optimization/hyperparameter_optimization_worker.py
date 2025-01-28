from src.training.supervised.supervised_training_worker import SupervisedWorker
import optuna, os, torch

class HyperparameterOptimizationWorker:
    def __init__(self, num_trials, timeout, dataset_path, train_indices_path, val_indices_path, num_workers, random_seed, lr_min, lr_max, wd_min, wd_max, batch_size, epochs_min, epochs_max, optimizer, scheduler, grad_clip_min, grad_clip_max, momentum_min, momentum_max, accumulation_steps_min, accumulation_steps_max, policy_weight_min, policy_weight_max, value_weight_min, value_weight_max, progress_callback, status_callback):
        self.num_trials, self.timeout, self.dataset_path, self.train_indices_path, self.val_indices_path, self.num_workers, self.random_seed = num_trials, timeout, dataset_path, train_indices_path, val_indices_path, num_workers, random_seed
        self.lr_min, self.lr_max, self.wd_min, self.wd_max, self.batch_size, self.epochs_min, self.epochs_max = lr_min, lr_max, wd_min, wd_max, batch_size, epochs_min, epochs_max
        self.optimizer, self.scheduler, self.grad_clip_min, self.grad_clip_max, self.momentum_min, self.momentum_max = optimizer, scheduler, grad_clip_min, grad_clip_max, momentum_min, momentum_max
        self.accumulation_steps_min, self.accumulation_steps_max, self.policy_weight_min, self.policy_weight_max, self.value_weight_min, self.value_weight_max, self.progress_callback, self.status_callback = accumulation_steps_min, accumulation_steps_max, policy_weight_min, policy_weight_max, value_weight_min, value_weight_max, progress_callback, status_callback
        self.current_trial = 0

    def objective(self, trial):
        params = {
            'lr': trial.suggest_loguniform('lr', self.lr_min, self.lr_max),
            'wd': trial.suggest_loguniform('wd', self.wd_min, self.wd_max),
            'batch_size': trial.suggest_categorical('batch_size', self.batch_size),
            'epochs': trial.suggest_int('epochs', self.epochs_min, self.epochs_max),
            'optimizer': trial.suggest_categorical('optimizer', self.optimizer),
            'scheduler': trial.suggest_categorical('scheduler', self.scheduler),
            'momentum': 0.0 if trial.suggest_categorical('optimizer', self.optimizer) in ['sgd', 'rmsprop'] else trial.suggest_uniform('momentum', self.momentum_min, self.momentum_max),
            'grad_clip': trial.suggest_uniform('grad_clip', self.grad_clip_min, self.grad_clip_max),
            'accumulation_steps': trial.suggest_int('accumulation_steps', self.accumulation_steps_min, self.accumulation_steps_max),
            'policy_weight': trial.suggest_uniform('policy_weight', self.policy_weight_min, self.policy_weight_max),
            'value_weight': trial.suggest_uniform('value_weight', self.value_weight_min, self.value_weight_max),
        }
        worker = SupervisedWorker(params['epochs'], params['batch_size'], params['lr'], params['wd'], 0, self.dataset_path, self.train_indices_path, self.val_indices_path, None, params['optimizer'], params['scheduler'],
                                  params['accumulation_steps'], self.num_workers, self.random_seed, params['policy_weight'], params['value_weight'], params['grad_clip'], params['momentum'], self.progress_callback, self.status_callback)
        result = worker.run()
        trial.set_user_attr("lr", params['lr'])
        trial.set_user_attr("weight_decay", params['wd'])
        trial.set_user_attr("batch_size", params['batch_size'])
        trial.set_user_attr("epochs", params['epochs'])
        trial.set_user_attr("optimizer", params['optimizer'])
        trial.set_user_attr("scheduler", params['scheduler'])
        trial.set_user_attr("momentum", params['momentum'])
        trial.set_user_attr("grad_clip", params['grad_clip'])
        trial.set_user_attr("accumulation_steps", params['accumulation_steps'])
        trial.set_user_attr("policy_weight", params['policy_weight'])
        trial.set_user_attr("value_weight", params['value_weight'])
        trial.set_user_attr("val_loss", result.get("val_loss", 0.0))
        trial.set_user_attr("val_accuracy", result.get("val_accuracy", 0.0))
        trial.set_user_attr("best_epoch", result.get("best_epoch", 0))
        trial.set_user_attr("training_time", result.get("training_time", 0.0))
        torch.cuda.reset_peak_memory_stats()
        trial.set_user_attr("gpu_memory", torch.cuda.max_memory_allocated())
        return result.get('metric', 0.0)

    def run(self):
        db_filename = "optimization_results.db"
        db_path = f"sqlite:///{db_filename}"
        if os.path.exists(db_filename):
            os.remove(db_filename)
        study = optuna.create_study(direction="minimize", study_name="HPO Study", storage=db_path, load_if_exists=True)
        def trial_callback(study, trial):
            self.current_trial += 1
            self.progress_callback((len(study.trials) / self.num_trials) * 100)
            self.status_callback(f"Trial {self.current_trial}/{self.num_trials} completed. Best metric: {study.best_value:.4f}")
        study.optimize(self.objective, n_trials=self.num_trials, timeout=self.timeout, callbacks=[trial_callback], gc_after_trial=True)
        return True