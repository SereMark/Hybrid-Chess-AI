import os
import time
import yaml
import wandb
import optuna
import numpy as np
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.utils.train import (
    set_seed, get_optimizer, get_scheduler,
    get_device, train_epoch, validate
)
from src.utils.chess import H5Dataset, get_move_count
from src.model import ChessModel

class HyperoptPipeline:
    def __init__(self, config: Config):
        self.config = config
        
        self.seed = config.get('project.seed', 42)
        set_seed(self.seed)
        
        self.trials = config.get('hyperopt.trials', 10)
        self.timeout = config.get('hyperopt.timeout', 3600)
        self.jobs = config.get('hyperopt.jobs', 1)
        
        self.lr_range = config.get('hyperopt.lr_range', [0.0001, 0.01])
        self.wd_range = config.get('hyperopt.wd_range', [0.00001, 0.001])
        self.batch_sizes = config.get('hyperopt.batch_sizes', [64, 128, 256])
        self.epochs_range = config.get('hyperopt.epochs_range', [5, 20])
        self.optimizers = config.get('hyperopt.optimizers', ["adam", "adamw"])
        self.schedulers = config.get('hyperopt.schedulers', ["linear", "onecycle"])
        
        self.dataset = config.get('data.dataset', 'data/dataset.h5')
        self.train_idx = config.get('data.train_idx', 'data/train_indices.npy')
        self.val_idx = config.get('data.val_idx', 'data/val_indices.npy')
        
        self.results_dir = '/content/drive/MyDrive/chess_ai/hyperopt_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.db_name = os.path.join(self.results_dir, "optimization.db")
        self.study_name = f"HPO_{self.config.mode}"
        
        self.config_path = os.path.join('/content/drive/MyDrive/chess_ai', 'config.yml')
    
    def setup(self):
        print("Setting up hyperparameter optimization...")
        
        try:
            local_dataset = '/content/drive/MyDrive/chess_ai/data/dataset.h5'
            local_train_idx = '/content/drive/MyDrive/chess_ai/data/train_indices.npy'
            local_val_idx = '/content/drive/MyDrive/chess_ai/data/val_indices.npy'
            
            os.makedirs('/content/drive/MyDrive/chess_ai/data', exist_ok=True)
            
            if os.path.exists(local_dataset):
                self.dataset = local_dataset
            if os.path.exists(local_train_idx):
                self.train_idx = local_train_idx
            if os.path.exists(local_val_idx):
                self.val_idx = local_val_idx
                
            print(f"Using dataset: {self.dataset}")
            print(f"Using train indices: {self.train_idx}")
            print(f"Using validation indices: {self.val_idx}")
        except Exception as e:
            print(f"Error accessing dataset files: {e}")
            print("Using original paths...")
        
        if self.config.get('wandb.enabled', True):
            try:
                wandb.init(
                    project=self.config.get('wandb.project', 'chess_ai'),
                    name=f"hyperopt_{self.config.mode}_{time.strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "mode": self.config.mode,
                        "trials": self.trials,
                        "timeout": self.timeout,
                        "jobs": self.jobs,
                        "lr_range": self.lr_range,
                        "wd_range": self.wd_range,
                        "batch_sizes": self.batch_sizes,
                        "epochs_range": self.epochs_range,
                        "optimizers": self.optimizers,
                        "schedulers": self.schedulers
                    }
                )
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                
        return True
    
    def objective(self, trial):
        lr = trial.suggest_float("lr", self.lr_range[0], self.lr_range[1], log=True)
        weight_decay = trial.suggest_float("weight_decay", self.wd_range[0], self.wd_range[1], log=True)
        batch_size = trial.suggest_categorical("batch_size", self.batch_sizes)
        epochs = trial.suggest_int("epochs", self.epochs_range[0], self.epochs_range[1])
        optimizer_type = trial.suggest_categorical("optimizer", self.optimizers)
        scheduler_type = trial.suggest_categorical("scheduler", self.schedulers)
        
        grad_clip = trial.suggest_float("grad_clip", 0.0, 5.0)
        accum_steps = trial.suggest_int("accum_steps", 1, 8)
        policy_weight = trial.suggest_float("policy_weight", 0.5, 2.0)
        value_weight = trial.suggest_float("value_weight", 0.5, 2.0)
        
        if optimizer_type in ["sgd", "rmsprop"]:
            momentum = trial.suggest_float("momentum", 0.5, 0.99)
        else:
            momentum = 0.0
        
        channels = trial.suggest_categorical("channels", [32, 48, 64, 96])
        
        trial_start = time.time()
        
        print(f"\nTrial {trial.number}: {trial.params}")
        print("-" * 50)
        
        try:
            device_info = get_device()
            device = device_info["device"]
            device_type = device_info["type"]
            use_tpu = device_type == "tpu"
            
            train_indices = np.load(self.train_idx)
            val_indices = np.load(self.val_idx)
            
            if self.config.mode == "test":
                if len(train_indices) > 10000:
                    np.random.shuffle(train_indices)
                    train_indices = train_indices[:10000]
                if len(val_indices) > 2000:
                    np.random.shuffle(val_indices)
                    val_indices = val_indices[:2000]
            
            train_dataset = H5Dataset(self.dataset, train_indices)
            val_dataset = H5Dataset(self.dataset, val_indices)
            
            workers = min(4, os.cpu_count() or 2)
            pin_memory = device_type == "gpu"
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                pin_memory=pin_memory
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=pin_memory
            )
            
            model = ChessModel(
                moves=get_move_count(),
                ch=channels,
                use_tpu=use_tpu
            ).to(device)
            
            optimizer = get_optimizer(
                model, optimizer_type, lr, weight_decay, momentum
            )
            
            total_steps = epochs * len(train_loader)
            scheduler = get_scheduler(optimizer, scheduler_type, total_steps)
            
            best_val_loss = float('inf')
            early_stop_counter = 0
            early_stopping_patience = 5
            
            for epoch in range(1, epochs + 1):
                train_metrics = train_epoch(
                    model,
                    train_loader,
                    device_info,
                    optimizer,
                    policy_weight,
                    value_weight,
                    accum_steps,
                    grad_clip,
                    scheduler,
                    compute_accuracy=True,
                    log_interval=100
                )
                
                val_metrics = validate(model, val_loader, device_info)
                
                composite_loss = policy_weight * val_metrics["policy_loss"] + value_weight * val_metrics["value_loss"]
                
                trial.report(composite_loss, epoch)
                
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.exceptions.TrialPruned()
                
                if composite_loss < best_val_loss:
                    best_val_loss = composite_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
                
                print(f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_metrics['policy_loss']:.4f}/{train_metrics['value_loss']:.4f}, "
                      f"Val Loss: {val_metrics['policy_loss']:.4f}/{val_metrics['value_loss']:.4f}, "
                      f"Loss: {composite_loss:.4f}")
            
            final_policy_loss = val_metrics["policy_loss"]
            final_value_loss = val_metrics["value_loss"]
            final_accuracy = val_metrics["accuracy"]
            final_loss = policy_weight * final_policy_loss + value_weight * final_value_loss
            
            training_time = time.time() - trial_start
            
            trial.set_user_attr("policy_loss", final_policy_loss)
            trial.set_user_attr("value_loss", final_value_loss)
            trial.set_user_attr("accuracy", final_accuracy)
            trial.set_user_attr("time", training_time)
            
            if wandb.run is not None:
                wandb.log({
                    "trial": trial.number,
                    "policy_loss": final_policy_loss,
                    "value_loss": final_value_loss,
                    "accuracy": final_accuracy,
                    "loss": final_loss,
                    "time": training_time,
                    **trial.params
                })
            
            print(f"Trial {trial.number} finished with loss: {final_loss:.6f}")
            
            return final_loss
            
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"Error in trial {trial.number}: {e}")
            return float('inf')
    
    def update_config_with_best_params(self, best_params):
        if self.config.mode != "prod":
            print("Config update skipped: Not in production mode")
            return False
            
        if not os.path.exists(self.config_path):
            print(f"Config update skipped: Config file not found at {self.config_path}")
            return False
            
        try:
            print(f"Updating config file with best hyperparameters...")
            
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            if 'supervised' not in config_data:
                config_data['supervised'] = {}
            if 'prod' not in config_data['supervised']:
                config_data['supervised']['prod'] = {}
                
            if 'model' not in config_data:
                config_data['model'] = {}
            if 'prod' not in config_data['model']:
                config_data['model']['prod'] = {}
                
            if 'data' not in config_data:
                config_data['data'] = {}
            if 'prod' not in config_data['data']:
                config_data['data']['prod'] = {}
                
            config_data['supervised']['prod']['lr'] = best_params.get('lr', config_data['supervised']['prod'].get('lr', 0.001))
            config_data['supervised']['prod']['weight_decay'] = best_params.get('weight_decay', config_data['supervised']['prod'].get('weight_decay', 0.0001))
            config_data['data']['prod']['batch'] = best_params.get('batch_size', config_data['data']['prod'].get('batch', 128))
            config_data['supervised']['prod']['epochs'] = best_params.get('epochs', config_data['supervised']['prod'].get('epochs', 20))
            config_data['supervised']['prod']['optimizer'] = best_params.get('optimizer', config_data['supervised']['prod'].get('optimizer', 'adamw'))
            config_data['supervised']['prod']['scheduler'] = best_params.get('scheduler', config_data['supervised']['prod'].get('scheduler', 'onecycle'))
            config_data['supervised']['prod']['grad_clip'] = best_params.get('grad_clip', config_data['supervised']['prod'].get('grad_clip', 1.0))
            config_data['supervised']['prod']['accum_steps'] = best_params.get('accum_steps', config_data['supervised']['prod'].get('accum_steps', 2))
            config_data['supervised']['prod']['policy_weight'] = best_params.get('policy_weight', config_data['supervised']['prod'].get('policy_weight', 1.0))
            config_data['supervised']['prod']['value_weight'] = best_params.get('value_weight', config_data['supervised']['prod'].get('value_weight', 1.0))
            config_data['model']['prod']['channels'] = best_params.get('channels', config_data['model']['prod'].get('channels', 64))
            
            if 'momentum' in best_params:
                config_data['supervised']['prod']['momentum'] = best_params['momentum']
                
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                
            print(f"Config file updated successfully at {self.config_path}")
            
            backup_path = f"{self.config_path}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
            with open(backup_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                
            print(f"Config backup created at {backup_path}")
            return True
            
        except Exception as e:
            print(f"Error updating config file: {e}")
            return False
    
    def run(self):
        if not self.setup():
            return False
        
        try:
            if wandb.run is not None:
                trials_table = wandb.Table(columns=[
                    "trial", "value", "lr", "weight_decay", "batch_size",
                    "optimizer", "scheduler", "epochs", "policy_weight", "value_weight",
                    "grad_clip", "channels", "time"
                ])
            
            try:
                optuna.delete_study(study_name=self.study_name, storage=f"sqlite:///{self.db_name}")
            except Exception:
                pass
            
            study = optuna.create_study(
                study_name=self.study_name,
                storage=f"sqlite:///{self.db_name}",
                direction="minimize",
                sampler=TPESampler(seed=self.seed),
                pruner=MedianPruner(n_warmup_steps=5),
                load_if_exists=False
            )
            
            print(f"Starting hyperparameter optimization with {self.trials} trials")
            print(f"Timeout: {self.timeout} seconds")
            print(f"Parallel jobs: {self.jobs}")
            
            study.optimize(
                self.objective,
                n_trials=self.trials,
                timeout=self.timeout if self.timeout > 0 else None,
                n_jobs=self.jobs,
                show_progress_bar=True
            )
            
            best_trial = study.best_trial
            best_value = study.best_value
            
            print("\n" + "=" * 50)
            print(f"Best trial: {best_trial.number}")
            print(f"Best value (composite loss): {best_value:.6f}")
            print("Best hyperparameters:")
            for param, value in best_trial.params.items():
                print(f"  {param}: {value}")
            print("User attributes:")
            for key, value in best_trial.user_attrs.items():
                print(f"  {key}: {value}")
            print("=" * 50)
            
            self.update_config_with_best_params(best_trial.params)
            
            with open(os.path.join(self.results_dir, "best_trial.txt"), "w") as f:
                f.write(f"Best Value (Composite Loss) = {best_value:.6f}\n")
                f.write(f"Best Trial = {best_trial.number}\n\n")
                
                f.write("Parameters:\n")
                for k, v in best_trial.params.items():
                    f.write(f"{k}: {v}\n")
                
                f.write("\nUser Attributes:\n")
                for k, v in best_trial.user_attrs.items():
                    f.write(f"{k}: {v}\n")
            
            if wandb.run is not None:
                wandb.run.summary.update({
                    "best_trial": best_trial.number,
                    "best_loss": best_value,
                    **{f"best_{k}": v for k, v in best_trial.params.items()},
                    "best_time": best_trial.user_attrs.get("time", 0)
                })
                
                for trial in study.trials:
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        row = [
                            trial.number,
                            trial.value,
                            trial.params.get("lr", None),
                            trial.params.get("weight_decay", None),
                            trial.params.get("batch_size", None),
                            trial.params.get("optimizer", None),
                            trial.params.get("scheduler", None),
                            trial.params.get("epochs", None),
                            trial.params.get("policy_weight", None),
                            trial.params.get("value_weight", None),
                            trial.params.get("grad_clip", None),
                            trial.params.get("channels", None),
                            trial.user_attrs.get("time", None)
                        ]
                        trials_table.add_data(*row)
                
                try:
                    importance = optuna.importance.get_param_importances(study)
                    importance_table = wandb.Table(columns=["parameter", "importance"])
                    
                    for param, score in importance.items():
                        importance_table.add_data(param, score)
                    
                    wandb.log({
                        "param_importance": wandb.plot.bar(
                            importance_table, "parameter", "importance", 
                            title="Parameter Importance"
                        ),
                        "trials": trials_table
                    })
                except Exception as e:
                    print(f"Error calculating parameter importance: {e}")
                
                try:
                    fig = optuna.visualization.plot_parallel_coordinate(study)
                    wandb.log({"parallel_coord": wandb.Image(fig)})
                except Exception as e:
                    print(f"Error creating parallel coordinate plot: {e}")
                
                wandb.finish()
            
            best_trial_path = os.path.join(self.results_dir, "best_trial.txt")
            db_path = self.db_name
            print(f"Saved hyperopt results to {best_trial_path} and {db_path}")
            
            return True
            
        except Exception as e:
            print(f"Error during hyperparameter optimization: {e}")
            if wandb.run is not None:
                wandb.finish()
            return False