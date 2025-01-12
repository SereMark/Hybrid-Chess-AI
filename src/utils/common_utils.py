import time
import random
import numpy as np
import torch
import torch.optim as optim
import os

def initialize_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_optimizer(model, optimizer_type, learning_rate, weight_decay, logger=None):
    optimizer_type = optimizer_type.lower()
    optimizers = {
        'adamw': optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9),
        'adam': optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
    }

    optimizer = optimizers.get(optimizer_type)
    if optimizer is None:
        if logger:
            logger.warning(f"Unsupported optimizer type: {optimizer_type}. Using AdamW by default.")
        optimizer = optimizers['adamw']
    return optimizer

def initialize_scheduler(optimizer, scheduler_type, total_steps=None, logger=None):
    scheduler_type = scheduler_type.lower()
    schedulers = {
        'cosineannealingwarmrestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
        'steplr': optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
        'cosineannealing': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
        'none': None,
    }

    if scheduler_type == 'onecyclelr':
        if total_steps is None:
            if logger:
                logger.error("total_steps must be provided for OneCycleLR scheduler.")
            raise ValueError("total_steps must be provided for OneCycleLR scheduler.")
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'], total_steps=total_steps)

    scheduler = schedulers.get(scheduler_type)
    if scheduler is None:
        if logger:
            logger.warning(f"Unsupported scheduler type: {scheduler_type}. Using CosineAnnealingWarmRestarts by default.")
        scheduler = schedulers['cosineannealingwarmrestarts']
    return scheduler

def format_time_left(seconds):
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    if days >= 1:
        return f"{int(days)}d {int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

def update_progress_time_left(progress_signal, time_left_signal, start_time, current_step, total_steps):
    if total_steps <= 0:
        if progress_signal:
            progress_signal.emit(0)
        if time_left_signal:
            time_left_signal.emit("Calculating...")
        return

    progress = max(0, min(100, int((current_step / total_steps) * 100)))
    if progress_signal:
        progress_signal.emit(progress)

    elapsed = time.time() - start_time
    if current_step > 0:
        steps_left = max(0, total_steps - current_step)
        time_left = max(0, (elapsed / current_step) * steps_left)
        time_left_str = format_time_left(time_left)
        if time_left_signal:
            time_left_signal.emit(time_left_str)
    else:
        if time_left_signal:
            time_left_signal.emit("Calculating...")

def wait_if_paused(pause_event):
    while not pause_event.is_set():
        time.sleep(0.1)

def estimate_total_games(file_paths, avg_game_size=5000, max_games=None, logger=None):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    total_games = 0
    for file_path in file_paths:
        try:
            if not os.path.isfile(file_path):
                if logger:
                    logger.warning(f"File not found: {file_path}. Skipping.")
                continue
            fsize = os.path.getsize(file_path)
            estimated_games = fsize // avg_game_size
            total_games += estimated_games
        except Exception as e:
            if logger:
                logger.error(f"Error estimating games for {file_path}: {e}")
    
    if max_games is not None:
        return min(total_games, max_games)
    return total_games