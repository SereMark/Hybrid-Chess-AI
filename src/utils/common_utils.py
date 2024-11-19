import time

def format_time_left(seconds):
    days = seconds // 86400
    remainder = seconds % 86400
    hours = remainder // 3600
    minutes = (remainder % 3600) // 60
    secs = remainder % 60
    if days >= 1:
        day_str = f"{int(days)}d "
        return f"{day_str}{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    else:
        return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

def log_message(message, log_callback=None):
    if log_callback:
        log_callback(message)

def should_stop(stop_event):
    return stop_event.is_set()

def wait_if_paused(pause_event):
    while not pause_event.is_set():
        time.sleep(0.1)

def initialize_random_seeds(random_seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False