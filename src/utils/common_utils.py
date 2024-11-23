import time, random, numpy as np, torch

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
    else:
        print(message)

def should_stop(stop_event):
    return stop_event.is_set()

def wait_if_paused(pause_event):
    while not pause_event.is_set():
        time.sleep(0.1)

def initialize_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def estimate_batch_size(model, device, desired_effective_batch_size=256, max_batch_size=1024, min_batch_size=32):
    try:
        if device.type == 'cuda':
            batch_size = min_batch_size
            while batch_size <= max_batch_size:
                try:
                    torch.cuda.empty_cache()
                    inputs = torch.randn(batch_size, 20, 8, 8).to(device)
                    with torch.no_grad():
                        _ = model(inputs)
                    batch_size *= 2
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        torch.cuda.empty_cache()
                        batch_size = max(batch_size // 2, min_batch_size)
                        break
                    else:
                        raise e
            batch_size = max(min(batch_size, max_batch_size), min_batch_size)
            return batch_size
        else:
            return desired_effective_batch_size
    except Exception as e:
        print(f"Failed to estimate batch size: {e}. Using default batch size of 128.")
        return 128