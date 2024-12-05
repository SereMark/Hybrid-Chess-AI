import time, random, numpy as np, torch, torch.optim as optim

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

def estimate_batch_size(model, device, desired_effective_batch_size=256, max_batch_size=1024, min_batch_size=32, log_fn=None):
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
        if log_fn:
            log_fn(f"Failed to estimate batch size: {e}. Using default batch size of 128.")
        return 128

def initialize_model(model_class, num_moves, device, automatic_batch_size=False, desired_effective_batch_size=256, log_fn=None):
    model = model_class(num_moves=num_moves).to(device)
    if automatic_batch_size:
        batch_size = estimate_batch_size(model, device, desired_effective_batch_size, log_fn=log_fn)
        if log_fn:
            log_fn(f"Automatic batch size estimation: Using batch size of {batch_size}.")
    else:
        batch_size = None
        if log_fn:
            log_fn("Using manual batch size.")
    return model, batch_size

def initialize_optimizer(model, optimizer_type, learning_rate, weight_decay, log_fn=None):
    optimizer_type_lower = optimizer_type.lower()
    if optimizer_type_lower == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type_lower == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type_lower == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        if log_fn:
            log_fn(f"Unsupported optimizer type: {optimizer_type}. Using AdamW by default.")
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def initialize_scheduler(optimizer, scheduler_type, total_steps=None, log_fn=None):
    scheduler_type_lower = scheduler_type.lower()
    if scheduler_type_lower == 'cosineannealingwarmrestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif scheduler_type_lower == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type_lower == 'onecyclelr':
        if total_steps is None:
            raise ValueError("total_steps must be provided for OneCycleLR scheduler.")
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'], total_steps=total_steps)
    elif scheduler_type_lower == 'cosineannealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif scheduler_type_lower == 'none':
        scheduler = None
    else:
        if log_fn:
            log_fn(f"Unsupported scheduler type: {scheduler_type}. Using CosineAnnealingWarmRestarts by default.")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    return scheduler