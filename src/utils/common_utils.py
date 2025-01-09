import time, random, numpy as np, torch, torch.optim as optim

def format_time_left(seconds):
    days = seconds // 86400
    remainder = seconds % 86400
    hours = remainder // 3600
    minutes = (remainder % 3600) // 60
    secs = remainder % 60
    if days >= 1:
        return f"{int(days)}d {int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
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

def estimate_batch_size(model, device, desired_effective_batch_size=256, max_batch_size=1024, min_batch_size=32, logger=None):
    try:
        if device.type == 'cuda':
            test_size = min_batch_size
            while test_size <= max_batch_size:
                try:
                    torch.cuda.empty_cache()
                    test_input = torch.randn(test_size, 20, 8, 8).to(device)
                    with torch.no_grad():
                        _ = model(test_input)
                    test_size *= 2
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        torch.cuda.empty_cache()
                        test_size = max(test_size // 2, min_batch_size)
                        if logger:
                            logger.warning(f"Out of memory at batch size {test_size * 2}. Reducing to {test_size}.")
                        break
                    else:
                        raise e
            final_size = max(min(test_size, max_batch_size), min_batch_size)
            return final_size
        else:
            return desired_effective_batch_size
    except Exception as e:
        if logger:
            logger.error(f"Failed to estimate batch size: {e}. Using default batch size of 128.")
        return 128

def initialize_model(
    model_class,
    num_moves,
    device,
    automatic_batch_size=False,
    desired_effective_batch_size=256,
    logger=None,
    filters=64,
    res_blocks=5,
    inplace_relu=True
):
    model = model_class(
        num_moves=num_moves,
        filters=filters,
        res_blocks=res_blocks,
        inplace_relu=inplace_relu
    ).to(device)
    if automatic_batch_size:
        batch_size = estimate_batch_size(
            model, device,
            desired_effective_batch_size=desired_effective_batch_size,
            logger=logger
        )
        if logger:
            logger.info(f"Automatic batch size estimation: Using batch size of {batch_size}.")
    else:
        batch_size = None
        if logger:
            logger.info("Using manual batch size.")
    return model, batch_size

def initialize_optimizer(model, optimizer_type, learning_rate, weight_decay, logger=None):
    ot = optimizer_type.lower()
    if ot == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif ot == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    elif ot == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        if logger:
            logger.warning(f"Unsupported optimizer type: {optimizer_type}. Using AdamW by default.")
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def initialize_scheduler(optimizer, scheduler_type, total_steps=None, logger=None):
    s_type = scheduler_type.lower()
    if s_type == 'cosineannealingwarmrestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif s_type == 'steplr':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif s_type == 'onecyclelr':
        if total_steps is None:
            if logger:
                logger.error("total_steps must be provided for OneCycleLR scheduler.")
            raise ValueError("total_steps must be provided for OneCycleLR scheduler.")
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=total_steps
        )
    elif s_type == 'cosineannealing':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif s_type == 'none':
        return None
    else:
        if logger:
            logger.warning(f"Unsupported scheduler type: {scheduler_type}. Using CosineAnnealingWarmRestarts by default.")
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

def update_progress_time_left(progress_signal, time_left_signal, start_time, current_step, total_steps):
    if total_steps <= 0:
        if progress_signal:
            progress_signal.emit(0)
        if time_left_signal:
            time_left_signal.emit("Calculating...")
        return
    progress = int((current_step / total_steps) * 100)
    if progress < 0:
        progress = 0
    if progress > 100:
        progress = 100
    if progress_signal:
        progress_signal.emit(progress)
    elapsed = time.time() - start_time
    if current_step > 0:
        steps_left = total_steps - current_step
        if steps_left < 0:
            steps_left = 0
        time_left = (elapsed / current_step) * steps_left
        if time_left < 0:
            time_left = 0
        time_left_str = format_time_left(time_left)
        if time_left_signal:
            time_left_signal.emit(time_left_str)
    else:
        if time_left_signal:
            time_left_signal.emit("Calculating...")