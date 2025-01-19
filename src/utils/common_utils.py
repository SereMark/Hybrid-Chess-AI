import os
import time
import chess

def format_time_left(seconds: float) -> str:
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    if days >= 1:
        return f"{int(days)}d {int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

def update_progress_time_left(progress_signal, time_left_signal, start_time: float, current_step: int, total_steps: int) -> None:
    # If total steps is invalid, reset signals and return.
    if total_steps <= 0:
        if progress_signal:
            progress_signal.emit(0)
        if time_left_signal:
            time_left_signal.emit("Calculating...")
        return

    # Convert step to percentage and emit to progress bar.
    progress = max(0, min(100, int((current_step / total_steps) * 100)))
    if progress_signal:
        progress_signal.emit(progress)

    # Estimate time left based on elapsed time and steps remaining.
    elapsed = time.time() - start_time
    if current_step > 0:
        steps_left = max(0, total_steps - current_step)
        time_left = max(0, (elapsed / current_step) * steps_left)
        time_left_str = format_time_left(time_left)
        if time_left_signal:
            time_left_signal.emit(time_left_str)
    else:
        # If we haven't made progress, we can't estimate time yet.
        if time_left_signal:
            time_left_signal.emit("Calculating...")

def wait_if_paused(pause_event):
    while not pause_event.is_set():
        time.sleep(0.1)

def estimate_total_games(file_paths, avg_game_size=5000, max_games=None, logger=None) -> int:
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

def get_game_result(board: chess.Board) -> float:
    result_map = {'1-0': 1.0, '0-1': -1.0, '1/2-1/2': 0.0}
    return result_map.get(board.result(), 0.0)

def parse_game_result(result: str) -> float:
    result_map = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
    return result_map.get(result, None)

def determine_outcome(result: str) -> str:
    outcome_map = {'1-0': 'win', '0-1': 'loss', '1/2-1/2': 'draw'}
    return outcome_map.get(result)