import functools
import logging
import time
from collections.abc import Callable
from typing import ClassVar

import torch
from config import config


class ColoredFormatter(logging.Formatter):
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[90m",
        "INFO": "\033[37m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"

    def format(self, record):
        if not config.logging.use_colors:
            return f"{self.formatTime(record, '%H:%M:%S')} {record.getMessage()}"

        timestamp = f"{self.DIM}{self.formatTime(record, '%H:%M:%S')}{self.RESET}"
        message = record.getMessage()

        if record.levelname == "ERROR":
            return f"{timestamp} {self.COLORS['ERROR']}ERROR: {message}{self.RESET}"
        elif record.levelname == "WARNING":
            return f"{timestamp} {self.COLORS['WARNING']}WARNING: {message}{self.RESET}"
        else:
            return f"{timestamp} {message}"


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("chess_ai")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ColoredFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_system_status() -> dict[str, float]:
    logger = logging.getLogger("chess_ai.system")
    status = {
        "gpu_memory_gb": 0.0,
        "gpu_utilization": 0.0,
        "gpu_available": False,
    }

    if torch.cuda.is_available():
        try:
            status["gpu_available"] = True
            status["gpu_memory_gb"] = (
                torch.cuda.memory_allocated() / config.system.bytes_to_gb
            )

            total_memory = torch.cuda.get_device_properties(0).total_memory
            status["gpu_utilization"] = (
                (torch.cuda.memory_allocated() / total_memory)
                * config.system.percentage_multiplier
                if total_memory > 0
                else 0.0
            )
        except RuntimeError as e:
            logger.warning(f"GPU monitoring failed: {e}")

    return status


def monitor_gpu_memory() -> dict[str, float]:
    status = get_system_status()
    return {
        "current_memory_gb": status["gpu_memory_gb"],
        "max_memory_gb": torch.cuda.max_memory_allocated() / config.system.bytes_to_gb
        if torch.cuda.is_available()
        else 0.0,
        "cached_memory_gb": torch.cuda.memory_reserved() / config.system.bytes_to_gb
        if torch.cuda.is_available()
        else 0.0,
        "memory_utilization": status["gpu_utilization"],
    }


def log_slow_operation(func_name: str, duration: float) -> None:
    if duration > config.system.slow_operation_threshold:
        logger = logging.getLogger("chess_ai.performance")
        logger.info(f"{func_name}: {duration:.2f}s")


def log_operation_error(func_name: str, duration: float, error: Exception) -> None:
    logger = logging.getLogger("chess_ai.performance")
    logger.error(f"Error in {func_name} after {duration:.2f}s: {error}")


def performance_monitor(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            log_slow_operation(func.__name__, execution_time)
            return result
        except (RuntimeError, ValueError) as e:
            execution_time = time.perf_counter() - start_time
            log_operation_error(func.__name__, execution_time, e)
            raise

    return wrapper


class ConsoleMetricsLogger:
    def __init__(self):
        self.logger = logging.getLogger("chess_ai")
        self.start_time = time.time()
        self.loss_history = []
        self.iteration_times = []
        self.best_loss = float("inf")
        self.last_loss = 0.0
        self.games_total = 0
        self.training_steps_total = 0

        self._print_header()

    def _print_header(self):
        if config.logging.use_colors:
            title = f"{ColoredFormatter.CYAN}Chess AI Training{ColoredFormatter.RESET}"
            timestamp = f"{ColoredFormatter.DIM}{time.strftime('%Y-%m-%d %H:%M:%S')}{ColoredFormatter.RESET}"
        else:
            title = "Chess AI Training"
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        self.logger.info(f"{title} | {timestamp}")

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def _get_trend_symbol(self) -> str:
        if len(self.loss_history) < 3:
            return ""

        recent = self.loss_history[-3:]
        if recent[-1] < recent[0] * 0.98:
            return (
                f" {ColoredFormatter.GREEN}DOWN{ColoredFormatter.RESET}"
                if config.logging.use_colors
                else " DOWN"
            )
        elif recent[-1] > recent[0] * 1.02:
            return (
                f" {ColoredFormatter.COLORS['ERROR']}UP{ColoredFormatter.RESET}"
                if config.logging.use_colors
                else " UP"
            )
        else:
            return (
                f" {ColoredFormatter.DIM}STABLE{ColoredFormatter.RESET}"
                if config.logging.use_colors
                else " STABLE"
            )

    def log_iteration(
        self, losses: dict[str, float], metrics: dict[str, int | float], iteration: int
    ) -> None:
        total_loss = losses.get("total_loss", 0.0)
        self.loss_history.append(total_loss)

        if total_loss < self.best_loss:
            self.best_loss = total_loss
            best_marker = "*"
        else:
            best_marker = ""

        trend = self._get_trend_symbol()

        iter_time = metrics.get("iteration_time", 0.0)
        if iter_time > 0:
            self.iteration_times.append(iter_time)

        games_completed = metrics.get("games_completed", 0)
        games_total = metrics.get("games_total", 0)
        buffer_size = metrics.get("buffer_size", 0)
        buffer_pct = (
            int(100 * buffer_size / config.training.buffer_size)
            if config.training.buffer_size > 0
            else 0
        )

        self.games_total += games_total
        self.training_steps_total = metrics.get(
            "training_steps", self.training_steps_total
        )

        parts = [
            f"[{iteration:03d}/{config.training.iterations}]",
            f"Loss: {total_loss:.4f}{trend}{best_marker}",
            f"Games: {games_completed}/{games_total}",
            f"Buf: {buffer_pct}%",
            f"Time: {self._format_time(iter_time)}",
        ]

        if config.logging.verbosity == "normal":
            lr = losses.get("learning_rate", 0)
            parts.append(f"LR: {lr:.5f}")

        self.logger.info(" | ".join(parts))

        if iteration % config.logging.detail_interval == 0 and iteration > 0:
            self._log_detailed_stats(losses, metrics, iteration)

    def _log_detailed_stats(
        self, losses: dict[str, float], metrics: dict[str, int | float], iteration: int
    ) -> None:
        elapsed = time.time() - self.start_time
        avg_loss = (
            sum(self.loss_history[-10:]) / min(10, len(self.loss_history))
            if self.loss_history
            else 0
        )

        self.logger.info("")
        self.logger.info(f"{ColoredFormatter.DIM}{'-' * 60}{ColoredFormatter.RESET}")
        self.logger.info(
            f"Progress Report - Iteration {iteration}/{config.training.iterations}"
        )
        self.logger.info(f"{ColoredFormatter.DIM}{'-' * 60}{ColoredFormatter.RESET}")

        progress_pct = iteration / config.training.iterations * 100
        eta = (
            (elapsed / iteration) * (config.training.iterations - iteration)
            if iteration > 0
            else 0
        )

        self.logger.info(
            f"Progress:  {progress_pct:.1f}% | Elapsed: {self._format_time(elapsed)} | ETA: {self._format_time(eta)}"
        )
        self.logger.info(
            f"Loss:      Current: {losses.get('total_loss', 0):.4f} | Avg: {avg_loss:.4f} | Best: {self.best_loss:.4f}"
        )
        self.logger.info(
            f"Training:  Games: {self.games_total} | Steps: {self.training_steps_total}"
        )

        if self.iteration_times:
            avg_iter_time = sum(self.iteration_times[-10:]) / min(
                10, len(self.iteration_times)
            )
            self.logger.info(
                f"Speed:     {avg_iter_time:.1f}s/iter | {self.games_total / elapsed * 60:.1f} games/min"
            )

        self.logger.info("")

    def log_system(self, system_status: dict[str, float], iteration: int) -> None:
        if config.logging.verbosity != "verbose":
            return

        gpu_gb = system_status.get("gpu_memory_gb", 0)
        gpu_pct = system_status.get("gpu_utilization", 0)

        self.logger.info(f"System: GPU {gpu_gb:.1f}GB ({gpu_pct:.0f}%)")

    def log_games(
        self,
        completed_games: int,
        timeout_games: int,
        avg_length: float,
        game_results: dict[str, int],
        iteration: int,
        **kwargs,
    ) -> None:
        if config.logging.verbosity == "minimal":
            return

        total = completed_games + timeout_games
        if total == 0:
            return

        completion_rate = completed_games / total * 100
        wins = game_results.get("1-0", 0)
        losses = game_results.get("0-1", 0)
        draws = game_results.get("1/2-1/2", 0)

        self.logger.info(
            f"Games: {completed_games}/{total} ({completion_rate:.0f}%) | W{wins} L{losses} D{draws} | Avg: {avg_length:.0f} moves"
        )

    def log_model(self, model: torch.nn.Module, iteration: int) -> None:
        if config.logging.verbosity != "verbose":
            return

        params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Model: {params:,} parameters")

    def log_params(
        self,
        hyperparams: dict[str, int | float | str],
        metrics: dict[str, float] | None = None,
    ) -> None:
        if config.logging.verbosity == "minimal":
            return

        self.logger.info("Configuration:")
        for key, value in hyperparams.items():
            self.logger.info(f"  {key}: {value}")

        if metrics:
            self.logger.info("\nFinal Metrics:")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value:.4f}")

    def log_milestone(
        self,
        milestone: str,
        iteration: int,
        additional_data: dict[str, int | float] | None = None,
    ) -> None:
        marker = (
            f"{ColoredFormatter.GREEN}DONE{ColoredFormatter.RESET}"
            if config.logging.use_colors
            else "DONE"
        )
        self.logger.info(f"\n{marker} {milestone}")

        if additional_data:
            for key, value in additional_data.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")

    def close(self) -> None:
        elapsed = time.time() - self.start_time

        self.logger.info(f"\n{ColoredFormatter.DIM}{'=' * 60}{ColoredFormatter.RESET}")
        self.logger.info("Training Complete")
        self.logger.info(f"{ColoredFormatter.DIM}{'=' * 60}{ColoredFormatter.RESET}")

        self.logger.info(f"Total Time: {self._format_time(elapsed)}")

        if self.loss_history:
            initial = self.loss_history[0]
            final = self.loss_history[-1]
            improvement = ((initial - final) / initial * 100) if initial != 0 else 0

            self.logger.info(f"Loss: {initial:.4f} → {final:.4f} ({improvement:+.1f}%)")
            self.logger.info(f"Best: {self.best_loss:.4f}")

        self.logger.info(f"Games: {self.games_total}")
        self.logger.info(f"Steps: {self.training_steps_total}")

        if self.iteration_times:
            avg_speed = sum(self.iteration_times) / len(self.iteration_times)
            self.logger.info(f"Avg Speed: {avg_speed:.1f}s/iter")


Logger = ConsoleMetricsLogger
