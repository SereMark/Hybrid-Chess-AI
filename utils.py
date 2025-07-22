import functools
import logging
import time
from collections.abc import Callable
from typing import ClassVar

import torch
from config import config


class ColoredFormatter(logging.Formatter):
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        timestamp = self.formatTime(record, "%H:%M:%S")
        level_color = self.COLORS.get(record.levelname, "")
        level = f"{level_color}{record.levelname[:4]}{self.RESET}"

        message = record.getMessage()
        if any(
            keyword in message
            for keyword in ["[INIT]", "[MILESTONE]", "[METRICS]", "[GAMES]", "[SYSTEM]"]
        ):
            message = f"{self.BOLD}{message}{self.RESET}"

        return f"{timestamp} | {level} | {message}"


def format_log_record(record: logging.LogRecord) -> str:
    formatter = ColoredFormatter()
    return formatter.format(record)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("chess_ai")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ColoredFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.info("[INIT] Logging system initialized")

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
        self.logger: logging.Logger = logging.getLogger("chess_ai.metrics")
        self.start_time: float = time.time()
        self.loss_history: list[float] = []
        self.completion_history: list[float] = []
        self.iteration_times: list[float] = []
        self.mcts_efficiency: list[float] = []
        self.total_positions_processed: int = 0

        self.logger.info("[INIT] Metrics logger initialized")
        self.logger.info("=" * 80)
        self.logger.info("[INIT] Training started")
        self.logger.info(f"[INIT] Session: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

    def log_iteration(
        self, losses: dict[str, float], metrics: dict[str, int | float], iteration: int
    ) -> None:
        total_loss = losses.get("total_loss", 0.0)
        value_loss = losses.get("value_loss", 0.0)
        policy_loss = losses.get("policy_loss", 0.0)

        self.loss_history.append(total_loss)

        trend = "STABLE"
        if len(self.loss_history) >= 3:
            recent = self.loss_history[-3:]
            if recent[-1] < recent[0] * 0.95:
                trend = "IMPROVING"
            elif recent[-1] > recent[0] * 1.05:
                trend = "DIVERGING"

        loss_std = 0.0
        if len(self.loss_history) >= 5:
            recent_losses = self.loss_history[-5:]
            loss_std = sum(
                (x - sum(recent_losses) / len(recent_losses)) ** 2
                for x in recent_losses
            ) / len(recent_losses)
            loss_std = loss_std**0.5

        iter_time = metrics.get("iteration_time", 0.0)
        if iter_time > 0:
            self.iteration_times.append(iter_time)

        games_played = metrics.get("games_played", 0)
        avg_game_length = metrics.get("avg_game_length", 0)
        positions_this_iter = 0
        if games_played > 0 and avg_game_length > 0:
            positions_this_iter = int(games_played * avg_game_length)
            self.total_positions_processed += positions_this_iter

        loss_str = f"Loss: {total_loss:.4f} (V:{value_loss:.4f} P:{policy_loss:.4f}) | Trend: {trend}"
        if loss_std > 0:
            loss_str += f" | Stability: {loss_std:.4f}"

        self.logger.info(f"[METRICS] [{iteration:3d}] {loss_str}")

        lr = losses.get("learning_rate", 0)
        grad_norm = losses.get("grad_norm", 0)
        opt_str = f"Optimizer: LR={lr:.5f} | GradNorm={grad_norm:.4f}"

        if "policy_entropy_avg" in losses:
            entropy = losses["policy_entropy_avg"]
            correlation = losses.get("value_correlation", 0)
            top3_mass = losses.get("top3_probability_mass_avg", 0)
            opt_str += f" | PolicyEntropy={entropy:.3f} | ValueCorr={correlation:.3f} | Top3Mass={top3_mass:.3f}"

        self.logger.info(f"         {opt_str}")

        if iter_time > 0:
            buffer_size = metrics.get("buffer_size", 0)
            buffer_util = f"{buffer_size}/{config.training.buffer_size} ({100 * buffer_size / config.training.buffer_size:.1f}%)"

            perf_str = f"Performance: Time={iter_time:.1f}s | Buffer={buffer_util}"

            if games_played > 0:
                games_per_min = games_played * 60 / iter_time
                perf_str += f" | Games/min={games_per_min:.1f}"

                if avg_game_length > 0:
                    positions_per_sec = positions_this_iter / iter_time
                    perf_str += f" | Pos/sec={positions_per_sec:.1f}"

            training_steps = metrics.get("training_steps", 0)
            if training_steps > 0:
                perf_str += f" | TrainSteps={training_steps}"

            self.logger.info(f"         {perf_str}")

    def log_mcts(
        self,
        search_time: float,
        eval_per_sec: float,
        positions: int,
        avg_legal_moves: float,
        avg_pieces: float,
        batch_size: int = 0,
        cache_hits: int = 0,
        total_searches: int = 0,
        avg_depth: float = 0.0,
        nodes_expanded: int = 0,
    ) -> None:
        if search_time > 0:
            efficiency = eval_per_sec / positions if positions > 0 else 0
            self.mcts_efficiency.append(efficiency)

            searches_per_sec = total_searches / search_time if search_time > 0 else 0
            nodes_per_search = (
                nodes_expanded / total_searches if total_searches > 0 else 0
            )
            cache_hit_rate = (
                cache_hits / total_searches * 100 if total_searches > 0 else 0
            )

            complexity_factor = (
                avg_legal_moves * (avg_pieces / 32.0) if avg_pieces > 0 else 0
            )

            mcts_str = (
                f"MCTS: SearchTime={search_time:.2f}s | Eval/sec={eval_per_sec:.0f} | "
                f"Positions={positions} | Efficiency={efficiency:.2f}"
            )

            detail_str = (
                f"      Searches/sec={searches_per_sec:.1f} | "
                f"Nodes/search={nodes_per_search:.1f} | "
                f"CacheHit={cache_hit_rate:.1f}% | "
                f"AvgDepth={avg_depth:.1f}"
            )

            board_str = (
                f"      BoardComplexity: LegalMoves={avg_legal_moves:.1f} | "
                f"Pieces={avg_pieces:.1f} | "
                f"ComplexityFactor={complexity_factor:.2f}"
            )

            if batch_size > 0:
                detail_str += f" | BatchSize={batch_size}"

            self.logger.info(f"[MCTS] {mcts_str}")
            self.logger.info(detail_str)
            self.logger.info(board_str)

    def log_gpu(
        self, gpu_memory: float, gpu_utilization: float, iteration: int
    ) -> None:
        uptime = time.time() - self.start_time
        gpu_status = (
            "OK"
            if gpu_utilization < 80
            else "High"
            if gpu_utilization < 95
            else "Critical"
        )

        uptime_str = (
            f"{uptime:.1f}s"
            if uptime < 60
            else f"{uptime / 60:.1f}min"
            if uptime < 3600
            else f"{uptime / 3600:.1f}h"
        )

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            max_memory = torch.cuda.max_memory_allocated() / (1024**3)

            memory_efficiency = (allocated / reserved * 100) if reserved > 0 else 0
            peak_usage = max_memory

            gpu_str = (
                f"GPU: Status={gpu_status} | Memory={gpu_memory:.1f}GB | "
                f"Utilization={gpu_utilization:.0f}% | Uptime={uptime_str}"
            )

            memory_str = (
                f"     Memory: Allocated={allocated:.2f}GB | "
                f"Reserved={reserved:.2f}GB | Peak={peak_usage:.2f}GB | "
                f"Efficiency={memory_efficiency:.1f}%"
            )

            if hasattr(self, "iteration_times") and self.iteration_times:
                avg_iter_time = sum(self.iteration_times[-5:]) / len(
                    self.iteration_times[-5:]
                )
                positions_per_gb = (
                    self.total_positions_processed / gpu_memory if gpu_memory > 0 else 0
                )
                memory_str += f" | AvgIterTime={avg_iter_time:.1f}s | Pos/GB={positions_per_gb:.0f}"

            self.logger.info(f"[SYSTEM] {gpu_str}")
            self.logger.info(f"         {memory_str}")
        else:
            self.logger.info(
                f"[SYSTEM] GPU: UNAVAILABLE | CPU Mode | Uptime: {uptime_str}"
            )

    def log_model(self, model: torch.nn.Module, iteration: int) -> None:
        total_grad_norm = 0.0
        layer_count = 0
        zero_grad_count = 0
        large_grad_count = 0
        param_stats = {"weights": 0, "biases": 0, "total_params": 0}

        layer_grad_norms = []
        weight_magnitudes = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                param_stats["total_params"] += param.numel()

                if "bias" in name:
                    param_stats["biases"] += param.numel()
                else:
                    param_stats["weights"] += param.numel()

                if param.data is not None:
                    weight_mag = param.data.norm().item()
                    weight_magnitudes.append(weight_mag)

                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    layer_grad_norms.append(grad_norm)
                    total_grad_norm += grad_norm
                    layer_count += 1

                    if grad_norm < 1e-7:
                        zero_grad_count += 1
                    elif grad_norm > 10.0:
                        large_grad_count += 1

        if layer_count > 0:
            avg_grad_norm = total_grad_norm / layer_count
            max_grad_norm = max(layer_grad_norms)
            min_grad_norm = min(layer_grad_norms)

            grad_flow_health = "OK"
            if zero_grad_count > layer_count * 0.3:
                grad_flow_health = "Poor"
            elif large_grad_count > layer_count * 0.2:
                grad_flow_health = "EXPLODING"
            elif avg_grad_norm < 1e-5:
                grad_flow_health = "VANISHING"

            model_str = (
                f"Model: Layers={layer_count} | "
                f"Params={param_stats['total_params']:,} | "
                f"GradFlow={grad_flow_health}"
            )

            grad_str = (
                f"      Gradients: Avg={avg_grad_norm:.4f} | "
                f"Range=[{min_grad_norm:.4f}, {max_grad_norm:.4f}] | "
                f"ZeroGrads={zero_grad_count}/{layer_count} | "
                f"LargeGrads={large_grad_count}/{layer_count}"
            )

            if weight_magnitudes:
                avg_weight_mag = sum(weight_magnitudes) / len(weight_magnitudes)
                max_weight_mag = max(weight_magnitudes)
                weight_str = (
                    f"      Weights: AvgMagnitude={avg_weight_mag:.4f} | "
                    f"MaxMagnitude={max_weight_mag:.4f} | "
                    f"WeightParams={param_stats['weights']:,} | "
                    f"BiasParams={param_stats['biases']:,}"
                )
            else:
                weight_str = "      Weights: No weight data available"

            self.logger.info(f"[MODEL] {model_str}")
            self.logger.info(grad_str)
            self.logger.info(weight_str)

            if grad_flow_health != "HEALTHY":
                if grad_flow_health == "EXPLODING":
                    self.logger.warning(
                        f"[MODEL] WARNING: Gradient explosion detected (avg={avg_grad_norm:.2f})"
                    )
                elif grad_flow_health == "VANISHING":
                    self.logger.warning(
                        f"[MODEL] WARNING: Gradient vanishing detected (avg={avg_grad_norm:.2e})"
                    )
                elif grad_flow_health == "POOR_FLOW":
                    self.logger.warning(
                        f"[MODEL] WARNING: Poor gradient flow ({zero_grad_count}/{layer_count} zero gradients)"
                    )

    def log_games(
        self,
        completed_games: int,
        timeout_games: int,
        avg_length: float,
        game_results: dict[str, int],
        iteration: int,
        move_quality_avg: float = 0.0,
        positions_evaluated: int = 0,
        avg_eval_time: float = 0.0,
    ) -> None:
        total_games = completed_games + timeout_games

        if total_games > 0:
            completion_rate = completed_games / total_games * 100
            self.completion_history.append(completion_rate)

            completion_status = (
                "Good"
                if completion_rate > 80
                else "OK"
                if completion_rate > 60
                else "Poor"
                if completion_rate > 30
                else "Bad"
            )
            timeout_rate = timeout_games / total_games * 100

            wins = game_results.get("1-0", 0)
            losses = game_results.get("0-1", 0)
            draws = game_results.get("1/2-1/2", 0)

            decisive_games = wins + losses
            decisive_rate = (
                decisive_games / completed_games * 100 if completed_games > 0 else 0
            )
            win_rate = wins / completed_games * 100 if completed_games > 0 else 0

            game_str = (
                f"Games: Completed={completed_games}/{total_games} ({completion_rate:.0f}%) | "
                f"Status={completion_status} | Timeouts={timeout_games} ({timeout_rate:.0f}%)"
            )

            outcome_str = (
                f"      Outcomes: W={wins} L={losses} D={draws} | "
                f"WinRate={win_rate:.0f}% | DecisiveRate={decisive_rate:.0f}% | "
                f"AvgLength={avg_length:.1f}moves"
            )

            quality_str = ""
            if move_quality_avg > 0:
                quality_str += f" | MoveQuality={move_quality_avg:.3f}"
            if positions_evaluated > 0:
                quality_str += f" | PosEvaluated={positions_evaluated}"
            if avg_eval_time > 0:
                quality_str += f" | AvgEvalTime={avg_eval_time:.3f}s"

            if quality_str:
                outcome_str += quality_str

            if hasattr(self, "completion_history") and len(self.completion_history) > 1:
                recent_completion = sum(self.completion_history[-3:]) / len(
                    self.completion_history[-3:]
                )
                completion_trend = (
                    "IMPROVING"
                    if completion_rate > recent_completion
                    else "DECLINING"
                    if completion_rate < recent_completion * 0.9
                    else "STABLE"
                )
                outcome_str += f" | CompletionTrend={completion_trend}"

            self.logger.info(f"[GAMES] {game_str}")
            self.logger.info(f"        {outcome_str}")

    def log_milestone(
        self,
        milestone: str,
        iteration: int,
        additional_data: dict[str, int | float] | None = None,
    ) -> None:
        self.logger.info(f"[MILESTONE] [Iter {iteration}]: {milestone}")

        if additional_data:
            formatted = [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in additional_data.items()
            ]
            self.logger.info(f"            {' | '.join(formatted)}")

    def log_params(
        self,
        hyperparams: dict[str, int | float | str],
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.logger.info("[CONFIG] HYPERPARAMETERS:")
        for key, value in hyperparams.items():
            self.logger.info(f"         {key}: {value}")

        if metrics:
            self.logger.info("[CONFIG] FINAL METRICS:")
            for key, value in metrics.items():
                self.logger.info(f"         {key}: {value:.4f}")

    def log_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_count = param_count - trainable_count

        layer_types = {}
        for _name, module in model.named_modules():
            module_type = type(module).__name__
            layer_types[module_type] = layer_types.get(module_type, 0) + 1

        param_size_mb = param_count * 4 / (1024 * 1024)

        if hasattr(model, "hidden_dim") and isinstance(model.hidden_dim, int):
            complexity_score = model.hidden_dim * param_count / 1000000
        else:
            complexity_score = param_count / 1000000

        arch_str = (
            f"Architecture: TotalParams={param_count:,} | "
            f"Trainable={trainable_count:,} | Frozen={frozen_count:,}"
        )

        memory_str = (
            f"            Memory: ParamSize={param_size_mb:.1f}MB | "
            f"ComplexityScore={complexity_score:.2f}"
        )

        major_layers = {
            k: v for k, v in layer_types.items() if v > 1 and k not in ["Module"]
        }
        if major_layers:
            layer_breakdown = " | ".join([f"{k}={v}" for k, v in major_layers.items()])
            memory_str += f" | Layers: {layer_breakdown}"

        self.logger.info(f"[MODEL] {arch_str}")
        self.logger.info(f"        {memory_str}")

    def close(self) -> None:
        total_time = time.time() - self.start_time

        time_str = (
            f"{total_time:.1f}s"
            if total_time < 60
            else f"{total_time / 60:.1f}min"
            if total_time < 3600
            else f"{total_time / 3600:.1f}h"
        )

        self.logger.info("=" * 80)
        self.logger.info("[SUMMARY] Training completed")
        self.logger.info(f"[SUMMARY] Total time: {time_str}")

        if self.loss_history:
            initial_loss = self.loss_history[0]
            final_loss = self.loss_history[-1]
            best_loss = min(self.loss_history)
            worst_loss = max(self.loss_history)
            improvement = (
                ((initial_loss - final_loss) / initial_loss) * 100
                if initial_loss != 0
                else 0
            )

            loss_variance = sum(
                (x - sum(self.loss_history) / len(self.loss_history)) ** 2
                for x in self.loss_history
            ) / len(self.loss_history)
            loss_stability = 1.0 / (1.0 + loss_variance)

            loss_str = (
                f"Loss Analysis: {initial_loss:.4f} → {final_loss:.4f} | "
                f"Best: {best_loss:.4f} | Worst: {worst_loss:.4f} | "
                f"Improvement: {improvement:+.1f}% | Stability: {loss_stability:.3f}"
            )
            self.logger.info(f"[SUMMARY] {loss_str}")

        if self.completion_history:
            avg_completion = sum(self.completion_history) / len(self.completion_history)
            final_completion = self.completion_history[-1]
            best_completion = (
                max(self.completion_history) if self.completion_history else 0
            )
            completion_trend = (
                final_completion - self.completion_history[0]
                if len(self.completion_history) > 1
                else 0
            )

            completion_str = (
                f"Game Analysis: AvgCompletion={avg_completion:.1f}% | "
                f"Final={final_completion:.1f}% | Best={best_completion:.1f}% | "
                f"Trend={completion_trend:+.1f}%"
            )
            self.logger.info(f"[SUMMARY] {completion_str}")

        if hasattr(self, "iteration_times") and self.iteration_times:
            avg_iter_time = sum(self.iteration_times) / len(self.iteration_times)
            fastest_iter = min(self.iteration_times)
            slowest_iter = max(self.iteration_times)

            performance_str = (
                f"Performance: AvgIterTime={avg_iter_time:.1f}s | "
                f"Range=[{fastest_iter:.1f}s, {slowest_iter:.1f}s] | "
                f"TotalPositions={self.total_positions_processed:,}"
            )
            if total_time > 0:
                positions_per_sec = self.total_positions_processed / total_time
                performance_str += f" | OverallPos/sec={positions_per_sec:.1f}"
            self.logger.info(f"[SUMMARY] {performance_str}")

        if hasattr(self, "mcts_efficiency") and self.mcts_efficiency:
            avg_mcts_efficiency = sum(self.mcts_efficiency) / len(self.mcts_efficiency)
            best_mcts_efficiency = max(self.mcts_efficiency)
            self.logger.info(
                f"[SUMMARY] MCTS Efficiency: Avg={avg_mcts_efficiency:.2f} | Best={best_mcts_efficiency:.2f}"
            )

        self.logger.info("[SUMMARY] Training complete")
        self.logger.info("=" * 80)


Logger = ConsoleMetricsLogger
