from __future__ import annotations

import csv
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set_theme(context='paper', style='whitegrid', palette='colorblind')
except ImportError:
    pass

RUN_NAME_MAPPING = {
    "20251125-124632": "Referencia",
    "20251126-100221": "Mély Keresés",
    "20251126-205609": "Nagy Áteresztőképesség",
    "20251127-153749": "Nagy Entrópia",
    "20251128-101517": "Hatékonyság",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("artifacts")

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / 'runs'
REPORTS_DIR = REPO_ROOT / 'benchmark_reports'
FIGURES_OUT = REPO_ROOT / 'thesis' / 'figures'
FIGURES_OUT.mkdir(parents=True, exist_ok=True)

import matplotlib.font_manager as fm

system_fonts = [f.name for f in fm.fontManager.ttflist]
has_times = 'Times New Roman' in system_fonts

if has_times:
    logger.info("Font 'Times New Roman' found.")
    serif_fonts = ['Times New Roman']
else:
    logger.error("CRITICAL ERROR: 'Times New Roman' font not found.")
    sys.exit("Execution stopped: Font requirement not met.")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': serif_fonts,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})

L10N = {
    "iteration": "Iteráció",
    "policy_loss": "Stratégia Veszteség",
    "value_loss": "Értékbecslési Veszteség",
    "entropy": "Entrópia",
    "win_rate": "Győzelmi Arány",
    "draw_rate": "Döntetlen Arány",
    "game_length": "Játékhossz",
    "count": "Relatív Gyakoriság",
    "elo": "Elo Pontszám",
    "nps": "Keresési Sebesség (NPS)",
    "batch_size": "Kötegméret",
    "speedup": "Maximális Gyorsulás",
    "baseline": "Python Bázis",
    "cpp_movegen": "C++ Lépésgenerálás",
    "batching": "GPU Kötegelés",
    "training_dynamics": "Tanulási Dinamika",
    "system_efficiency": "Rendszer Hatékonyság",
    "tournament_standings": "Bajnokság Eredmények",
    "sims": "Szimuláció",
    "games": "Játszma",
    "density": "Sűrűség",
    "time_elapsed": "Eltelt Idő (óra)",
    "workers": "Szálak Száma",
    "games_per_sec": "Játszma / mp",
    "ideal_scaling": "Elméleti Maximum",
    "Random": "Véletlen",
    "Greedy": "Mohó",
}

def t(key: str) -> str:
    return L10N.get(key, key)

@dataclass
class RunData:
    name: str
    config: Dict[str, Any]
    metrics: List[Dict[str, float]]
    win_rates: Dict[int, Dict[str, float]] = field(default_factory=dict)
    game_lengths: List[int] = field(default_factory=list)
    label: str = ""

def save_figure(fig: plt.Figure, name: str) -> None:
    path = FIGURES_OUT / f"{name}.pdf"
    fig.savefig(path)
    logger.info(f"Saved figure: {path}")

def ema(data: List[float], alpha: float = 0.1) -> List[float]:
    if not data:
        return []
    out = [data[0]]
    for x in data[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out

def flatten_config(cfg: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def generate_run_labels(runs: List[RunData]) -> None:
    if not runs:
        return

    configs = [flatten_config(r.config) for r in runs]
    all_keys = set().union(*[c.keys() for c in configs])

    diff_keys = []
    for key in all_keys:
        vals = set(str(c.get(key)) for c in configs)
        if len(vals) > 1:
            if "dir" in key or "path" in key or "run" in key:
                continue
            diff_keys.append(key)

    logger.info(f"Distinguishing config keys: {diff_keys}")

    known_names = set(RUN_NAME_MAPPING.values())
    for r, cfg in zip(runs, configs):
        if r.name in known_names:
            r.label = r.name
            continue

        parts = []
        if 'MCTS.train_simulations' in diff_keys:
            parts.append(f"{cfg.get('MCTS.train_simulations')} sim")
        if 'TRAIN.games_per_iter' in diff_keys:
            parts.append(f"{cfg.get('TRAIN.games_per_iter')} game")

        if not parts and diff_keys:
            k = diff_keys[0]
            parts.append(f"{k.split('.')[-1]}={cfg.get(k)}")

        r.label = ", ".join(parts) if parts else r.name[:8]

def parse_pgn_stats(run_path: Path) -> Tuple[Dict[int, Dict[str, float]], List[int]]:
    arena_dir = run_path / "arena_games"
    if not arena_dir.exists():
        return {}, []

    iter_pattern = re.compile(r"iter(\d+)_")

    wins_per_iter = defaultdict(float)
    games_per_iter = defaultdict(int)
    game_lengths = []

    for pgn_file in arena_dir.glob("*.pgn"):
        match = iter_pattern.search(pgn_file.name)
        if not match:
            continue

        iteration = int(match.group(1))
        try:
            content = pgn_file.read_text(encoding='utf-8', errors='ignore')

            if 'Result "1-0"' in content:
                wins_per_iter[iteration] += 1.0
            elif 'Result "0-1"' in content:
                pass
            elif 'Result "1/2-1/2"' in content:
                wins_per_iter[iteration] += 0.5

            games_per_iter[iteration] += 1

            moves = content.count('.')
            if moves > 0:
                game_lengths.append(moves * 2)

        except Exception:
            continue

    win_stats = {}
    for it in games_per_iter:
        total = games_per_iter[it]
        wins = wins_per_iter[it]
        rate = (wins / total) * 100.0
        win_stats[it] = {'rate': rate, 'total': total, 'wins': wins}

    return win_stats, game_lengths

def load_run_data() -> List[RunData]:
    data = []
    if not RUNS_DIR.exists():
        return data

    for p in sorted(RUNS_DIR.iterdir()):
        if not p.is_dir():
            continue

        config = {}
        cfg_path = p / 'config' / 'merged.json'
        if cfg_path.exists():
            try:
                config = json.loads(cfg_path.read_text())
            except:
                pass

        metrics = []
        metrics_file = p / 'metrics' / 'training.csv'
        if metrics_file.exists():
            try:
                with metrics_file.open(newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        m = {}
                        for k, v in row.items():
                            try:
                                m[k] = float(v)
                            except:
                                m[k] = v
                        metrics.append(m)
            except Exception as e:
                logger.warning(f"Failed to read metrics for {p.name}: {e}")

        if not metrics:
            continue

        win_stats, lengths = parse_pgn_stats(p)

        name = RUN_NAME_MAPPING.get(p.name, p.name)
        data.append(RunData(
            name=name,
            config=config,
            metrics=metrics,
            win_rates=win_stats,
            game_lengths=lengths
        ))

    return data

def plot_training_metrics(runs: List[RunData]) -> None:
    if not runs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for r in runs:
        iters = [m.get('iter', 0) for m in r.metrics]
        vals = ema([m.get('policy_loss', float('nan')) for m in r.metrics])
        ax.plot(iters, vals, label=r.label)
    ax.set_title(t("policy_loss"))
    ax.set_xlabel(t("iteration"))
    ax.set_ylabel("Keresztentrópia")

    ax = axes[0, 1]
    for r in runs:
        iters = [m.get('iter', 0) for m in r.metrics]
        vals = ema([m.get('value_loss', float('nan')) for m in r.metrics])
        ax.plot(iters, vals, label=r.label)
    ax.set_title(t("value_loss"))
    ax.set_xlabel(t("iteration"))
    ax.set_ylabel("MSE")

    ax = axes[1, 0]
    for r in runs:
        times = [m.get('elapsed_s', 0) / 3600.0 for m in r.metrics]
        vals = ema([m.get('policy_loss', float('nan')) for m in r.metrics])
        ax.plot(times, vals, label=r.label)
    ax.set_title(f"{t('policy_loss')} az idő függvényében")
    ax.set_xlabel(t("time_elapsed"))
    ax.set_ylabel("Keresztentrópia")

    ax = axes[1, 1]
    for r in runs:
        times = [m.get('elapsed_s', 0) / 3600.0 for m in r.metrics]
        vals = ema([m.get('value_loss', float('nan')) for m in r.metrics])
        ax.plot(times, vals, label=r.label)
    ax.set_title(f"{t('value_loss')} az idő függvényében")
    ax.set_xlabel(t("time_elapsed"))
    ax.set_ylabel("MSE")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(runs), frameon=False)

    fig.tight_layout()
    save_figure(fig, "training_dynamics")
    plt.close(fig)

def plot_win_rates(runs: List[RunData]) -> None:
    if not runs:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in runs:
        if not r.win_rates:
            continue

        sorted_iters = sorted(r.win_rates.keys())
        if not sorted_iters:
            continue

        rates = []
        uppers = []
        lowers = []
        valid_iters = []

        for i in sorted_iters:
            stats = r.win_rates[i]
            n = stats['total']
            if n < 1:
                continue

            p = stats['rate'] / 100.0
            se = np.sqrt(p * (1 - p) / n)
            margin = 1.96 * se * 100.0

            rates.append(stats['rate'])
            lowers.append(max(0, stats['rate'] - margin))
            uppers.append(min(100, stats['rate'] + margin))
            valid_iters.append(i)

        rates_smooth = ema(rates, 0.3)

        p_line = ax.plot(valid_iters, rates_smooth, label=r.label, marker='o', markersize=3)
        color = p_line[0].get_color()

        ax.fill_between(valid_iters, lowers, uppers, color=color, alpha=0.15)

    ax.set_title(t("win_rate"))
    ax.set_xlabel(t("iteration"))
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    ax.legend(loc='lower right')
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, "evaluation_win_rate")
    plt.close(fig)

def plot_game_lengths(runs: List[RunData]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    has_data = False
    for r in runs:
        if len(r.game_lengths) > 10:
            has_data = True
            sns.kdeplot(data=r.game_lengths, label=r.label, fill=True, alpha=0.1, ax=ax, linewidth=2, clip=(0, None))

    if not has_data:
        plt.close(fig)
        return

    ax.set_title(t("game_length") + " Eloszlás")
    ax.set_xlabel("Fél-lépések száma")
    ax.set_ylabel(t("density"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, "game_length_dist")
    plt.close(fig)

def plot_system_scaling() -> None:
    bench_path = REPORTS_DIR / "system_bench.csv"
    if not bench_path.exists():
        return

    df = pd.read_csv(bench_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sp_df = df[df['type'] == 'selfplay'].copy()
    sp_df['workers'] = pd.to_numeric(sp_df['workers'], errors='coerce')
    sp_df = sp_df.sort_values('workers')

    if not sp_df.empty:
        sns.barplot(data=sp_df, x='workers', y='games_per_sec', ax=ax, hue='workers', palette='viridis', alpha=0.8, legend=False)

        ax.set_xticklabels([int(float(x.get_text())) for x in ax.get_xticklabels()])

        base_row = sp_df.iloc[0]
        base_rate = base_row['games_per_sec'] / base_row['workers']

        x_indices = np.arange(len(sp_df))
        workers_vals = sp_df['workers'].values
        ideal_y = [base_rate * w for w in workers_vals]

        ax.plot(x_indices, ideal_y, 'r--', marker='x', linewidth=2, label=t('ideal_scaling'))
        ax.legend()

        ax.set_title("Önjáték Teljesítmény Skálázódása")
        ax.set_xlabel(t("workers"))
        ax.set_ylabel(t("games_per_sec"))

    ax = axes[1]

    mcts_path = REPORTS_DIR / "mcts_scaling.csv"
    if mcts_path.exists():
        mcts_df = pd.read_csv(mcts_path)
        sns.lineplot(data=mcts_df, x='batch_size', y='nps', marker='o', ax=ax, linewidth=2, color='#2ecc71')
        ax.set_xscale('log', base=2)
        ax.set_title("Keresési Sebesség (NPS)")
        ax.set_xlabel(t("batch_size"))
        ax.set_ylabel("NPS")

        batches = mcts_df['batch_size'].unique()
        ax.set_xticks(batches)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, which="both")

    elif not df[df['type'] == 'training'].empty:
        tr_df = df[df['type'] == 'training'].sort_values('batch_size')
        sns.lineplot(data=tr_df, x='batch_size', y='steps_per_sec', marker='o', ax=ax)
        ax.set_title("Tanítási Lépés/mp")

    fig.tight_layout()
    save_figure(fig, "nps_scaling")
    plt.close(fig)

def plot_tournament_elo() -> None:
    results_path = REPORTS_DIR / "ranking_results.json"
    if not results_path.exists():
        return

    try:
        with results_path.open(encoding='utf-8') as f:
            data = json.load(f)

        elos = data.get("elo", {})
        if not elos:
            return

        sorted_elos = sorted(elos.items(), key=lambda x: x[1])
        names = [t(x[0]) for x in sorted_elos]
        ratings = [x[1] for x in sorted_elos]

        orig_names = [x[0] for x in sorted_elos]
        colors = ['#e74c3c' if n in ["Random", "Greedy"] else '#3498db' for n in orig_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(names))

        bars = ax.barh(y_pos, ratings, color=colors, alpha=0.8, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel(t("elo"))
        ax.set_title(t("tournament_standings"))

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 10, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', ha='left', va='center', fontweight='bold')

        ax.grid(True, axis='x', alpha=0.3)

        fig.tight_layout()
        save_figure(fig, "tournament_elo")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Tournament plot error: {e}")

def main() -> None:
    logger.info("Starting Artifact Generation...")

    runs = load_run_data()
    generate_run_labels(runs)

    logger.info(f"Loaded {len(runs)} runs.")

    plot_training_metrics(runs)
    plot_win_rates(runs)
    plot_game_lengths(runs)
    plot_system_scaling()
    plot_tournament_elo()

    logger.info(f"Artifact generation complete. Figures saved to: {FIGURES_OUT}")

if __name__ == '__main__':
    main()
