from src.base.base_visualization import BasePlot, BaseVisualizationWidget

class BenchmarkVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reset_visualization()

    def init_visualization(self):
        self.ax_benchmark = self.figure.add_subplot(111)
        self.plots['benchmark_results'] = BasePlot(
            self.ax_benchmark,
            title='Benchmark Results',
            xlabel='Category',
            ylabel='Number of Games'
        )
        self.add_text_to_axis('benchmark_results', 'No Benchmark Data Yet')

    def update_benchmark_visualization(self, bot1_wins, bot2_wins, draws, total_games):
        self.bot1_wins = bot1_wins
        self.bot2_wins = bot2_wins
        self.draws = draws
        self.total_games = total_games
        self._update_bar_chart()
        self.update_visualization()

    def _update_bar_chart(self):
        self.clear_axis('benchmark_results')
        if self.total_games <= 0:
            self.add_text_to_axis('benchmark_results', 'No Benchmark Data Yet')
            return
        labels = ['Bot1 Wins', 'Bot2 Wins', 'Draws']
        values = [self.bot1_wins, self.bot2_wins, self.draws]
        colors = ['#2ca02c', '#d62728', '#1f77b4']
        self.ax_benchmark.bar(labels, values, color=colors, alpha=0.8)
        self.ax_benchmark.set_ylim(0, max(values) + 1)
        for i, v in enumerate(values):
            self.ax_benchmark.text(i, v + 0.1, f"{v}", ha='center', fontsize=9, fontweight='bold', color='#333333')
        self.ax_benchmark.set_title(f"Benchmark Results (Total Games: {self.total_games})")

    def reset_visualization(self):
        self.bot1_wins = 0
        self.bot2_wins = 0
        self.draws = 0
        self.total_games = 0
        super().reset_visualization()