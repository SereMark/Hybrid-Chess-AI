import os, json, time, numpy as np, chess, chess.pgn
from src.analysis.benchmark.bot import Bot

class BenchmarkWorker:
    def __init__(self, bot1_path, bot2_path, num_games, bot1_use_mcts, bot1_use_opening_book, bot2_use_mcts, bot2_use_opening_book, wandb_flag=False, progress_callback=None, status_callback=None):
        self.num_games = num_games
        self.wandb_flag = wandb_flag
        self.progress_callback = progress_callback or (lambda x: None)
        self.status_callback = status_callback or (lambda x: None)
        self.bot1 = Bot(bot1_path, bot1_use_mcts, bot1_use_opening_book)
        self.bot2 = Bot(bot2_path, bot2_use_mcts, bot2_use_opening_book)
        self.games_dir = os.path.join("data","games","benchmark")
        os.makedirs(self.games_dir, exist_ok=True)
        ob_path = os.path.join("data","processed","opening_book.json")
        if os.path.isfile(ob_path):
            try:
                with open(ob_path,"r",encoding="utf-8") as f:
                    self.opening_book = json.load(f)
            except Exception as e:
                self.status_callback(f"Failed to load opening book: {e}")
                self.opening_book = {}
        else:
            self.opening_book = {}
            self.status_callback("Opening book file not found.")
    def run(self):
        if self.wandb_flag:
            import wandb
            wandb.init(entity="chess_ai",project="chess_ai_app",name="benchmark",config=self.__dict__,reinit=True)
        results = {"1-0":0,"0-1":0,"1/2-1/2":0,"*":0}
        durations, move_counts = [], []
        b1, b2, dr = 0, 0, 0
        b1_wot, b2_wot, draws_ot = [], [], []
        for g in range(1,self.num_games+1):
            self.status_callback(f"Playing game {g}/{self.num_games}")
            start_t = time.time()
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers.update({"Event":"Bot Benchmarking","Site":"Local","Date":time.strftime("%Y.%m.%d"),"Round":str(g),"White":"Bot1","Black":"Bot2","Result":"*"})
            node = game
            moves = 0
            while not board.is_game_over():
                current = self.bot1 if board.turn == chess.WHITE else self.bot2
                mv = current.get_move(board, self.opening_book)
                if (not mv) or (mv == chess.Move.null()) or (mv not in board.legal_moves):
                    self.status_callback("Bot returned invalid/null move.")
                    break
                board.push(mv)
                node = node.add_variation(mv)
                moves+=1
            r = board.result()
            if r not in results: results[r] = 0
            results[r]+=1
            game.headers["Result"] = r
            path = os.path.join(self.games_dir,f"game_{g}.pgn")
            with open(path,"w",encoding="utf-8") as pf:
                pf.write(str(game))
            end_t = time.time()
            dur = end_t - start_t
            durations.append(dur)
            move_counts.append(moves)
            if r=="1-0": b1+=1
            elif r=="0-1": b2+=1
            elif r=="1/2-1/2": dr+=1
            b1_wot.append(b1)
            b2_wot.append(b2)
            draws_ot.append(dr)
            self.progress_callback(100*g/self.num_games)
            self.status_callback(f"Game {g} finished result={r} in {dur:.2f}s")
            if self.wandb_flag:
                import wandb
                wandb.log({"game_index":g,"game_result":r,"game_duration_sec":dur,"moves_made":moves,"bot1_wins_so_far":b1,"bot2_wins_so_far":b2,"draws_so_far":dr})
        avg_dur = float(np.mean(durations)) if durations else 0.0
        avg_moves = float(np.mean(move_counts)) if move_counts else 0.0
        self.status_callback(f"All {self.num_games} games done.")
        self.status_callback(f"Results: Bot1={results['1-0']}, Bot2={results['0-1']}, draws={results['1/2-1/2']}, unfinished={results['*']}")
        if self.wandb_flag:
            import wandb
            wandb.log({"total_games":self.num_games,"wins_bot1":results["1-0"],"wins_bot2":results["0-1"],"draws":results["1/2-1/2"],"unfinished":results["*"],"avg_game_duration":avg_dur,"avg_moves_per_game":avg_moves})
            tb = wandb.Table(columns=["Result","Count"], data=[["1-0",results["1-0"]],["0-1",results["0-1"]],["1/2-1/2",results["1/2-1/2"]],["*",results["*"]]])
            wandb.log({"results_bar": wandb.plot.bar(tb,"Result","Count","Game Outcomes")})
            if durations:
                wandb.log({"game_length_distribution": wandb.plot.histogram(wandb.Table(data=[[x] for x in durations], columns=["DurationSec"]),"DurationSec","Game Duration Distribution")})
            if move_counts:
                wandb.log({"move_count_distribution": wandb.plot.histogram(wandb.Table(data=[[m] for m in move_counts], columns=["Moves"]),"Moves","Game Move Count Distribution")})
            data_line = list(zip(range(1,self.num_games+1),b1_wot,b2_wot,draws_ot))
            line_tb = wandb.Table(data=data_line,columns=["Game","Bot1WinsSoFar","Bot2WinsSoFar","DrawsSoFar"])
            wandb.log({"wins_over_time": wandb.plot.line_series(xs=line_tb.get_column("Game"),ys=[line_tb.get_column("Bot1WinsSoFar"),line_tb.get_column("Bot2WinsSoFar"),line_tb.get_column("DrawsSoFar")],keys=["Bot1 Wins","Bot2 Wins","Draws"],title="Wins Over Time")})
            wandb.run.summary.update({"Wins(Bot1)":results["1-0"],"Wins(Bot2)":results["0-1"],"Draws":results["1/2-1/2"],"Unfinished":results["*"],"Avg Duration (sec)":avg_dur,"Avg Moves Per Game":avg_moves})
            try:
                wandb.finish()
            except Exception as e:
                self.status_callback(f"Error finishing wandb: {e}")
        return results