import os, h5py, time, chess, chess.pgn, numpy as np, chess.engine, asyncio, platform, json
from collections import defaultdict
from src.utils.chess_utils import convert_board_to_transformer_input, get_move_mapping, flip_board, flip_move, mirror_rank, mirror_move_rank

class DataPreparationWorker:
    def __init__(self, raw_pgn, max_games, min_elo, max_elo, batch_size, engine_path, engine_depth, engine_threads, engine_hash, pgn_file, max_opening_moves, wandb_flag, progress_callback=None, status_callback=None, skip_min_moves=0, skip_max_moves=99999, use_time_analysis=False, analysis_time=0.5):
        self.raw_pgn_file = raw_pgn
        self.max_games = max_games
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.batch_size = batch_size
        self.engine_path = engine_path
        self.engine_depth = engine_depth
        self.engine_threads = engine_threads
        self.engine_hash = engine_hash
        self.pgn_file = pgn_file
        self.max_opening_moves = max_opening_moves
        self.wandb_flag = wandb_flag
        self.progress_callback = progress_callback or (lambda x: None)
        self.status_callback = status_callback or (lambda x: None)
        self.skip_min_moves = skip_min_moves
        self.skip_max_moves = skip_max_moves
        self.use_time_analysis = use_time_analysis
        self.analysis_time = analysis_time
        self.positions = defaultdict(lambda: defaultdict(lambda: {"win":0,"draw":0,"loss":0,"eco":"","name":""}))
        self.game_counter = 0
        self.start_time = None
        self.batch_inputs = []
        self.batch_policy_targets = []
        self.batch_value_targets = []
        self.move_mapping = get_move_mapping()
        self.output_dir = os.path.abspath(os.path.join("data", "processed"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.total_games_processed = 0
        self.current_dataset_size = 0
        self.elo_list = []
        self.game_lengths = []
        self.time_control_stats = defaultdict(int)
        self.augment_flip = True
        self.augment_mirror_rank = True
    def run(self):
        import wandb
        if self.wandb_flag:
            wandb.init(entity="chess_ai", project="chess_ai_app", name="data_preparation", config=self.__dict__, reinit=True)
            batch_table = wandb.Table(columns=["Batch","Batch Size","Mean Value","Std Value"])
            game_table = wandb.Table(columns=["Games Processed","Games Skipped","Progress","Batch Size","Dataset Size"])
            opening_table = wandb.Table(columns=["Opening Games Processed","Opening Games Skipped","Opening Progress","Unique Positions"])
        else:
            batch_table = None
            game_table = None
            opening_table = None
        self.start_time = time.time()
        if platform.system()=="Windows":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
                engine.configure({"Threads":self.engine_threads,"Hash":self.engine_hash})
                self.status_callback("✅ Chess engine initialized successfully.")
                h5_path = os.path.join(self.output_dir,"dataset.h5")
                with h5py.File(h5_path,"w") as h5_file:
                    h5_file.create_dataset("inputs",(0,64,18),maxshape=(None,64,18),dtype=np.float32,compression="lzf")
                    h5_file.create_dataset("policy_targets",(0,),maxshape=(None,),dtype=np.int64,compression="lzf")
                    h5_file.create_dataset("value_targets",(0,),maxshape=(None,),dtype=np.float32,compression="lzf")
                    skipped_games = 0
                    last_update = time.time()
                    with open(self.raw_pgn_file,"r",errors="ignore") as f:
                        while self.total_games_processed<self.max_games:
                            game = chess.pgn.read_game(f)
                            if not game:
                                break
                            if "Variant" in game.headers:
                                skipped_games+=1
                                continue
                            if game.headers.get("WhiteTitle")=="BOT" or game.headers.get("BlackTitle")=="BOT":
                                skipped_games+=1
                                continue
                            we = game.headers.get("WhiteElo")
                            be = game.headers.get("BlackElo")
                            if not we or not be:
                                skipped_games+=1
                                continue
                            try:
                                we2 = int(we)
                                be2 = int(be)
                                if we2<self.min_elo or be2<self.min_elo or we2>self.max_elo or be2>self.max_elo:
                                    skipped_games+=1
                                    continue
                            except:
                                skipped_games+=1
                                continue
                            tc = game.headers.get("TimeControl","")
                            if tc:
                                self.time_control_stats[tc]+=1
                            self.elo_list.extend([we2,be2])
                            rm={"1-0":1.0,"0-1":-1.0,"1/2-1/2":0.0}
                            gr = rm.get(game.headers.get("Result"))
                            if gr is None:
                                skipped_games+=1
                                continue
                            b = game.board()
                            n = game
                            mc=0
                            while n.variations:
                                nn = n.variation(0)
                                mv = nn.move
                                mc+=1
                                if mc<self.skip_min_moves or mc>self.skip_max_moves:
                                    b.push(mv)
                                    n=nn
                                    continue
                                inp = convert_board_to_transformer_input(b)
                                mid = self.move_mapping.get_index_by_move(mv)
                                if mid is None:
                                    b.push(mv)
                                    n=nn
                                    continue
                                if self.use_time_analysis:
                                    limit = chess.engine.Limit(time=self.analysis_time)
                                else:
                                    limit = chess.engine.Limit(depth=self.engine_depth)
                                try:
                                    info = engine.analyse(b,limit)
                                    s = info["score"].pov(b.turn)
                                    if s.is_mate():
                                        v = 1.0 if s.mate()>0 else -1.0
                                    else:
                                        c = s.score()
                                        v = float(np.clip(c/1000.0,-1.0,1.0)) if c is not None else 0.0
                                except:
                                    self.status_callback("❌ Engine error")
                                    v=0.0
                                fv = v if b.turn else -v
                                self.batch_inputs.append(inp)
                                self.batch_policy_targets.append(mid)
                                self.batch_value_targets.append(fv)
                                if self.augment_flip:
                                    fb=flip_board(b)
                                    fm=flip_move(mv)
                                    fm_id = self.move_mapping.get_index_by_move(fm)
                                    if fm_id is not None:
                                        fi = convert_board_to_transformer_input(fb)
                                        self.batch_inputs.append(fi)
                                        self.batch_policy_targets.append(fm_id)
                                        self.batch_value_targets.append(-fv)
                                if self.augment_mirror_rank:
                                    rb=mirror_rank(b)
                                    rmv=mirror_move_rank(mv)
                                    rm_id = self.move_mapping.get_index_by_move(rmv)
                                    if rm_id is not None:
                                        ri = convert_board_to_transformer_input(rb)
                                        self.batch_inputs.append(ri)
                                        self.batch_policy_targets.append(rm_id)
                                        self.batch_value_targets.append(-fv)
                                b.push(mv)
                                n=nn
                            if mc<self.skip_min_moves or mc==0:
                                skipped_games+=1
                                continue
                            self.total_games_processed+=1
                            if len(self.batch_inputs)>=self.batch_size:
                                self._write_batch_to_h5(h5_file,batch_table,wandb)
                            if (self.total_games_processed%10==0) or (time.time()-last_update>5):
                                pr = min(int((self.total_games_processed/self.max_games)*100),100)
                                self.progress_callback(pr)
                                self.status_callback("✅ Processed {}/{} games. Skipped {} games.".format(self.total_games_processed,self.max_games,skipped_games))
                                if self.wandb_flag:
                                    self._log_game_stats_to_wandb(wandb,game_table,skipped_games,pr)
                                last_update=time.time()
                    if self.batch_inputs:
                        self._write_batch_to_h5(h5_file,batch_table,wandb)
                    self._generate_opening_book(wandb,opening_table)
                    self._create_train_val_test_split(h5_path,wandb)
                self.status_callback("✅ Data Preparation completed successfully. Processed {} games with {} skipped. Time: {:.2f} seconds.".format(self.total_games_processed,skipped_games,time.time()-self.start_time))
        except chess.engine.EngineError as e:
            self.status_callback("❌ Failed to initialize engine: {}".format(e))
        except Exception as e:
            self.status_callback("❌ An unexpected error occurred: {}".format(e))
        if self.wandb_flag:
            self._final_wandb_logs()
            try:
                wandb.finish()
            except Exception as e:
                self.status_callback("⚠️ Error finishing wandb run: {}".format(e))
        return True
    def _write_batch_to_h5(self,h5_file,batch_table,wandb):
        bs = len(self.batch_inputs)
        e = self.current_dataset_size+bs
        try:
            i_np = np.array(self.batch_inputs,dtype=np.float32)
            p_np = np.array(self.batch_policy_targets,dtype=np.int64)
            v_np = np.array(self.batch_value_targets,dtype=np.float32)
            h5_file["inputs"].resize((e,64,i_np.shape[2]))
            h5_file["policy_targets"].resize((e,))
            h5_file["value_targets"].resize((e,))
            h5_file["inputs"][self.current_dataset_size:e,:,:] = i_np
            h5_file["policy_targets"][self.current_dataset_size:e] = p_np
            h5_file["value_targets"][self.current_dataset_size:e] = v_np
            m = float(np.mean(v_np))
            s = float(np.std(v_np))
            if wandb:
                try:
                    batch_table.add_data(str(self.total_games_processed),bs,m,s)
                    wandb.log({"batch_size":bs,"mean_value_targets":m,"std_value_targets":s})
                except:
                    pass
            self.current_dataset_size+=bs
        except:
            self.status_callback("❌ Error writing to HDF5")
        finally:
            self.batch_inputs.clear()
            self.batch_policy_targets.clear()
            self.batch_value_targets.clear()
    def _log_game_stats_to_wandb(self,wandb,gt,sk,pr):
        try:
            gt.add_data(str(self.total_games_processed),sk,pr,len(self.batch_inputs),self.current_dataset_size)
            wandb.log({"games_processed":self.total_games_processed,"games_skipped":sk,"progress":pr,"current_batch_size":len(self.batch_inputs),"total_dataset_size":self.current_dataset_size})
        except:
            pass
    def _generate_opening_book(self,wandb,ot):
        if not self.pgn_file or not self.max_opening_moves:
            return
        self.status_callback("🔍 Processing Opening Book...")
        sb=0
        lu=time.time()
        with open(self.pgn_file,"r",encoding="utf-8",errors="ignore") as pf:
            while self.game_counter<self.max_games:
                g = chess.pgn.read_game(pf)
                if not g:
                    self.status_callback("🔍 Reached end of PGN file for opening book.")
                    break
                h=g.headers
                try:
                    we=int(h.get("WhiteElo",0))
                    be=int(h.get("BlackElo",0))
                    if we<self.min_elo or be<self.min_elo or we>self.max_elo or be>self.max_elo:
                        sb+=1
                        continue
                    om={"1-0":"win","0-1":"loss","1/2-1/2":"draw"}
                    o=om.get(h.get("Result"))
                    if not o:
                        sb+=1
                        continue
                    eco=h.get("ECO","")
                    nm=h.get("Opening","")
                    b=g.board()
                    for c,m in enumerate(g.mainline_moves(),1):
                        if c>self.max_opening_moves:
                            break
                        fen=b.fen()
                        u=m.uci()
                        md=self.positions[fen][u]
                        md[o]+=1
                        if not md["eco"]:
                            md["eco"]=eco
                        if not md["name"]:
                            md["name"]=nm
                        b.push(m)
                    self.game_counter+=1
                    if (self.game_counter%10==0) or (time.time()-lu>5):
                        pr=min(int((self.game_counter/self.max_games)*100),100)
                        self.progress_callback(pr)
                        self.status_callback("✅ Processed {}/{} opening games. Skipped {} games.".format(self.game_counter,self.max_games,sb))
                        if wandb:
                            try:
                                ot.add_data(str(self.game_counter),sb,pr,len(self.positions))
                                wandb.log({"opening_games_processed":self.game_counter,"opening_games_skipped":sb,"opening_progress":pr,"unique_positions":len(self.positions)})
                            except:
                                pass
                        lu=time.time()
                except:
                    sb+=1
                    self.status_callback("❌ Invalid or error in opening game.")
        try:
            pd={k:dict(v) for k,v in self.positions.items()}
            bf=os.path.join(self.output_dir,"opening_book.json")
            with open(bf,"w") as f:
                json.dump(pd,f,indent=4)
            self.status_callback("✅ Opening book saved at {}.".format(bf))
        except:
            self.status_callback("❌ Failed to save opening book.")
    def _create_train_val_test_split(self,h5_path,wandb):
        self.status_callback("🔍 Splitting dataset into train, validation, and test sets...")
        try:
            with h5py.File(h5_path,"r") as hf:
                n=hf["inputs"].shape[0]
                if n==0:
                    self.status_callback("❌ No samples to split.")
                    return
                idx=np.random.permutation(n)
                tr=int(n*0.8)
                va=int(n*0.9)
                sp={"train":idx[:tr],"val":idx[tr:va],"test":idx[va:]}
                for k,v in sp.items():
                    np.save(os.path.join(self.output_dir,"{}_indices.npy".format(k)),v)
                if wandb:
                    wandb.log({"train_size":len(sp["train"]),"val_size":len(sp["val"]),"test_size":len(sp["test"])})
                self.status_callback("✅ Dataset split into Train ({}), Validation ({}), Test ({}) samples.".format(len(sp["train"]),len(sp["val"]),len(sp["test"])))
        except:
            self.status_callback("❌ Error splitting dataset.")
    def _final_wandb_logs(self):
        import wandb
        if self.elo_list:
            et=wandb.Table(data=[[e] for e in self.elo_list],columns=["ELO"])
            wandb.log({"ELO Distribution":wandb.plot.histogram(et,"ELO","ELO Distribution")})
        if self.game_lengths:
            lt=wandb.Table(data=[[l] for l in self.game_lengths],columns=["Length"])
            wandb.log({"Game Length Distribution":wandb.plot.histogram(lt,"Length","Game Length Distribution")})
        if self.time_control_stats:
            tc=wandb.Table(columns=["TimeControl","Count"])
            for k,v in self.time_control_stats.items():
                tc.add_data(k,v)
            wandb.log({"Time Control Breakdown":wandb.plot.bar(tc,"TimeControl","Count","Time Control Stats")})
        da=wandb.Artifact("chess_dataset",type="dataset")
        da.add_file(os.path.join(self.output_dir,"dataset.h5"))
        wandb.log_artifact(da)
        bf=os.path.join(self.output_dir,"opening_book.json")
        if os.path.exists(bf):
            ba=wandb.Artifact("opening_book",type="dataset")
            ba.add_file(bf)
            wandb.log_artifact(ba)