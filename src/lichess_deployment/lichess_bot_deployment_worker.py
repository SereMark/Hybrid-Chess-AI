import os
import json
import time
import wandb
import torch
import chess
import random
import berserk
import logging
import threading
import chess.pgn
from src.training.reinforcement.mcts import MCTS
from src.utils.chess_utils import get_total_moves, convert_board_to_input
from typing import Any,Dict,Optional
from src.models.cnn import CNNModel

logger=logging.getLogger("LichessBot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h=logging.StreamHandler()
    f=logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    h.setFormatter(f)
    logger.addHandler(h)

def load_opening_book(p:str)->Dict[str,Any]:
    with open(p,'r')as f:
        return json.load(f)

def choose_opening_move(b:chess.Board,d:Dict[str,Any])->Optional[chess.Move]:
    fen=b.fen()
    m=d.get(fen)
    if m:
        sc={}
        for s,st in m.items():
            w=st.get("win",0);dr=st.get("draw",0);l=st.get("loss",0);t=w+dr+l
            sc[s]=((w+0.5*dr)+10*0.5)/(t+10)if t>0 else 0
        mx=max(sc.values())
        c=[mv for mv,v in sc.items() if v==mx]
        return chess.Move.from_uci(random.choice(c))
    return None

def update_board_from_moves(ms:str)->chess.Board:
    b=chess.Board()
    if ms:
        for m in ms.split():
            b.push_uci(m)
    return b

def determine_bot_color(g:Dict[str,Any],bot_id:str)->Optional[bool]:
    c=g.get("color")
    if c:
        if c.lower()=="white":return chess.WHITE
        elif c.lower()=="black":return chess.BLACK
    def ext_id(x:Dict[str,Any])->Optional[str]:
        return x.get("id")or x.get("user",{}).get("id")
    p=g.get("players")
    if p:
        w=ext_id(p.get("white",{}))
        bl=ext_id(p.get("black",{}))
        if w==bot_id:return chess.WHITE
        if bl==bot_id:return chess.BLACK
    else:
        w=ext_id(g.get("white",{}))
        bl=ext_id(g.get("black",{}))
        if w==bot_id:return chess.WHITE
        if bl==bot_id:return chess.BLACK
    logger.warning(f"Could not determine bot color. Full game info: {json.dumps(g,indent=2)}")
    return None

class LichessBotDeploymentWorker:
    def __init__(self,model_path,opening_book_path,lichess_token,time_control,rating_range,use_mcts,mcts_simulations,mcts_c_puct,auto_resign,save_game_logs,enable_model_eval_fallback,wandb_flag,progress_callback,status_callback):
        self.model_path=model_path
        self.opening_book_path=opening_book_path
        self.lichess_token=lichess_token
        self.time_control=time_control
        self.rating_range=rating_range
        self.use_mcts=use_mcts
        self.mcts_simulations=mcts_simulations
        self.mcts_c_puct=mcts_c_puct
        self.auto_resign=auto_resign
        self.save_game_logs=save_game_logs
        self.enable_model_eval_fallback=enable_model_eval_fallback
        self.wandb_flag=wandb_flag
        self.progress_callback=progress_callback
        self.status_callback=status_callback
        self.model:Optional[CNNModel]=None
        self.opening_book:Optional[Dict[str,Any]]=None
        self.mcts:Optional[MCTS]=None
        self.device=torch.device("cuda"if torch.cuda.is_available()else"cpu")
        self.bot_id:Optional[str]=None
        logger.setLevel(logging.INFO)
        self._initialize_engine()

    def _initialize_engine(self)->None:
        try:
            self.status_callback("Loading model...")
            logger.info(f"Loading model from {self.model_path}")
            c=torch.load(self.model_path,map_location=self.device)
            self.model=CNNModel(num_moves=get_total_moves()).to(self.device)
            if isinstance(c,dict)and"model_state_dict"in c:self.model.load_state_dict(c["model_state_dict"],strict=False)
            else:self.model.load_state_dict(c,strict=False)
            self.model.eval()
            self.progress_callback(25)
            self.status_callback("Loading opening book...")
            logger.info(f"Loading opening book from {self.opening_book_path}")
            self.opening_book=load_opening_book(self.opening_book_path)
            self.progress_callback(50)
            if self.use_mcts:
                self.status_callback("Initializing MCTS...")
                self.mcts=MCTS(model=self.model,device=self.device,c_puct=self.mcts_c_puct,n_simulations=self.mcts_simulations)
                logger.info(f"MCTS initialized with {self.mcts_simulations} simulations and c_puct={self.mcts_c_puct:.2f}")
            else:
                self.mcts=None
            self.progress_callback(75)
            self.status_callback("Chess engine compiled successfully.")
            self.progress_callback(100)
        except Exception as e:
            em=f"Engine initialization failed: {e}"
            self.status_callback(em)
            logger.exception(em)
            raise

    def start_bot(self)->None:
        self.status_callback("Starting Lichess bot...")
        logger.info("Starting Lichess bot...")
        try:
            s=berserk.TokenSession(self.lichess_token)
            c=berserk.Client(session=s)
            a=c.account.get()
            self.bot_id=a.get('id')
            if not self.bot_id:raise ValueError("Bot ID not found in account information.")
            url=f"https://lichess.org/@/{self.bot_id}"
            self.status_callback(f"Logged in as {self.bot_id}. Challenge the bot at: {url}")
            logger.info(f"Logged in as {self.bot_id}. Challenge your bot at: {url}")
            while True:
                try:
                    for e in c.bots.stream_incoming_events():
                        t=e.get('type')
                        if t=='challenge':
                            ch=e.get('challenge')
                            if ch:self._handle_challenge_event(c,ch)
                            else:logger.warning("Received challenge event without challenge details.")
                        elif t=='gameStart':
                            g=e.get('game')
                            if not g:
                                logger.warning("Received gameStart event without game info.")
                                continue
                            bc=determine_bot_color(g,self.bot_id)
                            if bc is None:
                                self.status_callback("Bot color could not be determined. Skipping game.")
                                logger.warning(f"Bot color could not be determined for game {g.get('id')}")
                                continue
                            th=threading.Thread(target=self._play_game,args=(c,g,bc),daemon=True)
                            th.start()
                        else:
                            logger.debug(f"Unhandled event type: {t}")
                except Exception as e:
                    logger.exception(f"Error in event stream: {e}")
                    self.status_callback(f"Streaming error: {e}. Reconnecting in 5 seconds...")
                    time.sleep(5)
        except Exception as e:
            em=f"Error starting bot or streaming events: {e}"
            self.status_callback(em)
            logger.exception(em)
            raise

    def _handle_challenge_event(self,client:berserk.Client,ch:Dict[str,Any])->None:
        try:
            cr=ch.get('challenger',{}).get('rating')
            if cr is None:
                logger.warning("Challenge missing rating information. Declining challenge.")
                client.bots.decline_challenge(ch['id'])
                return
            if self.rating_range[0]<=cr<=self.rating_range[1]:
                client.bots.accept_challenge(ch['id'])
                self.status_callback(f"Accepted challenge from rating {cr}. Playing game...")
                logger.info(f"Accepted challenge from rating {cr}")
                if self.wandb_flag:wandb.log({"challenge_rating":cr,"challenge_accepted":True,"phase":"challenge_response"})
            else:
                client.bots.decline_challenge(ch['id'])
                self.status_callback(f"Declined challenge from rating {cr}.")
                logger.info(f"Declined challenge from rating {cr}")
                if self.wandb_flag:wandb.log({"challenge_rating":cr,"challenge_accepted":False,"phase":"challenge_response"})
        except Exception as e:
            logger.exception(f"Error handling challenge event: {e}")
            self.status_callback(f"Error handling challenge event: {e}")

    def _play_game(self,client:berserk.Client,game_info:Dict[str,Any],bot_color:bool)->None:
        gid=game_info.get('id')
        if not gid:
            logger.error("Game information missing game ID.")
            return
        cs='White' if bot_color==chess.WHITE else'Black'
        logger.info(f"Starting game {gid} as {cs}")
        b=chess.Board()
        ms=""
        try:
            for e in client.bots.stream_game_state(gid):
                st=e.get('type')
                if st=='gameFull':
                    ms=e.get('state',{}).get('moves','')
                    b=update_board_from_moves(ms)
                    if b.turn==bot_color:
                        self._make_move(client,gid,b)
                elif st=='gameState':
                    ms=e.get('moves','')
                    b=update_board_from_moves(ms)
                    if b.is_game_over():
                        r=b.result()
                        logger.info(f"Game {gid} over with result {r}")
                        if self.wandb_flag:wandb.log({"game_id":gid,"result":r,"num_moves":len(ms.split()),"phase":"game_over"})
                        break
                    if b.turn==bot_color:
                        self._make_move(client,gid,b)
                else:
                    logger.debug(f"Unhandled game state event type: {st}")
                if b.is_game_over():
                    r=b.result()
                    logger.info(f"Game {gid} over with result {r}")
                    if self.wandb_flag:wandb.log({"game_id":gid,"result":r,"num_moves":len(ms.split()),"phase":"game_over"})
                    break
            if self.save_game_logs:
                self._save_game_pgn(gid,ms)
        except Exception as e:
            logger.exception(f"Error during game {gid}: {e}")

    def _make_move(self,client:berserk.Client,gid:str,b:chess.Board)->None:
        try:
            bm:Optional[chess.Move]=None
            if self.opening_book:
                bm=choose_opening_move(b,self.opening_book)
                if bm:logger.info(f"Using opening book move: {bm.uci()}")
            if bm is None and self.mcts:
                st=time.time()
                self.mcts.set_root_node(b)
                mp=self.mcts.get_move_probs()
                dt=time.time()-st
                if mp:
                    bm=max(mp,key=mp.get)
                    logger.info(f"MCTS selected move: {bm.uci()} (in {dt:.3f} sec)")
                    if self.wandb_flag:wandb.log({"mcts_time":dt,"selected_move":bm.uci(),"phase":"mcts_move_selection","game_id":gid})
                else:
                    logger.warning("MCTS returned no moves.")
            if bm is None and self.enable_model_eval_fallback and self.model:
                et=time.time()
                bm=self._evaluate_moves(b)
                dt=time.time()-et
                logger.info(f"Model evaluation selected move: {bm.uci()} (in {dt:.3f} sec)")
                if self.wandb_flag:wandb.log({"model_eval_time":dt,"selected_move":bm.uci(),"phase":"model_eval_fallback","game_id":gid})
            if bm is None:
                lm=list(b.legal_moves)
                if not lm:
                    if self.auto_resign:
                        try:
                            client.bots.resign(gid)
                            logger.info(f"Resigned game {gid} due to no legal moves.")
                        except Exception as e:
                            logger.exception(f"Failed to resign game {gid}: {e}")
                        return
                    else:
                        logger.warning(f"No legal moves available in game {gid}")
                        return
                bm=lm[0]
                logger.info(f"Defaulted to first legal move: {bm.uci()}")
            if bm not in b.legal_moves:
                logger.error(f"Selected move {bm.uci()} is not legal in game {gid}.")
                return
            b.push(bm)
            client.bots.make_move(gid,bm.uci())
            logger.info(f"Made move {bm.uci()} in game {gid}")
            if self.wandb_flag:wandb.log({"move_made":bm.uci(),"game_id":gid,"phase":"move_made"})
        except Exception as e:
            logger.exception(f"Failed to make move in game {gid}: {e}")
            raise

    def _evaluate_moves(self,b:chess.Board)->chess.Move:
        lm=list(b.legal_moves)
        c=b.turn
        bm=None
        bs=-float('inf') if c==chess.WHITE else float('inf')
        mv={}
        for m in lm:
            b.push(m)
            s=self._evaluate_board(b)
            mv[m.uci()]=s
            b.pop()
            if c==chess.WHITE and s>bs:
                bs=s;bm=m
            elif c==chess.BLACK and s<bs:
                bs=s;bm=m
        if self.wandb_flag:wandb.log({"move_evaluations":mv,"phase":"model_eval_moves"})
        return bm if bm else lm[0]

    def _evaluate_board(self,b:chess.Board)->float:
        try:
            t=convert_board_to_input(b)
            t=torch.from_numpy(t).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                sc=self.model(t)[1]
            return sc.item()
        except Exception as e:
            logger.exception(f"Failed to evaluate board: {e}")
            return 0.0

    def _save_game_pgn(self,gid:str,moves_str:str)->None:
        try:
            g=chess.pgn.Game()
            n=g
            b=chess.Board()
            for m in moves_str.split():
                mv=chess.Move.from_uci(m)
                n=n.add_variation(mv)
                b.push(mv)
            p=os.path.join("data","games","lichess",f"{gid}.pgn")
            os.makedirs(os.path.dirname(p),exist_ok=True)
            with open(p,"w")as f:
                f.write(str(g))
            logger.info(f"Saved game {gid} PGN log at {p}")
            if self.wandb_flag:wandb.log({"pgn_log_saved":True,"game_id":gid,"phase":"pgn_saved"})
        except Exception as e:
            logger.exception(f"Failed to save PGN log for game {gid}: {e}")

    def run(self)->None:
        try:
            self.start_bot()
        except Exception as e:
            em=f"Pipeline failed: {e}"
            self.status_callback(em)
            logger.exception(em)
            raise