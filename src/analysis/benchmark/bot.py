import os
import chess
import torch
import numpy as np
from src.training.reinforcement.mcts import MCTS
from src.utils.chess_utils import get_total_moves,convert_board_to_input,get_move_mapping
from src.models.cnn import CNNModel

class Bot:
    def __init__(self,path,use_mcts,use_opening_book):
        self.use_mcts=use_mcts
        self.use_opening_book=use_opening_book
        self.device=torch.device("cuda"if torch.cuda.is_available()else"cpu")
        self.model=CNNModel(get_total_moves()).to(self.device)
        self._load_model_checkpoint(path)
        self.mcts=MCTS(self.model,self.device,c_puct=1.4,n_simulations=100)if self.use_mcts else None
        self.move_map=get_move_mapping()
    def _load_model_checkpoint(self,path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            c=torch.load(path,map_location=self.device)
            if isinstance(c,dict)and"model_state_dict"in c:
                self.model.load_state_dict(c["model_state_dict"],strict=False)
            else:
                self.model.load_state_dict(c,strict=False)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    def get_move(self,b,opening_book):
        try:
            if self.use_opening_book:
                bm=self._choose_book_move(b,opening_book)
                if bm:return bm
            if self.use_mcts and self.mcts:
                return self._choose_mcts_move(b)
            return self._choose_direct_policy_move(b)
        except:
            return chess.Move.null()
    def _choose_book_move(self,b,ob):
        d=ob.get(b.fen())
        if not d:return None
        r=None
        s=-1
        for u,st in d.items():
            t=st.get("win",0)+st.get("draw",0)+st.get("loss",0)
            if t>0:
                sc=(st.get("win",0)+0.5*st.get("draw",0))/t
                mv=chess.Move.from_uci(u)
                if mv in b.legal_moves and sc>s:
                    r=mv
                    s=sc
        return r
    def _choose_mcts_move(self,b):
        self.mcts.set_root_node(b.copy())
        p=self.mcts.get_move_probs(temperature=1e-3)
        if b.fullmove_number==1 and b.turn==chess.WHITE and len(p)>1:
            ml=list(p.keys())
            arr=np.array(list(p.values()),dtype=np.float32)
            arr/=arr.sum()
            p=dict(zip(ml,arr))
        return max(p,key=p.get) if p else chess.Move.null()
    def _choose_direct_policy_move(self,b):
        x=torch.from_numpy(convert_board_to_input(b)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            l,v=self.model(x)
            pol=torch.softmax(l[0],dim=0).cpu().numpy()
        lm=list(b.legal_moves)
        if not lm:return chess.Move.null()
        mp={}
        for m in lm:
            i=self.move_map.get_index_by_move(m)
            mp[m]=pol[i] if i is not None else 1e-12
        s=sum(mp.values())
        if s>0:
            for m in mp:mp[m]/=s
        else:
            for m in mp:mp[m]=1/len(lm)
        return max(mp,key=mp.get)