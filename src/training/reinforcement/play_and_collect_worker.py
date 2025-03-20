import time
import torch
import chess.pgn
import numpy as np
from src.training.reinforcement.mcts import MCTS
from src.models.cnn import CNNModel
from src.utils.train_utils import initialize_random_seeds
from src.utils.chess_utils import convert_board_to_input,get_move_mapping,get_total_moves

class PlayAndCollectWorker:
    @classmethod
    def run_process(cls,model_state_dict,device_type,simulations,c_puct,temperature,games_per_process,seed):
        initialize_random_seeds(seed)
        d=torch.device(device_type)
        model=CNNModel(get_total_moves())
        try:
            model.load_state_dict(model_state_dict,strict=False)
        except:
            return([],[],[],{},[])
        model.to(d).eval()
        inps,pols,vals=[],[],[]
        res=[]
        pgns=[]
        s={"wins":0,"losses":0,"draws":0,"game_lengths":[],"results":[]}
        mm=get_move_mapping()
        g=chess.pgn.Game()
        for _ in range(games_per_process):
            b=chess.Board()
            mcts=MCTS(model,d,c_puct,simulations)
            mcts.set_root_node(b)
            st=[]
            mp=[]
            cp=[]
            mc=0
            g=chess.pgn.Game()
            g.headers.update({"Event":"Reinforcement Self-Play","Site":"Self-Play","Date":time.strftime("%Y.%m.%d"),"Round":"-","White":"Agent","Black":"Opponent","Result":"*"})
            node=g
            while not b.is_game_over()and mc<200:
                ap=mcts.get_move_probs(temperature)
                if mc==0 and ap:
                    ml=list(ap.keys())
                    no=np.random.dirichlet([0.3]*len(ml))
                    for i,mv in enumerate(ml):
                        ap[mv]=0.75*ap[mv]+0.25*no[i]
                if not ap:break
                ma=list(ap.keys())
                probs=np.array(list(ap.values()),dtype=np.float32)
                probs/=probs.sum()
                move=np.random.choice(ma,p=probs)
                st.append(convert_board_to_input(b))
                pa=np.zeros(get_total_moves(),dtype=np.float32)
                for mv,pr in ap.items():
                    idx=mm.get_index_by_move(mv)
                    if idx is not None and 0<=idx<get_total_moves():
                        pa[idx]=pr
                mp.append(pa)
                cp.append(b.turn)
                try:
                    b.push(move)
                except ValueError:
                    break
                node=node.add_variation(move)
                mcts.update_with_move(move)
                mc+=1
            dmap={'1-0':1.0,'0-1':-1.0,'1/2-1/2':0.0}
            out=dmap.get(b.result(),0.0)
            wr=[]
            for pl in cp:
                sg=out if pl==chess.WHITE else -out
                wr.append(sg)
            g.headers["Result"]='1-0'if out>0 else'0-1'if out<0 else'1/2-1/2'
            ex=chess.pgn.StringExporter(headers=True,variations=True,comments=True)
            gs=g.accept(ex)
            pgns.append(gs)
            inps.extend(st)
            pols.extend(mp)
            vals.extend(wr)
            res.append(out)
            s['game_lengths'].append(mc)
            s['results'].append(out)
            if out==1.0:
                s['wins']+=1
            elif out==-1.0:
                s['losses']+=1
            else:
                s['draws']+=1
        return(inps,pols,vals,s,pgns)