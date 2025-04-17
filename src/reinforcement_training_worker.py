import os
import time
import torch
import wandb
import chess.pgn
import numpy as np
from utils.mcts import MCTS
from src.cnn import CNNModel
from torch.amp import GradScaler
from multiprocessing import Pool,cpu_count
from torch.utils.data import DataLoader,TensorDataset
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.chess_utils import get_total_moves,convert_board_to_input,get_move_mapping
from src.utils.train_utils import initialize_optimizer,initialize_scheduler,initialize_random_seeds,train_epoch

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

class ReinforcementWorker:
    def __init__(self,model_path,num_iterations,num_games_per_iteration,simulations_per_move,c_puct,temperature,epochs_per_iteration,batch_size,num_selfplay_threads,checkpoint_interval,random_seed,optimizer_type,learning_rate,weight_decay,scheduler_type,accumulation_steps,num_workers,policy_weight,value_weight,grad_clip,momentum):
        initialize_random_seeds(random_seed)
        self.device=torch.device("cuda"if torch.cuda.is_available()else"cpu")
        self.model=CNNModel(get_total_moves()).to(self.device)
        self.optimizer=initialize_optimizer(self.model,optimizer_type,learning_rate,weight_decay,momentum)
        self.scheduler=None
        self.scaler=GradScaler(enabled=(self.device.type=="cuda"))
        self.accumulation_steps=accumulation_steps
        self.checkpoint_interval=checkpoint_interval
        self.checkpoint_manager=CheckpointManager(os.path.join("models","checkpoints","reinforcement"),"iteration",checkpoint_interval)
        self.model_path=model_path
        self.num_iterations=num_iterations
        self.num_games_per_iteration=num_games_per_iteration
        self.simulations_per_move=simulations_per_move
        self.c_puct=c_puct
        self.temperature=temperature
        self.epochs_per_iteration=epochs_per_iteration
        self.batch_size=batch_size
        self.num_selfplay_threads=num_selfplay_threads
        self.random_seed=random_seed
        self.num_workers=num_workers
        self.optimizer_type=optimizer_type
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.grad_clip=grad_clip
        self.momentum=momentum
        self.scheduler_type=scheduler_type
        self.policy_weight=policy_weight
        self.value_weight=value_weight
        self.best_metric=float('inf')
        self.best_iteration=0
        self.start_iteration=1
        self.loaded_checkpoint=None
        if self.model_path and os.path.exists(self.model_path):
            self.loaded_checkpoint=self.checkpoint_manager.load(self.model_path,self.device,self.model,self.optimizer,self.scheduler)
            if self.loaded_checkpoint and"iteration"in self.loaded_checkpoint:
                self.start_iteration=self.loaded_checkpoint["iteration"]+1
    def run(self):
        wandb.watch(self.model,log="parameters",log_freq=100)
        st=time.time()
        if self.start_iteration>self.num_iterations:
            raise ValueError("Start iteration exceeds total iterations.")
        for it in range(self.start_iteration,self.num_iterations+1):
            npcp=min(self.num_selfplay_threads,cpu_count())
            gpp,r=divmod(self.num_games_per_iteration,npcp)
            seeds=[self.random_seed+it*1000+i+int(time.time())for i in range(npcp)]
            ms={k:v.cpu()for k,v in self.model.state_dict().items()}
            tasks=[]
            for i in range(npcp):
                lg=gpp+(1 if i<r else 0)
                tasks.append((ms,self.device.type,self.simulations_per_move,self.c_puct,self.temperature,lg,seeds[i]))
            with Pool(processes=npcp)as pool:
                wr=pool.starmap(PlayAndCollectWorker.run_process,tasks)
            ai,ap,av=[],[],[]
            pgns=[]
            s={"wins":0,"losses":0,"draws":0,"game_lengths":[],"results":[]}
            for inp,pol,val,ws,pg in wr:
                ai.extend(inp)
                ap.extend(pol)
                av.extend(val)
                pgns.extend(pg)
                s['wins']+=ws.get('wins',0)
                s['losses']+=ws.get('losses',0)
                s['draws']+=ws.get('draws',0)
                s['game_lengths'].extend(ws.get('game_lengths',[]))
                s['results'].extend(ws.get('results',[]))
            s['total_games']=len(s['results'])
            s['avg_game_length']=float(np.mean(s['game_lengths']))if s['game_lengths']else 0.0
            wandb.log({
                "iteration":it,
                "total_games":s['total_games'],
                "wins":s['wins'],
                "losses":s['losses'],
                "draws":s['draws'],
                "avg_game_length":s['avg_game_length']
            })
            if s['results']:
                wandb.log({"results_histogram":wandb.Histogram(s['results'])})
            if s['game_lengths']:
                wandb.log({"game_length_histogram":wandb.Histogram(s['game_lengths'])})
            cm=float('inf')
            if ai:
                itensor=torch.from_numpy(np.array(ai,dtype=np.float32))
                ptensor=torch.from_numpy(np.array(ap,dtype=np.float32))
                vtensor=torch.tensor(av,dtype=torch.float32)
                dl=DataLoader(TensorDataset(itensor,ptensor,vtensor),batch_size=self.batch_size,shuffle=True,pin_memory=(self.device.type=="cuda"),num_workers=self.num_workers,persistent_workers=(self.num_workers>0))
                ts=(self.num_iterations-it+1)*len(dl)
                if self.scheduler is None:
                    self.scheduler=initialize_scheduler(self.optimizer,self.scheduler_type,ts)
                    if self.loaded_checkpoint and'scheduler_state_dict'in self.loaded_checkpoint:
                        self.scheduler.load_state_dict(self.loaded_checkpoint['scheduler_state_dict'])
                for ep in range(1,self.epochs_per_iteration+1):
                    tm=train_epoch(self.model,dl,self.device,self.scaler,self.optimizer,self.scheduler,self.accumulation_steps,False,self.policy_weight,self.value_weight,self.grad_clip,True)
                cm=self.policy_weight*tm["policy_loss"]+self.value_weight*tm["value_loss"]
                if cm<self.best_metric:
                    self.best_metric=cm
                    self.best_iteration=it
                    self.checkpoint_manager.save(self.model,self.optimizer,self.scheduler,it,None)
            if self.checkpoint_interval>0 and it%self.checkpoint_interval==0:
                self.checkpoint_manager.save(self.model,self.optimizer,self.scheduler,it)
            wandb.log({
                "iteration":it,
                "learning_rate":self.scheduler.get_last_lr()[0]
            })
            os.makedirs("data/games/self-play",exist_ok=True)
            for idx,g in enumerate(pgns,1):
                fn=os.path.join("data","games","self-play",f"game_{int(time.time())}_{idx}.pgn")
                with open(fn,"w",encoding="utf-8")as f:
                    f.write(str(g))
        self.checkpoint_manager.save(self.model,self.optimizer,self.scheduler,self.num_iterations,os.path.join("models","saved_models","reinforcement_model.pth"))
        wandb.run.summary.update({"best_metric":self.best_metric,"best_iteration":self.best_iteration,"training_time":time.time()-st})
        return True