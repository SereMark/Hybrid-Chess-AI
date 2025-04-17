import os,json,time,torch,wandb,chess,chess.pgn,numpy as np
from utils.mcts import MCTS
from src.cnn import CNNModel
from src.utils.chess_utils import get_total_moves,convert_board_to_input,get_move_mapping

class Bot:
    def __init__(self,p,mcts,book):
        self.use_mcts=mcts;self.use_opening_book=book;self.dev=torch.device('cuda'if torch.cuda.is_available()else'cpu')
        self.model=CNNModel(get_total_moves()).to(self.dev);self._load(p)
        self.mcts=MCTS(self.model,self.dev,c_puct=1.4,n_simulations=100)if mcts else None;self.mp=get_move_mapping()
    def _load(self,p):
        if not os.path.isfile(p):raise FileNotFoundError(p)
        ck=torch.load(p,map_location=self.dev)
        self.model.load_state_dict(ck['model_state_dict']if isinstance(ck,dict)and'model_state_dict'in ck else ck,strict=False);self.model.eval()
    def get_move(self,b,ob):
        try:
            if self.use_opening_book:
                d=ob.get(b.fen());best=None;sc=-1
                if d:
                    for u,s in d.items():
                        t=s.get('win',0)+s.get('draw',0)+s.get('loss',0)
                        if t>0:
                            v=(s.get('win',0)+.5*s.get('draw',0))/t;m=chess.Move.from_uci(u)
                            if m in b.legal_moves and v>sc:best,sc=m,v
                if best:return best
            if self.use_mcts and self.mcts:
                self.mcts.set_root_node(b.copy());p=self.mcts.get_move_probs(temperature=1e-3)
                if b.fullmove_number==1 and b.turn==chess.WHITE and len(p)>1:
                    k=list(p);v=np.array(list(p.values()),dtype=np.float32);v/=v.sum();p=dict(zip(k,v))
                return max(p,key=p.get) if p else chess.Move.null()
            x=torch.from_numpy(convert_board_to_input(b)).float().unsqueeze(0).to(self.dev)
            with torch.no_grad():l,_=self.model(x);pol=torch.softmax(l[0],0).cpu().numpy()
            mp={m:pol[self.mp.get_index_by_move(m)] if self.mp.get_index_by_move(m) is not None else 1e-12 for m in b.legal_moves}
            s=sum(mp.values())
            if s>0:
                for k in mp:mp[k]/=s
            else:
                n=len(mp)
                for k in mp:mp[k]=1/n
            return max(mp,key=mp.get)
        except:return chess.Move.null()

class BenchmarkWorker:
    def __init__(self,p1,p2,n,g1m,g1b,g2m,g2b,switch=False):
        self.n=n;self.sw=switch
        self.bot1=Bot(p1,g1m,g1b);self.bot2=Bot(p2,g2m,g2b);self.dir=os.path.join('data','games','benchmark');os.makedirs(self.dir,exist_ok=True)
        obp=os.path.join('data','processed','opening_book.json')
        try:self.ob=json.load(open(obp)) if os.path.isfile(obp) else{}
        except Exception as e:self.sc(f'Opening book load err:{e}');self.ob={}
        self._wb=None
    @property
    def wb(self):
        return self._wb
    def run(self):
        res={'1-0':0,'0-1':0,'1/2-1/2':0,'*':0};dur=[],mc=[];b1=b2=d=0;b1s=b2s=ds=[]
        col=True
        for gi in range(1,self.n+1):
            self.sc(f'Game {gi}/{self.n}');t0=time.time();bd=chess.Board();g=chess.pgn.Game()
            g.headers.update({'Event':'Bench','Site':'Local','Date':time.strftime('%Y.%m.%d'),'Round':str(gi),'White':'Bot1'if col else'Bot2','Black':'Bot2'if col else'Bot1','Result':'*'})
            node=g;mvs=0
            while not bd.is_game_over():
                m=(self.bot1 if(bd.turn==chess.WHITE)==col else self.bot2).get_move(bd,self.ob)
                if not m or m==chess.Move.null() or m not in bd.legal_moves:break
                bd.push(m);node=node.add_variation(m);mvs+=1
            r=bd.result();res[r]+=1;g.headers['Result']=r;open(os.path.join(self.dir,f'game_{gi}.pgn'),'w').write(str(g))
            dta=time.time()-t0;dur.append(dta);mc.append(mvs)
            if r=='1-0':
                if col:b1+=1
                else:b2+=1
            elif r=='0-1':
                if col:b2+=1
                else:b1+=1
            elif r=='1/2-1/2':d+=1
            b1s.append(b1);b2s.append(b2);ds.append(d)
            if self.wb:self.wb.log({'game_idx':gi,'result':r,'duration':dta,'moves':mvs,'bot1':b1,'bot2':b2,'draws':d})
            if self.sw:col=not col
        ad=float(np.mean(dur)) if dur else 0;am=float(np.mean(mc)) if mc else 0
        self.sc(f'Done. Bot1={b1},Bot2={b2},Draws={d},Unfinished={res["*"]}')
        if self.wb:
            self.wb.log({'total':self.n,'wins_bot1':b1,'wins_bot2':b2,'draws':d,'unfinished':res['*'],'avg_dur':ad,'avg_moves':am})
            tb=wandb.Table(data=[['1-0',b1],['0-1',b2],['1/2-1/2',d],['*',res['*']]],columns=['res','cnt'])
            self.wb.log({'res_dist':wandb.plot.bar(tb,'res','cnt',title='Results')})
            t2=wandb.Table(data=list(zip(range(1,self.n+1),b1s,b2s,ds)),columns=['game','bot1','bot2','draws'])
            self.wb.log({'wins_over_time':wandb.plot.line(t2,'game',['bot1','bot2','draws'],title='Cumulative')})
            self.wb.finish()
        return res