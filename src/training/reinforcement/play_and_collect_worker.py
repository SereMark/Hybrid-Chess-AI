import time
import chess
import torch
import numpy as np
from src.training.reinforcement.mcts import MCTS
from src.models.transformer import TransformerChessModel
from src.utils.train_utils import initialize_random_seeds
from src.utils.chess_utils import convert_board_to_transformer_input, get_move_mapping, get_total_moves

class PlayAndCollectWorker:
    @classmethod
    def run_process(cls, model_state_dict, device_type, simulations, c_puct, temperature, games_per_process, seed):
        initialize_random_seeds(seed)
        device=torch.device(device_type)
        model=TransformerChessModel(get_total_moves())
        try:
            model.load_state_dict(model_state_dict,strict=False)
        except:
            return ([],[],[],{},[])
        model.to(device).eval()
        inputs,policy_targets,value_targets=[],[],[]
        results=[]
        pgn_games=[]
        stats={"wins":0,"losses":0,"draws":0,"game_lengths":[],"results":[]}
        move_mapping=get_move_mapping()
        for _ in range(games_per_process):
            board=chess.Board()
            mcts=MCTS(model,device,c_puct,simulations)
            mcts.set_root_node(board)
            states=[]
            mcts_probs=[]
            current_players=[]
            move_count=0
            max_moves=get_total_moves()
            game=chess.pgn.Game()
            game.headers.update({"Event":"Reinforcement Self-Play","Site":"Self-Play","Date":time.strftime("%Y.%m.%d"),"Round":"-","White":"Agent","Black":"Opponent","Result":"*"})
            node=game
            while not board.is_game_over() and move_count<max_moves:
                action_probs=mcts.get_move_probs(temperature)
                if move_count==0:
                    moves_list=list(action_probs.keys())
                    if moves_list:
                        noise=np.random.dirichlet([0.3]*len(moves_list))
                        for i,mv in enumerate(moves_list):
                            action_probs[mv]=0.75*action_probs[mv]+0.25*noise[i]
                if not action_probs:
                    break
                moves_available=list(action_probs.keys())
                probs=np.array(list(action_probs.values()),dtype=np.float32)
                probs/=probs.sum()
                chosen_move=np.random.choice(moves_available,p=probs)
                states.append(convert_board_to_transformer_input(board))
                prob_arr=np.zeros(get_total_moves(),dtype=np.float32)
                for mv,prob in action_probs.items():
                    idx=move_mapping.get_index_by_move(mv)
                    if idx is not None and 0<=idx<get_total_moves():
                        prob_arr[idx]=prob
                mcts_probs.append(prob_arr)
                current_players.append(board.turn)
                try:
                    board.push(chosen_move)
                except ValueError:
                    break
                node=node.add_variation(chosen_move)
                mcts.update_with_move(chosen_move)
                move_count+=1
            result_map={'1-0':1.0,'0-1':-1.0,'1/2-1/2':0.0}
            outcome=result_map.get(board.result(),0.0)
            winners=[]
            for pl in current_players:
                sign=outcome if pl==chess.WHITE else -outcome
                winners.append(sign)
            game.headers["Result"]='1-0' if outcome>0 else '0-1' if outcome<0 else '1/2-1/2'
            pgn_exporter=chess.pgn.StringExporter(headers=True,variations=True,comments=True)
            game_string=game.accept(pgn_exporter)
            pgn_games.append(game_string)
            inputs.extend(states)
            policy_targets.extend(mcts_probs)
            value_targets.extend(winners)
            results.append(outcome)
            stats['game_lengths'].append(move_count)
            stats['results'].append(outcome)
            if outcome==1.0:
                stats['wins']+=1
            elif outcome==-1.0:
                stats['losses']+=1
            else:
                stats['draws']+=1
        return (inputs,policy_targets,value_targets,stats,pgn_games)