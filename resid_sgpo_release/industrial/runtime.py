from pathlib import Path
import json
import torch
from resid_gr.model import GenerativeRecommender,ModelConfig
from resid_gr.data import load_data
from resid_gr.retrieval import SIDTrie
ROOT=Path(__file__).resolve().parent

def load_checkpoint(checkpoint):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=ModelConfig(**state['config'])
    splits,sids,freq,_,audit=load_data(ROOT/'data','Industrial_and_Scientific',cfg.max_history,cfg.code_size)
    groups={i:'zero' if freq[i]==0 else 'tail' if freq[i]<=6 else 'middle' if freq[i]<=11 else 'head' for i in sids}
    model=GenerativeRecommender(cfg).eval();model.load_state_dict(state['model'])
    return model,splits,sids,freq,groups,SIDTrie(sids,cfg.code_size),audit
