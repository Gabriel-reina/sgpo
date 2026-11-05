"""Verify release inputs, final weights and saved rank metrics."""
from pathlib import Path
import json,hashlib
import torch
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,want in manifest['sha256'].items():
        path=ROOT/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=want:raise AssertionError(name)
    expected=json.loads((ROOT/'references/expected_metrics.json').read_text())['per_seed']
    for seed in range(42,47):
        for method in ['grpo','gspo','sgpo']:
            state=torch.load(ROOT/f'checkpoints/seed{seed}/{method}.pt',map_location='cpu',weights_only=True)
            assert state['step']==600 and state['config']['levels']==6 and state['config']['code_size']==16
            with np.load(ROOT/f'references/seed{seed}-{method}-ranks.npz') as f:
                for matching in ['exact','mask0-3']:
                    ranks=f[matching].astype(float);assert len(ranks)==4532
                    for k in [5,10,20]:
                        good=(ranks>0)&(ranks<=k);safe=np.maximum(ranks,1)
                        actual={f'HitRate@{k}':float(good.mean()),f'NDCG@{k}':float(np.where(good,1/np.log2(safe+1),0).mean()),f'MRR@{k}':float(np.where(good,1/safe,0).mean())}
                        for name,value in actual.items():assert value==expected[f'{seed}-{method}'][name]
    print('Verified input hashes, 15 final checkpoints, and all saved HR/NDCG/MRR reference metrics.')
if __name__=='__main__':main()
