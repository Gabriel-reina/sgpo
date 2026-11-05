"""Recompute single-target metrics and paired seed-level significance."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--evaluated',type=Path);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    D={}
    if args.evaluated:
        raw=json.loads(args.evaluated.read_text())
        for key,values in raw.items():
            v=values['mask0-3']['all'];D[key]={f'{m}@{k}':v[f'{"Recall" if m=="HitRate" else m}@{k}'] for m in ['HitRate','NDCG','MRR'] for k in [5,10,20]}
    else:
        for seed in range(42,47):
            for method in ['grpo','gspo','sgpo']:
                with np.load(ROOT/f'references/seed{seed}-{method}-ranks.npz') as f:r=f['mask0-3'].astype(float)
                d={}
                for k in [5,10,20]:
                    good=(r>0)&(r<=k);safe=np.maximum(r,1)
                    d.update({f'HitRate@{k}':float(good.mean()),f'NDCG@{k}':float(np.where(good,1/np.log2(safe+1),0).mean()),f'MRR@{k}':float(np.where(good,1/safe,0).mean())})
                D[f'{seed}-{method}']=d
    methods=['grpo','gspo','sgpo'];summary={};tests=[]
    for m in ['HitRate','NDCG','MRR']:
        for k in [5,10,20]:
            key=f'{m}@{k}';v={n:np.array([D[f'{s}-{n}'][key] for s in range(42,47)]) for n in methods}
            summary[key]={n:{'mean':float(x.mean()),'sample_std':float(x.std(ddof=1))} for n,x in v.items()}
            # Rounded sorting treats exact-count ties consistently.
            order=sorted(methods,key=lambda n:round(float(v[n].mean()),14),reverse=True);best,second=order[:2]
            t,p=stats.ttest_rel(v[best],v[second]);tests.append(dict(metric=key,best=best,runner_up=second,t=float(t),p_two_sided=float(p),n=5,df=4,significant=bool(p<.05)))
    result=dict(per_seed=D,summary=summary,tests=tests,validation_requests=4866,unit='paired RL seed',multiplicity='unadjusted')
    (args.output/'statistics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
