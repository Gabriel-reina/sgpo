"""Evaluate the shipped final-step checkpoints or a new checkpoint."""
import argparse,json
from pathlib import Path
import torch
from runtime import ROOT,load_checkpoint
from resid_gr.metrics import report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');ap.add_argument('--checkpoint',type=Path);ap.add_argument('--split',choices=['valid','test'],default='valid');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.all==bool(args.checkpoint):ap.error('Choose --all or --checkpoint')
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=False)
    jobs=[(f'{s}-{m}',ROOT/f'checkpoints/seed{s}/{m}.pt') for s in range(42,47) for m in ['grpo','gspo','sgpo']] if args.all else [('checkpoint',args.checkpoint)]
    expected=json.loads((ROOT/'references/expected_metrics.json').read_text())['per_seed'];all_results={}
    for name,ck in jobs:
        model,splits,sids,freq,groups,trie,audit=load_checkpoint(ck)
        metrics=report(model,splits[args.split],sids,trie,groups,rank_output=args.output/f'{name}-ranks.npz')
        if args.all and args.split=='valid':
            for matching in ['exact','mask0-3']:
                for k in [5,10,20]:
                    for metric,old in [('Recall','HitRate'),('NDCG','NDCG'),('MRR','MRR')]:
                        error=abs(metrics[matching]['all'][f'{metric}@{k}']-expected[name][f'{old}@{k}'])
                        if error>1e-12:raise AssertionError(f'{name}/{matching}/{metric}@{k}: {error}')
        all_results[name]=metrics
        (args.output/'metrics.json').write_text(json.dumps(all_results,indent=2))
        print(f'Completed {name}',flush=True)
    (args.output/'DONE').write_text('completed\n')
if __name__=='__main__':main()
