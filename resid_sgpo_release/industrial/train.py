"""Train GRPO, GSPO or SGPO on the parallel-SID GR backbone."""
import sys,json,copy,time,hashlib,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
from resid_gr.model import GenerativeRecommender,ModelConfig
from resid_gr.data import load_data
from resid_gr.retrieval import SIDTrie,beam_search
from resid_gr.evaluate import validation_loss
from resid_gr.metrics import evaluate,metric_table,KS,report

def objective(lp,old,adv,method,step,steps):
    ratio=(lp-old).mean(-1).exp()
    if method=='grpo':
        token_ratio=(lp-old).exp()
        token_adv=adv.unsqueeze(-1)
        return -torch.minimum(token_ratio*token_adv,token_ratio.clamp(.9,1.1)*token_adv).mean()
    if method=='gspo':
        return -torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv).mean()
    u=lp.detach()-lp.detach().mean(-1,keepdim=True)
    c=1+3.0*max(0,1-step/200)
    lam=adv.abs().unsqueeze(-1)*torch.sqrt((c-1)/(u.square().mean(-1,keepdim=True)+1e-8))
    at=adv.unsqueeze(-1)+adv.sign().unsqueeze(-1)*lam*u
    eff=ratio.detach().unsqueeze(-1)*(lp-lp.detach()).exp()
    return -torch.minimum(eff*at,eff.clamp(.9,1.1)*at).mean()

def main(args):
    torch.set_num_threads(4)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    ck=ROOT/'checkpoints/base/gr.pt'
    state=torch.load(ck,weights_only=True);cfg=ModelConfig(**state['config'])
    ds,sids,freq,_,audit=load_data(ROOT/'data','Industrial_and_Scientific',cfg.max_history,cfg.code_size)
    groups={i:'zero' if freq[i]==0 else 'tail' if freq[i]<=6 else 'middle' if freq[i]<=11 else 'head' for i in sids}
    torch.manual_seed(args.seed); model=GenerativeRecommender(cfg);model.load_state_dict(state['model']);model.eval()
    ref=copy.deepcopy(model).eval();trie=SIDTrie(sids,cfg.code_size)
    steps=600;batch=32;rollouts=8
    manifest=dict(seed=args.seed,checkpoint='checkpoints/base/gr.pt',checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),audit=audit,
        sgpo_anneal_steps=200,lr_schedule='20-step warmup then cosine',gradient_clip=.5,steps=steps,batch=batch,rollout_beam=rollouts,eval_beam=20,lr=1e-5,clip=.1,kl=.001,sgpo_c_max=4,sgpo_c_min=1,
        reward='position-wise SID matches / L',rollout='catalogue constrained beam; raw model log-probs, deterministic top-beam surrogate as supplied SID implementation',
        selection='mean VALID all/tail Recall@50/100/200, checkpoints0/100/200/300/400/500/600, earlier tie',
        scope='Parallel6x16 SID GR with policy optimization and exact/partial matching',
        kl_definition='exact categorical KL(current || frozen initial) on visited prefixes; shared across all methods')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    gen=torch.Generator().manual_seed(args.seed); order=torch.randperm(len(ds['train']),generator=gen)
    initial_valid=validation_loss(model,ds['valid'],128,'cpu')
    results={'baseline':report(model,ds['test'],sids,trie,groups)}
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    for method in [args.method]:
        torch.manual_seed(args.seed);model.load_state_dict(state['model']);model.eval()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0)
        initial_metrics=report(model,ds['valid'],sids,trie,groups)
        def select_score(metrics):
            return float(np.mean([metrics['mask0-3'][g][f'Recall@{k}'] for g in ['all','tail'] for k in [50,100,200]]))
        bestscore=select_score(initial_metrics)
        valid_records=[dict(step=0,selection_score=bestscore,nll=initial_valid,metrics=initial_metrics)]
        best=initial_valid;beststate=copy.deepcopy(model.state_dict());beststep=0
        with (out/f'{method}.jsonl').open('w',buffering=1) as log:
            for step in range(steps):
                warm=min(1.,(step+1)/20);decay=.5*(1+math.cos(math.pi*max(0,step-20)/max(1,steps-20)))
                opt.param_groups[0]['lr']=1e-5*warm*decay
                ix=order[step*batch:(step+1)*batch];h=ds['train'].history[ix];target=ds['train'].targets[ix]
                with torch.no_grad():
                    paths,scores,_=beam_search(model,h,trie,rollouts)
                    valid=scores.isfinite()
                    assert valid.all(), 'Insufficient legal rollouts'
                    hh=h.repeat_interleave(rollouts,0); tokens=paths.flatten(0,1)
                    oldlogits=model(hh,tokens)
                    old=oldlogits.log_softmax(-1).gather(-1,tokens.unsqueeze(-1)).squeeze(-1).reshape(batch,rollouts,-1)
                    reward=(paths==target[:,None]).float().mean(-1)
                    adv=(reward-reward.mean(1,keepdim=True))/(reward.std(1,keepdim=True,unbiased=False)+1e-6)
                    refprob=ref(hh,tokens).log_softmax(-1)
                logits=model(hh,tokens); logp=logits.log_softmax(-1)
                lp=logp.gather(-1,tokens.unsqueeze(-1)).squeeze(-1).reshape(batch,rollouts,-1)
                pg=objective(lp,old,adv,method,step,steps)
                kl=(logp.exp()*(logp-refprob)).sum(-1).mean()
                loss=pg+.001*kl
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
                rec=dict(step=step+1,loss=loss.item(),reward=reward.mean().item(),kl=kl.item(),grad_norm=float(gn),nonconstant_groups=int((reward.std(1)>0).sum()))
                if (step+1)%100==0:
                    v=validation_loss(model,ds['valid'],128,'cpu');rec['valid_nll']=v
                    vm=report(model,ds['valid'],sids,trie,groups);vs=select_score(vm)
                    rec['valid_selection_score']=vs
                    valid_records.append(dict(step=step+1,selection_score=vs,nll=v,metrics=vm))
                    (out/'valid-checkpoints.json').write_text(json.dumps(valid_records,indent=2))
                    if vs>bestscore:bestscore=vs;best=v;beststate=copy.deepcopy(model.state_dict());beststep=step+1
                    torch.save(dict(model=model.state_dict(),config=state['config'],step=step+1),out/f'{method}-step{step+1}.pt')
                log.write(json.dumps(rec)+'\n')
                if (step+1)%10==0:print(method,rec,flush=True)
        model.load_state_dict(beststate)
        torch.save(dict(model=beststate,config=state['config'],step=beststep),out/f'{method}-best.pt')
        results[method]={'selected_step':beststep,'valid_nll':best,'valid_selection_score':bestscore,'metrics':report(model,ds['test'],sids,trie,groups)}
        (out/'summary.json').write_text(json.dumps(results,indent=2))
    lines=['# Public Industrial: GR + GSPO / SGPO', '', 'RL experiment; beam20; TRAIN frequency groups: tail1-6, middle7-11, head>=12; zero separate.', '', 'Checkpoint selection: mean VALID all/tail Recall@50/100/200.', '']
    for matching in ['exact','mask0-3']:
        lines += ['## '+matching,'','| Method | Group | K | Recall | NDCG | MRR | Effective coverage | Exposure coverage |','|---|---|---:|---:|---:|---:|---:|---:|']
        for name,entry in results.items():
            metrics=entry if name=='baseline' else entry['metrics']
            for group in ['all','head','tail']:
                r=metrics[matching][group]
                for k in KS:
                    lines.append(f"| {name} | {group} | {k} | {r[f'Recall@{k}']:.6f} | {r[f'NDCG@{k}']:.6f} | {r[f'MRR@{k}']:.6f} | {r[f'EffectiveCoverage@{k}']:.6f} | {r[f'ExposureCoverage@{k}']:.6f} |")
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    (out/'DONE').write_text('completed\n')
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['grpo','gspo','sgpo'],required=True)
    parser.add_argument('--seed',type=int,choices=[42,43,44,45,46],required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
