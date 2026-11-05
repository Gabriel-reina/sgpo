"""Item-level exact and partial matching with deterministic item ties."""
import math
import numpy as np
import torch
from .retrieval import beam_search
KS=[1,5,10,20,50,100,200]

def evaluate(codes,paths,scores,targets,labels,methods):
    ranks={name:np.zeros(len(targets),dtype=np.int16) for name in methods}
    predictions={name:np.full((len(targets),200),-1,dtype=np.int32) for name in methods}
    for row in range(len(targets)):
        seeds=paths[row][np.isfinite(scores[row])]
        distance=(codes[:,None,:]!=seeds[None,:,:]).sum(-1)
        count=distance.min(1);best_beam=distance.argmin(1)
        order=np.lexsort((np.arange(len(codes)),best_beam,count))
        for name,tolerance in methods.items():
            selected=order[count[order]<=tolerance][:200]
            predictions[name][row,:len(selected)]=selected
            hit=np.flatnonzero(selected==targets[row])
            ranks[name][row]=int(hit[0]+1) if len(hit) else 0
    return ranks,predictions,None,None

def metric_table(ranks,labels):
    result={}
    for group in ['all','head','tail','middle','zero']:
        r=ranks if group=='all' else ranks[labels==group]
        values={'samples':len(r)}
        for k in KS:
            good=(r>0)&(r<=k);safe=np.maximum(r,1).astype(float)
            values[f'Recall@{k}']=float(good.mean())
            values[f'NDCG@{k}']=float(np.where(good,1/np.log2(safe+1),0).mean())
            values[f'MRR@{k}']=float(np.where(good,1/safe,0).mean())
        result[group]=values
    return result

@torch.no_grad()
def report(model,data,sids,trie,groups,rank_output=None):
    model.eval();paths=[];scores=[]
    for start in range(0,len(data),128):
        p,s,_=beam_search(model,data.history[start:start+128],trie,20)
        paths.append(p.numpy());scores.append(s.numpy())
    targets=data.item_ids.numpy();labels=np.array([groups[int(i)] for i in targets])
    codes=np.array([sids[i] for i in range(len(sids))])
    ranks,predictions,_,_=evaluate(codes,np.concatenate(paths),np.concatenate(scores),targets,labels,{'exact':0,'mask0-3':3})
    if rank_output is not None:np.savez_compressed(rank_output,**ranks)
    result={}
    for method,rr in ranks.items():
        result[method]=metric_table(rr,labels)
        for group in ['all','head','tail','middle','zero']:
            rows=np.ones(len(targets),bool) if group=='all' else labels==group
            catalog=set(sids) if group=='all' else {i for i in sids if groups[i]==group}
            target_set=set(targets[rows].tolist())
            for k in KS:
                hit=rows&(rr>0)&(rr<=k)
                result[method][group][f'EffectiveCoverage@{k}']=len(set(targets[hit].tolist()))/max(1,len(target_set))
                result[method][group][f'ExposureCoverage@{k}']=len(set(predictions[method][:,:k].flatten().tolist())&catalog)/max(1,len(catalog))
    return result
