"""Item retrieval, SID retrieval and training-frequency-group diagnostics."""
import json
import math
from collections import defaultdict

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .retrieval import beam_search, expand_items


@torch.no_grad()
def validation_loss(model, dataset, batch_size, device):
    model.eval()
    total = 0.0
    for history, target, _ in DataLoader(dataset, batch_size=batch_size):
        logits = model(history.to(device), target.to(device))
        total += F.cross_entropy(logits.flatten(0, 1), target.to(device).flatten(), reduction='sum').item()
    return total / (len(dataset) * model.config.levels)


@torch.no_grad()
def evaluate(model, dataset, trie, groups, frequency, device, batch_size=128, beam_size=20, ks=(5, 10, 20), predictions_path=None):
    model.eval()
    accum = defaultdict(lambda: defaultdict(float))
    stream = open(predictions_path, 'w') if predictions_path else None
    row_index = 0
    try:
        for history, target, item_ids in DataLoader(dataset, batch_size=batch_size):
            history, target = history.to(device), target.to(device)
            logits = model(history, target)
            probability = logits.softmax(-1).gather(2, target[:, :, None]).squeeze(-1)
            correct = logits.argmax(-1) == target
            losses = F.cross_entropy(logits.transpose(1, 2), target, reduction='none')
            prefixes, scores, survives = beam_search(model, history, trie, beam_size, target)
            for j, item in enumerate(item_ids.tolist()):
                ranked = expand_items(prefixes[j].cpu(), scores[j].cpu(), trie, max(ks))
                rank = ranked.index(item) + 1 if item in ranked else None
                sid_rank = next((r + 1 for r, (p, s) in enumerate(zip(prefixes[j], scores[j]))
                    if s.isfinite() and torch.equal(p, target[j])), None)
                label = groups[item]
                for key in ['all', label]:
                    a = accum[key]
                    a['samples'] += 1
                    for depth in range(model.config.levels):
                        a[f'token_accuracy_l{depth + 1}'] += correct[j, depth].item()
                        a[f'target_probability_l{depth + 1}'] += probability[j, depth].item()
                        a[f'token_nll_l{depth + 1}'] += losses[j, depth].item()
                        a[f'beam_survival_l{depth + 1}'] += survives[j, depth].item()
                    for k in ks:
                        a[f'Recall@{k}'] += int(rank is not None and rank <= k)
                        a[f'NDCG@{k}'] += 1 / math.log2(rank + 1) if rank is not None and rank <= k else 0
                        a[f'SID_Recall@{k}'] += int(sid_rank is not None and sid_rank <= k)
                if stream:
                    stream.write(json.dumps({'row': row_index, 'target_item': item, 'group': label,
                        'training_frequency': frequency[item], 'target_sid': target[j].tolist(),
                        'candidate_items': ranked, 'target_rank': rank,
                        'target_token_probability': probability[j].tolist(),
                        'target_token_correct': correct[j].tolist(), 'beam_survival': survives[j].tolist()}) + '\n')
                row_index += 1
    finally:
        if stream:
            stream.close()
    return {group: {'samples': int(stats['samples']), **{name: value / stats['samples']
             for name, value in stats.items() if name != 'samples'}} for group, stats in accum.items()}
