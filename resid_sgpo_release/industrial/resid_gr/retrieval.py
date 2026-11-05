"""Catalogue-constrained autoregressive beam search with prefix diagnostics."""
from collections import defaultdict

import torch


class SIDTrie:
    def __init__(self, sids, code_size, device='cpu'):
        self.levels = len(next(iter(sids.values())))
        self.leaves = defaultdict(list)
        prefixes = [{()}] + [set() for _ in range(self.levels)]
        for item, code in sids.items():
            self.leaves[tuple(code)].append(item)
            for depth in range(1, self.levels + 1):
                prefixes[depth].add(tuple(code[:depth]))
        nodes = [{key: i for i, key in enumerate(sorted(keys))} for keys in prefixes]
        self.transitions = []
        for depth in range(self.levels):
            transition = torch.full((len(nodes[depth]), code_size), -1, dtype=torch.long)
            for key, child in nodes[depth + 1].items():
                transition[nodes[depth][key[:-1]], key[-1]] = child
            self.transitions.append(transition.to(device))
        for items in self.leaves.values():
            items.sort()


@torch.no_grad()
def beam_search(model, history, trie, beam_size=20, target=None):
    memory, padding = model.encode(history)
    batch, device = len(history), history.device
    prefix = torch.empty(batch, 1, 0, dtype=torch.long, device=device)
    score = torch.zeros(batch, 1, device=device)
    nodes = torch.zeros(batch, 1, dtype=torch.long, device=device)
    survives = []
    for depth in range(trie.levels):
        width = prefix.shape[1]
        mem = memory.repeat_interleave(width, dim=0)
        pad = padding.repeat_interleave(width, dim=0)
        logits = model.decode(mem, pad, prefix.reshape(batch * width, depth))[:, -1]
        # Preserve original model probabilities; masking does not renormalize valid children.
        logp = logits.log_softmax(-1).reshape(batch, width, -1)
        next_nodes = trie.transitions[depth][nodes.clamp_min(0)]
        candidates = (score[:, :, None] + logp).masked_fill(next_nodes < 0, -torch.inf)
        count = min(beam_size, candidates.shape[1] * candidates.shape[2])
        score, indices = candidates.flatten(1).topk(count, dim=1)
        parent, token = indices // logits.shape[-1], indices % logits.shape[-1]
        old_prefix = prefix.gather(1, parent[:, :, None].expand(-1, -1, depth))
        prefix = torch.cat([old_prefix, token[:, :, None]], -1)
        nodes = next_nodes.gather(1, parent[:, :, None].expand(-1, -1, next_nodes.shape[-1])).gather(2, token[:, :, None]).squeeze(-1)
        if target is not None:
            survives.append(((prefix == target[:, None, :depth + 1]).all(-1) & score.isfinite()).any(1))
    return prefix, score, torch.stack(survives, 1) if target is not None else None


def expand_items(prefixes, scores, trie, k):
    candidates = []
    for prefix, score in zip(prefixes, scores):
        if torch.isfinite(score):
            # Equal SID means equal score; ties resolved by item ID, never target or popularity.
            candidates.extend(trie.leaves[tuple(prefix.tolist())])
    return candidates[:k]
