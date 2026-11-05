"""Raw history/next-item data; catalogue SID lookup and training-only groups."""
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SIDData(Dataset):
    def __init__(self, history, targets, item_ids, users, raw_histories):
        self.history, self.targets, self.item_ids = history, targets, item_ids
        self.users, self.raw_histories = users, raw_histories

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.history[i], self.targets[i], self.item_ids[i]


def load_data(data_dir, dataset, max_history=10, code_size=None):
    root = Path(data_dir)
    index_path = root / f'{dataset}.index.json'
    raw = json.loads(index_path.read_text())
    sids = {}
    for key, tokens in raw.items():
        codes = []
        for level, token in enumerate(tokens):
            match = re.fullmatch(r'<([a-z])_(\d+)>', token)
            if not match or match[1] != chr(97 + level):
                raise ValueError(f'Invalid SID token {token}')
            codes.append(int(match[2]))
        sids[int(key)] = codes
    levels = len(next(iter(sids.values())))
    if not levels or any(len(code) != levels for code in sids.values()):
        raise ValueError('SID lengths must be uniform')
    observed_size = max(max(code) for code in sids.values()) + 1
    code_size = observed_size if code_size is None else code_size
    if code_size < observed_size:
        raise ValueError('Declared code size is smaller than observed SID tokens')
    splits, hashes, counts = {}, {}, {}
    hashes[index_path.name] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    for split in ['train', 'valid', 'test']:
        # Distribution uses explicit split filenames to make inputs unambiguous.
        path = root / f'{split}.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        histories, targets, item_ids, users, raw_histories = [], [], [], [], []
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                history = [int(x) for x in ast.literal_eval(row['history_item_id'])]
                target = int(row['item_id'])
                if target not in sids or any(i not in sids for i in history):
                    raise ValueError('Item without SID')
                if row.get('item_sid') and ''.join(raw[str(target)]) != row['item_sid']:
                    raise ValueError('Target SID mapping mismatch')
                if row.get('history_item_sid'):
                    tokens = ast.literal_eval(row['history_item_sid'])
                    if len(tokens) != len(history) or any(''.join(raw[str(i)]) != t for i, t in zip(history, tokens)):
                        raise ValueError('History SID mapping mismatch')
                visible = history[-max_history:]
                hist = [sids[i] for i in visible] + [[code_size] * levels] * (max_history - len(visible))
                histories.append(hist)
                targets.append(sids[target])
                item_ids.append(target)
                users.append(row['user_id'])
                raw_histories.append(history)
        splits[split] = SIDData(torch.tensor(histories), torch.tensor(targets),
                               torch.tensor(item_ids), users, raw_histories)
        counts[split] = {'rows': len(targets), 'empty_history': sum(not h for h in raw_histories),
                         'truncated_history': sum(len(h) > max_history for h in raw_histories)}
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    frequency = Counter(splits['train'].item_ids.tolist())
    positive = Counter(frequency.values())
    rank, cumulative = {}, 0
    for value, n in sorted(positive.items()):
        rank[value] = (cumulative + n / 2) / sum(positive.values())
        cumulative += n
    group = {i: 'zero' if frequency[i] == 0 else 'tail' if frequency[i] <= 6
             else 'head' if frequency[i] >= 12 else 'middle' for i in sids}
    keys = {split: [(u, tuple(h), int(i)) for u, h, i in zip(d.users, d.raw_histories, d.item_ids)]
            for split, d in splits.items()}
    overlap = {f'{a}-{b}': len(set(keys[a]) & set(keys[b]))
               for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]}
    audit = {'splits': counts, 'sha256': hashes, 'exact_example_overlap': overlap,
             'within_split_duplicate_rows': {s: len(k) - len(set(k)) for s, k in keys.items()},
             'sid_levels': levels, 'observed_code_range': [0, observed_size - 1], 'catalogue_items': len(sids),
             'model_code_size': code_size,
             'history_policy': 'Keep the entire input history, truncating only beyond max_history; right padding uses a separate ID.',
             'group_policy': 'TRAIN target counts: tail1-6, middle7-11, head>=12; zero separate.'}
    return splits, sids, frequency, group, audit
