import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_data(data_rows):
    corpus_name = 'data/corpus.jsonl'
    splits = ('data/rl_train.jsonl', 'data/rl_val.jsonl', 'data/rl_test.jsonl')
    if set(data_rows) != {corpus_name, *splits}:
        raise RuntimeError('数据清单必须包含语料及 train/val/test 四个文件')
    corpus_ids = set()
    with (ROOT / corpus_name).open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        item_id = row.get('item_id')
        if not isinstance(item_id, str) or not item_id or item_id in corpus_ids:
            raise RuntimeError('语料 item_id 必须为唯一的非空字符串')
        if not isinstance(row.get('metadata'), str):
            raise RuntimeError('语料缺少文本 metadata')
        corpus_ids.add(item_id)
    if len(rows) != data_rows[corpus_name]:
        raise RuntimeError(f'数据行数不匹配：{corpus_name}')
    print(f'{corpus_name}: {len(rows)} 行，数据结构校验通过')
    for name in splits:
        count = 0
        with (ROOT / name).open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                prompt = row.get('prompt')
                if not isinstance(prompt, str) or not prompt.endswith('<|im_start|>assistant\n'):
                    raise RuntimeError(f'prompt 格式不符合已模板化数据约定：{name}:{count}')
                if row.get('ground_truth') not in corpus_ids:
                    raise RuntimeError(f'ground_truth 不在语料：{name}:{count}')
                if row.get('tag') != 'qgen' or row.get('domain') != 'qgen':
                    raise RuntimeError(f'tag/domain 不符合配置：{name}:{count}')
        if count != data_rows[name]:
            raise RuntimeError(f'数据行数不匹配：{name}')
        print(f'{name}: {count} 行，数据结构校验通过')


def main():
    if (ROOT / 'roll').exists():
        raise RuntimeError('交付目录不应包含 roll；框架必须放在目录外')
    manifest = json.loads((ROOT / 'tools/release_manifest.json').read_text())
    if not set(manifest['data_rows']).issubset(manifest['sha256']):
        raise RuntimeError('数据文件必须全部包含在 SHA256 清单中')
    for name, expected in manifest['sha256'].items():
        path = ROOT / name
        if path.is_symlink() or ROOT not in path.resolve().parents:
            raise RuntimeError(f'清单路径必须是交付目录内的普通文件：{name}')
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f'交付文件 SHA256 不匹配：{name}')
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    verify_data(manifest['data_rows'])
    print(f'通过：{len(manifest["sha256"])} 个交付文件、Python 语法与数据结构。')


if __name__ == '__main__':
    main()
