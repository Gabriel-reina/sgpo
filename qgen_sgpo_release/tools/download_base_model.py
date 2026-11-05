import argparse
import hashlib
import json
from pathlib import Path

MODEL = 'Qwen/Qwen2.5-3B-Instruct'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('模型输出目录非空；请使用新目录以防混入其他权重')
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=str(output),
                      allow_patterns=['*.json', '*.safetensors', '*.txt', 'LICENSE'])
    hashes = {}
    for path in sorted(output.iterdir()):
        if path.is_file():
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            hashes[path.name] = h.hexdigest()
    (output / 'qgen_base_manifest.json').write_text(json.dumps(
        {'model': MODEL, 'revision': REVISION, 'sha256': hashes}, indent=2) + '\n')
    print(f'基础权重就绪：{output}；revision={REVISION}')


if __name__ == '__main__':
    main()
