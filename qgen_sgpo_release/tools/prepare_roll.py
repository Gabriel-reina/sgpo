import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'patches/manifest.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def verify_roll(root, expected='after'):
    root = Path(root).resolve()
    if root == ROOT or ROOT in root.parents:
        raise RuntimeError('外部 ROLL 必须放在交付目录之外')
    manifest = json.loads(MANIFEST.read_text())
    if git(root, 'rev-parse', 'HEAD') != manifest['commit']:
        raise RuntimeError(f"ROLL 必须固定到 {manifest['commit']}")
    for name, hashes in manifest['files'].items():
        if digest(root / name) != hashes[expected]:
            raise RuntimeError(f'ROLL 文件校验失败：{name} ({expected})')
    patch = ROOT / 'patches/roll-v0.3.0-sgpo.patch'
    if digest(patch) != manifest['patch_sha256']:
        raise RuntimeError('补丁 SHA256 不匹配')
    dirty = set(git(root, 'diff', '--name-only', 'HEAD').splitlines())
    if not dirty.issubset(manifest['files']):
        raise RuntimeError(f'ROLL 含补丁以外的修改：{sorted(dirty)}')
    untracked = git(root, 'ls-files', '--others', '--exclude-standard').splitlines()
    injected = [name for name in untracked if name.startswith(('roll/', 'mcore_adapter/'))
                and name.endswith(('.py', '.so'))]
    if injected:
        raise RuntimeError(f'ROLL 含未跟踪的源码覆盖：{injected}')
    return root


def installed_roll_root():
    spec = importlib.util.find_spec('roll')
    if spec is None or spec.origin is None:
        raise RuntimeError('未安装外部 ROLL；按 README 安装固定版本的 editable 工作副本')
    return Path(spec.origin).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roll-root', type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--runtime', action='store_true', help='检查 GPU 训练依赖可导入并保存环境记录')
    parser.add_argument('--report', type=Path, default=Path('environment.json'))
    args = parser.parse_args()
    root = args.roll_root or installed_roll_root()
    if args.apply:
        manifest = json.loads(MANIFEST.read_text())
        states = {name: digest(root / name) for name in manifest['files']}
        if all(states[n] == h['after'] for n, h in manifest['files'].items()):
            verify_roll(root)
            print('补丁已应用且校验通过；无需重复应用。')
        else:
            verify_roll(root, 'before')
            patch = ROOT / 'patches/roll-v0.3.0-sgpo.patch'
            subprocess.run(['git', '-C', str(root), 'apply', '--check', str(patch)], check=True)
            subprocess.run(['git', '-C', str(root), 'apply', str(patch)], check=True)
    verify_roll(root)
    print(f'ROLL 固定版本及补丁 SHA256 校验通过：{root}')
    if args.runtime:
        import importlib
        import importlib.metadata
        report = {'python': sys.version, 'roll_commit': git(root, 'rev-parse', 'HEAD'),
                  'packages': {}, 'imports': {}, 'gpu': {}, 'errors': []}
        for module in ('torch', 'transformers', 'vllm', 'ray', 'datasets',
                       'megatron.core', 'mcore_adapter', 'transformer_engine.pytorch',
                       'flash_attn', 'qgen_sgpo.reward_worker',
                       'roll.pipeline.rlvr.rlvr_pipeline'):
            try:
                importlib.import_module(module)
                report['imports'][module] = 'ok'
            except Exception as exc:
                report['imports'][module] = str(exc)
                report['errors'].append(module)
        for name in ('torch', 'transformers', 'vllm', 'ray', 'datasets', 'tensordict',
                     'megatron-core', 'mcore_adapter', 'transformer-engine', 'flash-attn'):
            try:
                report['packages'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report['packages'][name] = 'missing'
        try:
            import torch
            report['gpu'] = {'cuda': torch.version.cuda, 'available': torch.cuda.is_available(),
                             'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
            if not torch.cuda.is_available():
                report['errors'].append('CUDA 不可用')
        except Exception as exc:
            report['errors'].append(str(exc))
        report['pip_freeze'] = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(f'环境检查报告：{args.report}；错误数：{len(report["errors"])}')
        if report['errors']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
