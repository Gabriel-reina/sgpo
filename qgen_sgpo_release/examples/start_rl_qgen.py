import argparse
import os
import sys

_EXAMPLES = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_EXAMPLES)
sys.path.insert(0, os.path.join(_REPO, "tools"))
from prepare_roll import installed_roll_root, verify_roll

_ROLL_ROOT = verify_roll(installed_roll_root())

from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.pipeline.rlvr.rlvr_config import RLVRConfig

def module_source(module_path):
    root = _ROLL_ROOT if module_path.startswith("roll.") else _REPO
    p = os.path.join(root, *module_path.split(".")) + ".py"
    with open(p) as f:
        return f.read()


def class_exists(module_path, cls_name):
    import importlib

    return hasattr(importlib.import_module(module_path), cls_name)


def dry_run_checks(cfg: RLVRConfig):
    from roll.datasets.chat_template import chat_templates

    for key in {cfg.global_template, cfg.actor_train.data_args.template,
                cfg.actor_infer.data_args.template}:
        assert key in chat_templates, f"template {key} 未注册，已注册: {sorted(chat_templates)}"

    for domain, rc in cfg.rewards.items():
        module, cls = rc.worker_cls.rsplit(".", 1)
        assert class_exists(module, cls), f"{rc.worker_cls} 不存在"
        print(f"[dry_run] reward {domain} -> {rc.worker_cls} ok, tags={rc.tag_included}")

    factory_src = module_source("roll.distributed.strategy.factory")
    for name in {cfg.actor_train.strategy_args.strategy_name,
                 cfg.actor_infer.strategy_args.strategy_name,
                 cfg.reference.strategy_args.strategy_name}:
        assert f'"{name}"' in factory_src, f"factory.py 里没有 {name} 分支"
        print(f"[dry_run] strategy {name} ok")

    ga = cfg.actor_infer.generating_args
    assert ga.do_sample, "do_sample 必须为 True，GRPO 需要同一 prompt 的多条独立采样"
    assert ga.num_return_sequences == cfg.num_return_sequences_in_group, (
        f"num_return_sequences({ga.num_return_sequences}) != "
        f"num_return_sequences_in_group({cfg.num_return_sequences_in_group})"
    )
    assert ga.top_k == -1 and ga.top_p == 1.0 and ga.temperature == 1.0, (
        f"temperature={ga.temperature} top_p={ga.top_p} top_k={ga.top_k}："
        f"本交付固定使用不截断的训练采样协议"
    )
    assert cfg.adv_estimator == "grpo", cfg.adv_estimator

    assert not cfg.whiten_advantages, "whiten_advantages 必须为 false，理由见 yaml 注释"
    assert cfg.advantage_clip is None, f"advantage_clip={cfg.advantage_clip} 必须留空"
    assert not cfg.dual_clip_loss, "dual_clip_loss 必须为 false"
    assert cfg.init_kl_coef == 0.0, (
        f"init_kl_coef={cfg.init_kl_coef} 必须显式为 0：>0 会走 token-level KL "
        f"并触发 old_logprobs 重算"
    )

    dp = len(cfg.actor_train.device_mapping)
    ta = cfg.actor_train.training_args
    rows = ta.per_device_train_batch_size * ta.gradient_accumulation_steps * dp
    prompt_rows = cfg.rollout_batch_size * cfg.num_return_sequences_in_group
    assert rows == prompt_rows, f"训练侧 {rows} 行 != rollout {prompt_rows} 行"

    for name, sa in (("actor_train", cfg.actor_train.strategy_args),
                     ("reference", cfg.reference.strategy_args)):
        tp = sa.strategy_config.get("tensor_model_parallel_size", 1)
        assert tp == 1, f"{name}.tensor_model_parallel_size={tp}，三臂协议固定 TP=1"

    mml = cfg.actor_infer.strategy_args.strategy_config.get("max_model_len")
    assert mml is None or mml >= cfg.sequence_length, (
        f"max_model_len={mml} < sequence_length({cfg.sequence_length})"
    )
    resp_len = cfg.sequence_length - cfg.prompt_length
    assert cfg.actor_infer.generating_args.max_new_tokens == resp_len, (
        f"max_new_tokens={cfg.actor_infer.generating_args.max_new_tokens} != {resp_len}"
    )

    if not os.path.isdir(cfg.pretrain):
        assert cfg.pretrain.count("/") == 1 and " " not in cfg.pretrain, (
            f"pretrain={cfg.pretrain!r} 既不是本地目录，也不像 <org>/<model> 仓库 id"
        )

    corpus = os.environ.get("QGEN_CORPUS", "")
    assert corpus, (
        "QGEN_CORPUS 未设置：reward worker 的 initialize 会当场断掉。"
        "按 README 手动 export，或通过 --job_env 指定。"
    )
    assert os.path.isfile(corpus), f"语料文件不存在：{corpus}"
    assert os.path.isabs(corpus), "QGEN_CORPUS 必须为绝对路径，避免 worker 工作目录不同"
    for path in cfg.actor_train.data_args.file_name:
        assert os.path.isfile(path), f"训练数据不存在：{path}"
    assert cfg.seed == 42 and cfg.max_steps == 600, "本交付三臂固定 seed=42、600 步"
    print(f"[dry_run] QGEN_CORPUS={corpus} 本地可读")

    print(f"[dry_run] RLVRConfig ok | exp_name={cfg.exp_name} max_steps={cfg.max_steps}")
    print(f"[dry_run] rollout {cfg.rollout_batch_size} prompt x "
          f"{cfg.num_return_sequences_in_group} 采样 = {prompt_rows} 行/步；"
          f"训练 {ta.per_device_train_batch_size} x {ta.gradient_accumulation_steps} x dp{dp}")
    print(f"[dry_run] len: prompt={cfg.prompt_length} response={resp_len} "
          f"sequence={cfg.sequence_length} max_model_len={mml}")
    print(f"[dry_run] pretrain={cfg.pretrain} | generate_opt_level={cfg.generate_opt_level}")
    print(f"[dry_run] train_file={cfg.actor_train.data_args.file_name}")


def set_job_env(spec: str) -> None:
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--job_env 每项必须是 K=V，收到: {item!r}")
        k, v = item.split("=", 1)
        os.environ[k.strip()] = v

    if os.environ.get('SGPO_ENABLE', '0') not in {'0', '1'}:
        raise ValueError('SGPO_ENABLE 必须为 0 或 1')
    if os.environ.get('QGEN_TOP_K', '100') != '100' or os.environ.get('QGEN_MATCH_MODE', 'or_all') != 'or_all':
        raise ValueError('历史三臂奖励协议固定 QGEN_TOP_K=100、QGEN_MATCH_MODE=or_all')

    got = {k: os.environ[k] for k in sorted(os.environ)
           if k.startswith(_WORKER_ENV_PREFIXES)}
    print(f"[job_env] 收到 {len(got)} 个: {got}", flush=True)

    assert os.environ.get("QGEN_CORPUS"), (
        "环境或 --job_env 中必须指定 QGEN_CORPUS。"
    )
    if os.environ.get("SGPO_ENABLE") == "1":
        from qgen_sgpo.sgpo import _sgpo_cfg

        required = ("SGPO_C_MAX", "SGPO_C_MIN", "SGPO_ANNEAL_STEPS",
                    "SGPO_TOKEN_RATIO", "SGPO_TOKEN_GRAD")
        missing = [k for k in required if k not in os.environ]
        if missing:
            raise ValueError(f"SGPO 正式提交必须显式设置：{missing}")
        cfg = _sgpo_cfg()
        if not (cfg["token_ratio"] or cfg["token_grad"]):
            raise ValueError("拒绝无逐 token 梯度的 SGPO：零均值重塑会抵消")


_WORKER_ENV_PREFIXES = ("QGEN_", "SGPO_")


def propagate_env_to_workers(cfg: RLVRConfig) -> None:
    payload = {k: v for k, v in os.environ.items() if k.startswith(_WORKER_ENV_PREFIXES)}
    if not payload:
        return

    targets = []
    for name in ("actor_train", "actor_infer", "reference", "critic"):
        wc = getattr(cfg, name, None)
        if wc is not None and hasattr(wc, "system_envs"):
            targets.append((name, wc))
    for name, wc in (getattr(cfg, "rewards", None) or {}).items():
        if hasattr(wc, "system_envs"):
            targets.append((f"rewards.{name}", wc))

    for _name, wc in targets:
        wc.system_envs.update(payload)

    print(f"[env] 注入 {len(payload)} 个变量到 {len(targets)} 个 worker: "
          f"{', '.join(n for n, _ in targets)}")
    print(f"[env] keys={sorted(payload)}")
    print(f"[env] QGEN_CORPUS={payload.get('QGEN_CORPUS', '未设置')}")
    if payload.get("SGPO_ENABLE") == "1":
        print(f"[env] SGPO={{{', '.join(f'{k}={v}' for k, v in sorted(payload.items()) if k.startswith('SGPO_'))}}}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="recr1-qgen-amazon")
    parser.add_argument("--config_name", default="qgen_grpo_3b_2gpu")
    parser.add_argument("--pretrain", default=os.environ.get("QGEN_PRETRAIN"),
                        help="固定 revision 下载的本地基础模型目录")
    parser.add_argument("--dry_run", action="store_true", help="只校验配置，不开训")
    parser.add_argument("--job_env", default="",
                        help="可选环境变量列表 K=V,K=V")
    args = parser.parse_args()

    set_job_env(args.job_env)

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)
    if args.pretrain:
        cfg.pretrain = os.path.abspath(args.pretrain)
        cfg.reward_pretrain = cfg.pretrain
        if not os.path.isfile(os.path.join(cfg.pretrain, "config.json")):
            raise FileNotFoundError(f"基础模型缺 config.json：{cfg.pretrain}")
    elif not args.dry_run:
        raise ValueError("正式复跑必须指定 QGEN_PRETRAIN 或 --pretrain，固定基础权重")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    rlvr_config: RLVRConfig = from_dict(
        data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True)
    )

    propagate_env_to_workers(rlvr_config)
    expected_sgpo = "sgpo" in args.config_name
    if expected_sgpo != (os.environ.get("SGPO_ENABLE", "0") == "1"):
        raise ValueError("配置名与 SGPO_ENABLE 不一致，拒绝运行错误算法")
    if expected_sgpo and os.environ.get("SGPO_SIGNAL", "confidence") != "confidence":
        raise ValueError("SGPO_SIGNAL 必须为 confidence")
    dry_run_checks(rlvr_config)

    if args.dry_run:
        return

    from roll.distributed.scheduler.initialize import init
    from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline

    init()

    pipeline = RLVRPipeline(pipeline_config=rlvr_config)
    pipeline.run()


if __name__ == "__main__":
    main()
