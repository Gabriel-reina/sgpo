
import argparse
import json
import os
import math
from pathlib import Path
import sys
import time


def resolve_model_in_subprocess(uri, workdir):
    path = Path(uri).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"本地 HF checkpoint 不存在或缺 config.json：{path}")
    if not (list(path.glob("*.safetensors")) or list(path.glob("pytorch_model*.bin"))):
        raise FileNotFoundError(f"checkpoint 缺 HF 权重：{path}")
    return str(path)


def load_scorer(corpus_path, top_k, match_mode):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from qgen_sgpo import qgen_bm25

    t0 = time.time()
    retriever = qgen_bm25.BM25Index.from_corpus(corpus_path)
    print(f"BM25 索引 build 完成: {len(retriever.doc_ids)} 篇文档  "
          f"{time.time() - t0:.2f}s  top_k={top_k}  mode={match_mode}", flush=True)
    return qgen_bm25, retriever


def score_one(mod, retriever, text, gt, top_k, match_mode):
    d = mod.compute_score(text, gt, retriever=retriever, top_k=top_k,
                          mode=match_mode, return_detail=True)
    label = str(gt)
    rank, n_hits, query = 0, 0, None
    if d["format_correct"] and d["answer_text"]:
        try:
            query = json.loads(d["answer_text"])["query"]
            if not isinstance(query, str):
                query = str(query)
            hits = retriever.search(query, top_k=top_k, mode=match_mode)
            n_hits = len(hits)
            for i, h in enumerate(hits):
                if h[0] == label:
                    rank = i + 1
                    break
        except Exception:
            pass
    return {
        "total": d["total"], "format": d["format"], "answer": d["answer"],
        "format_correct": bool(d["format_correct"]),
        "rank": rank, "n_retrieved": n_hits, "query": query,
    }


def compute_metrics(records, top_k):
    n = len(records)
    if n == 0:
        return {"n": 0}

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    fmt_ok = [r for r in records if r["format_correct"]]
    hits = [r for r in records if 0 < r["rank"] <= top_k]
    ranks = [r["rank"] for r in hits]

    ks = [1, 2, 3, 5, 10, 20, 30, 50, 100]
    ks = [k for k in ks if k <= top_k]
    m = {
        "n": n,
        "top_k": top_k,
        "score_mean": mean([r["total"] for r in records]),
        "score_min": min(r["total"] for r in records),
        "score_max": max(r["total"] for r in records),
        "format_correct_rate": len(fmt_ok) / n,
        "format_score_mean": mean([r["format"] for r in records]),
        "answer_ndcg_mean": mean([max(r["answer"], 0.0) for r in records]),
        "answer_ndcg_mean_on_format_ok": mean([max(r["answer"], 0.0) for r in fmt_ok]),
        "hit_rate": len(hits) / n,
        "ndcg_mean_on_hit": mean([r["answer"] for r in hits]),
        "mrr": sum(1.0 / r for r in ranks) / n,
        "rank_mean_on_hit": mean([float(r) for r in ranks]),
        "rank_median_on_hit": (sorted(ranks)[len(ranks) // 2] if ranks else 0),
        "empty_retrieval_rate": sum(1 for r in records if r["n_retrieved"] == 0) / n,
        "answer_error_rate": sum(1 for r in records if r["answer"] < 0) / n,
        "query_none_rate": sum(1 for r in records if r["query"] is None) / n,
        "query_word_mean": mean([len(r["query"].split()) for r in records
                                 if r["query"]]),
    }
    for k in ks:
        m[f"hit@{k}"] = sum(1 for r in ranks if r <= k) / n
        m[f"ndcg@{k}"] = sum(1.0 / math.log2(r + 1) for r in ranks if r <= k) / n
        m[f"mrr@{k}"] = sum(1.0 / r for r in ranks if r <= k) / n
    return m


METRIC_ORDER = [
    "n", "top_k",
    "score_mean", "score_min", "score_max",
    "format_correct_rate", "format_score_mean",
    "answer_ndcg_mean", "answer_ndcg_mean_on_format_ok",
    "hit_rate", "ndcg_mean_on_hit", "mrr",
    "rank_mean_on_hit", "rank_median_on_hit",
    *[f"{metric}@{k}" for metric in ("ndcg", "mrr", "hit") for k in (1, 2, 3, 5, 10, 20, 30, 50, 100)],
    "empty_retrieval_rate", "answer_error_rate", "query_none_rate",
    "query_word_mean", "gen_len_mean",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True,
                    help="本地 HF checkpoint 目录；不支持仓库 URI")
    ap.add_argument("--version", default="local", help="仅作输出元数据，不拼接模型路径")
    ap.add_argument("--ckpt_ids", default="checkpoint-599",
                    help="本地 checkpoint 的输出标签；一次仅评一个 checkpoint")
    ap.add_argument("--data_file", required=True, help="rl_val.jsonl 或 rl_test.jsonl")
    ap.add_argument("--corpus", required=True, help="corpus.jsonl，与训练同一份")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--top_k", type=int, default=int(os.environ.get("QGEN_TOP_K", "100")))
    ap.add_argument("--match_mode", default=os.environ.get("QGEN_MATCH_MODE", "or_all"))
    ap.add_argument("--max_new_tokens", type=int, default=256, help="= 训练 response_length")
    ap.add_argument("--max_model_len", type=int, default=896, help="= prompt_length+response_length")
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 条，用于冒烟")
    ap.add_argument("--keep_weights", action="store_true",
                    help="兼容旧参数；现在始终保留权重")
    ap.add_argument("--tag", default="qgen-eval")
    args = ap.parse_args()
    if args.world_size != 1 or int(os.environ.get("RANK", "0")) != 0:
        ap.error("公开主表评测使用单张 GPU、world_size=1、RANK=0")
    if args.top_k != 100 or args.match_mode != "or_all":
        ap.error("主表协议固定 top_k=100、match_mode=or_all")
    if (',' in args.ckpt_ids or args.ckpt_ids in {'', '.', '..'}
            or Path(args.ckpt_ids).name != args.ckpt_ids):
        ap.error("ckpt_ids 必须是单个安全输出标签")
    if args.limit < 0 or args.max_new_tokens <= 0 or args.max_model_len <= 0:
        ap.error("limit 必须非负，生成和上下文长度必须为正")

    rank = int(os.environ.get("RANK", "0"))
    ws = args.world_size
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[rank{rank}/{ws}] args={vars(args)}", flush=True)

    recs = []
    with open(args.data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if args.limit > 0:
        recs = recs[: args.limit]
    if not recs:
        raise ValueError('评测数据为空，不能产出主表指标')
    model_dir = resolve_model_in_subprocess(args.model_name, args.out_dir)
    for row in recs:
        if not isinstance(row.get('prompt'), str) or 'ground_truth' not in row:
            raise ValueError('评测记录缺少 prompt 或 ground_truth')

    start = rank * len(recs) // ws
    end = (rank + 1) * len(recs) // ws
    shard = recs[start:end]
    print(f"[rank{rank}] 全量 {len(recs)} 条，本分片 [{start}:{end}) = {len(shard)} 条", flush=True)

    mod, retriever = load_scorer(args.corpus, args.top_k, args.match_mode)

    from vllm import LLM, SamplingParams

    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=-1,
                        max_tokens=args.max_new_tokens)

    ckpt_ids = [args.ckpt_ids]
    for ckpt_id in ckpt_ids:
        shard_path = os.path.join(args.out_dir, f"{ckpt_id}_shard_{rank}.json")
        skip_path = os.path.join(args.out_dir, f"{ckpt_id}_skipped_{rank}.json")
        if os.path.exists(shard_path) or os.path.exists(skip_path):
            raise FileExistsError("输出目录已有旧评测记录；请为本次评测选择新目录")

        t0 = time.time()
        t_dl = 0.0
        print(f"[rank{rank}] {ckpt_id} 使用本地权重 {model_dir}", flush=True)

        llm = LLM(model=model_dir, dtype="bfloat16", trust_remote_code=True,
                  max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_mem_util)

        t1 = time.time()
        outs = llm.generate([r["prompt"] for r in shard], sp)
        if len(outs) != len(shard) or any(len(o.outputs) != 1 for o in outs):
            raise RuntimeError('生成输出数量与评测数据不一致，拒绝缩小指标分母')
        t_gen = time.time() - t1
        print(f"[rank{rank}] {ckpt_id} 生成完成 {len(outs)} 条  {t_gen:.0f}s", flush=True)

        out_recs, gen_lens = [], []
        for r, o in zip(shard, outs):
            text = o.outputs[0].text
            gen_lens.append(len(o.outputs[0].token_ids))
            d = score_one(mod, retriever, text, r["ground_truth"],
                          args.top_k, args.match_mode)
            d["ground_truth"] = str(r["ground_truth"])
            d["gen_len"] = gen_lens[-1]
            d["response"] = text
            out_recs.append(d)

        with open(shard_path, "w") as f:
            json.dump({"rank": rank, "start": start, "end": end,
                       "ckpt_id": ckpt_id, "t_download": t_dl, "t_generate": t_gen,
                       "records": out_recs}, f)
        print(f"[rank{rank}] {ckpt_id} 分片写出 {shard_path}  用时 {time.time() - t0:.0f}s",
              flush=True)

        del llm
        try:
            import gc

            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[rank{rank}] 显存回收异常（不影响本轮结果）: {e}", flush=True)


    all_metrics = {}
    for ckpt_id in ckpt_ids:
        with open(os.path.join(args.out_dir, f"{ckpt_id}_shard_0.json")) as f:
            merged = json.load(f)['records']
        if len(merged) != len(recs):
            raise RuntimeError('输出记录数量与全量数据不一致')

        m = compute_metrics(merged, args.top_k)
        m["gen_len_mean"] = sum(r["gen_len"] for r in merged) / len(merged)
        m["ckpt_id"] = ckpt_id
        m["model_name"] = args.model_name
        m["version"] = args.version
        m["data_file"] = os.path.basename(args.data_file)
        all_metrics[ckpt_id] = m
        with open(os.path.join(args.out_dir, f"metrics_{ckpt_id}.json"), "w") as f:
            json.dump(m, f, indent=2)
        with open(os.path.join(args.out_dir, f"predictions_{ckpt_id}.json"), "w") as f:
            json.dump(merged, f)

    with open(os.path.join(args.out_dir, "metrics_all.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 100, flush=True)
    print(f"qgen 离线评测  {args.model_name}  {os.path.basename(args.data_file)}  "
          f"top_k={args.top_k}  mode={args.match_mode}  贪心解码", flush=True)
    print("=" * 100, flush=True)
    head = f"{'metric':<32}" + "".join(f"{c.replace('checkpoint-', 'step'):>14}"
                                       for c in ckpt_ids)
    print(head, flush=True)
    for key in METRIC_ORDER:
        row = f"{key:<32}"
        for c in ckpt_ids:
            v = all_metrics.get(c, {}).get(key)
            if v is None:
                row += f"{'--':>14}"
            elif isinstance(v, int):
                row += f"{v:>14d}"
            else:
                row += f"{v:>14.5f}"
        print(row, flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    sys.exit(main())
