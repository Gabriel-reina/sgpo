import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bm25 import BM25Index, DEFAULT_TOP_K, load_corpus
from reward_qgen import compute_score

DEFAULT_DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..", "..", "data"))
DEFAULT_TOKENIZER = os.environ.get('QGEN_PRETRAIN')
PROMPT_LENGTH = 640
RESPONSE_LENGTH = 256

_fails = []


def check(ok, msg):
    print(f"  [{'ok ' if ok else 'FAIL'}] {msg}")
    if not ok:
        _fails.append(msg)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_schema(rows, name):
    print(f"\n[schema] {name}  {len(rows)} 行")
    check(len(rows) > 0, "非空")
    bad_keys = [i for i, r in enumerate(rows[:2000])
                if set(r) != {"prompt", "ground_truth", "tag", "domain"}]
    check(not bad_keys, f"字段集合恒为 prompt/ground_truth/tag/domain（异常行 {bad_keys[:3]}）")
    check(all(r["tag"] == "qgen" and r["domain"] == "qgen" for r in rows),
          "tag/domain 全为 qgen（对上 yaml 的 tag_included 与 domain_interleave_probs）")
    check(all(isinstance(r["ground_truth"], str) and r["ground_truth"] for r in rows),
          "ground_truth 全为非空字符串")
    tail = "<|im_start|>assistant\n"
    check(all(r["prompt"].endswith(tail) for r in rows), f"prompt 全部以 {tail!r} 结尾")
    check(all(r["prompt"].startswith("<|im_start|>system\n") for r in rows),
          "prompt 全部以 system 段开头")


def check_labels_in_corpus(rows_by_split, corpus_ids, name):
    print(f"\n[label] ground_truth 是否都能在 corpus（{len(corpus_ids)} 个 item）里找到")
    for split, rows in rows_by_split.items():
        gts = {r["ground_truth"] for r in rows}
        missing = sorted(gts - corpus_ids)
        check(not missing,
              f"{split}: {len(gts)} 个不同 label 全部命中（缺失 {len(missing)} 个"
              f"{'，例: ' + str(missing[:3]) if missing else ''}）")


def check_lengths(rows, tokenizer_path):
    print(f"\n[length] prompt token <= prompt_length({PROMPT_LENGTH})")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        check(False, "没装 transformers，长度检查跳过（这一条必须在有 tokenizer 的机器上过）")
        return
    if not os.path.isdir(tokenizer_path):
        check(False, f"tokenizer 路径不存在: {tokenizer_path}")
        return
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    lens = sorted(len(tok(r["prompt"], add_special_tokens=False)["input_ids"]) for r in rows)
    n = len(lens)
    print(f"  token: median={lens[n // 2]} p99={lens[int(n * 0.99)]} max={lens[-1]}")
    check(lens[-1] <= PROMPT_LENGTH, f"最长 {lens[-1]} <= {PROMPT_LENGTH}")
    check(lens[0] > 5, f"最短 {lens[0]} > 5")


def check_reward_semantics(index):
    print("\n[reward] 分段语义")
    any_id = index.doc_ids[0]

    oracle_q = " ".join(index.doc_texts[0].split()[:8])
    d = compute_score('<think>r</think>\n<answer>' + json.dumps({"query": oracle_q}) +
                      '</answer>', any_id, retriever=index,
                      top_k=DEFAULT_TOP_K, return_detail=True)
    check(d["answer"] > 0, f"oracle query 确实命中: answer={d['answer']:.4f}")
    check(abs(d["total"] - (d["format"] + d["answer"])) < 1e-6,
          f"answer>0 时 total == format + answer（{d['total']:.4f} == "
          f"{d['format']:.4f} + {d['answer']:.4f}）")

    d = compute_score('<think>r</think>\n<answer>{"query": "zzzqqqxxxnotaword"}</answer>',
                      any_id, retriever=index, top_k=DEFAULT_TOP_K, return_detail=True)
    check(d["format"] > 0 and abs(d["total"]) < 1e-6,
          f"格式对但检索无果: total={d['total']:.4f} format={d['format']}"
          f"（total 必须为 0，否则模型能靠空 query 白拿格式分）")

    good_q = "wrench socket steel"
    bad_cases = [
        ("缺 think 标签", f'<answer>{{"query": "{good_q}"}}</answer>'),
        ("answer 不是合法 json", '<think>r</think>\n<answer>wrench socket</answer>'),
        ("json 里没有 query 键", '<think>r</think>\n<answer>{"q": "wrench"}</answer>'),
        ("标签顺序反了", f'<answer>{{"query": "{good_q}"}}</answer>\n<think>r</think>'),
    ]
    for desc, resp in bad_cases:
        d = compute_score(resp, any_id, retriever=index,
                          top_k=DEFAULT_TOP_K, return_detail=True)
        check(abs(d["total"] - (-2.0)) < 1e-6,
              f"{desc}: total={d['total']:.4f}（期望 -2.0）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="公开基础模型的本地 tokenizer 目录；也可设置 QGEN_PRETRAIN")
    ap.add_argument("--n_len", type=int, default=4000,
                    help="长度检查抽多少条 train（全量 3.6w 条跑 tokenizer 太慢）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.tokenizer:
        ap.error('必须指定 --tokenizer 或 QGEN_PRETRAIN，不能借用其他模型的 tokenizer')
    if args.n_len <= 0:
        ap.error('n_len 必须为正数')
    _fails.clear()

    data_dir = os.path.abspath(args.data_dir)
    print(f"data_dir = {data_dir}")
    print(f"top_k    = {DEFAULT_TOP_K}（与 README 的三臂奖励协议一致）")

    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    assert os.path.isfile(corpus_path), f"缺 {corpus_path}，先跑 tools/build_corpus.py"
    doc_ids, doc_texts = load_corpus(corpus_path)
    index = BM25Index()
    index.build(doc_ids, doc_texts)
    corpus_ids = set(doc_ids)
    print(f"corpus   = {len(doc_ids)} item")

    rows_by_split = {}
    for split, name in (("train", "rl_train.jsonl"), ("val", "rl_val.jsonl"),
                        ("test", "rl_test.jsonl")):
        p = os.path.join(data_dir, name)
        assert os.path.isfile(p), f"缺 {p}，先跑 tools/export_rl_jsonl.py"
        rows_by_split[split] = read_jsonl(p)

    for split, rows in rows_by_split.items():
        check_schema(rows, split)
    check_labels_in_corpus(rows_by_split, corpus_ids, "all")

    rnd = random.Random(args.seed)
    train = rows_by_split["train"]
    check_lengths(rnd.sample(train, min(args.n_len, len(train))), os.path.abspath(args.tokenizer))
    check_reward_semantics(index)

    print("\n" + "=" * 60)
    if _fails:
        print(f"{len(_fails)} 项未通过：")
        for m in _fails:
            print(f"  - {m}")
        return 1
    print("数据与奖励预检通过；不代表 GPU 训练或完整评测已通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
