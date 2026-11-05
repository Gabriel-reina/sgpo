
import argparse
import ast
import csv
import json
import os
import re

MAX_HIST = 5
MAX_TITLE_CHARS = 120

SYSTEM = (
    "You are a search query generator for an e-commerce product search engine. "
    "Given a user's purchase history, you write one keyword query that retrieves "
    "the product the user will buy next."
)

USER_TMPL = """The user has bought the following {desc} products, in chronological order (oldest first):
{history}

Write ONE keyword query that retrieves the product this user will buy next from a {desc} product catalogue. The catalogue is searched with BM25 over each product's title, description and brand.

Rules:
- Use only words that would literally appear in the target product's title, description or brand.
- Prefer specific attributes (material, size, model number, brand, standard) over generic category words. Generic category words retrieve thousands of products and are close to useless here.
- Combine terms with OR. Use AND only for a term that the target product must contain.
- Do not copy a whole product title verbatim, and do not invent product names.

Reason first, then answer. Output exactly this format:
<think> your reasoning </think>
<answer> {{"query": "term1 OR term2 OR term3"}} </answer>"""


def parse_title_list(cell):
    cell = (cell or "").strip()
    if not cell:
        return []
    if not (cell.startswith("[") and cell.endswith("]")):
        return [cell]
    try:
        val = ast.literal_eval(cell)
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val]
    except (ValueError, SyntaxError):
        pass
    pairs = re.findall(r"'([^']*)'|\"([^\"]*)\"", cell)
    return [a or b for a, b in pairs if (a or b)]


def load_split(csv_path, limit=None):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "hist_titles": parse_title_list(row["history_item_title"]),
                "target_id": str(row["item_id"]).strip(),
            })
            if limit and len(rows) >= limit:
                break
    return rows


def build_prompt(hist_titles, desc):
    kept = [t.strip() for t in hist_titles if t and t.strip()][-MAX_HIST:]
    history = "\n".join(f"{i + 1}. {t[:MAX_TITLE_CHARS]}" for i, t in enumerate(kept))
    user = USER_TMPL.format(desc=desc, history=history)
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def export(csv_path, out_path, desc, limit=None):
    rows = load_split(csv_path, limit=limit)
    n_skip = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    prompts = []
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            if not r["hist_titles"] or not r["target_id"]:
                n_skip += 1
                continue
            prompt = build_prompt(r["hist_titles"], desc)
            prompts.append(prompt)
            f.write(json.dumps({
                "prompt": prompt,
                "ground_truth": str(r["target_id"]),
                "tag": "qgen",
                "domain": "qgen",
            }, ensure_ascii=False) + "\n")
    print(f"[export] {out_path}: {len(prompts)} 行（跳过无历史 {n_skip} 行）")
    return prompts


def report_token_lengths(prompts, tokenizer_path):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[len] 没装 transformers，跳过 token 长度统计")
        return
    if not tokenizer_path or not os.path.isdir(tokenizer_path):
        print(f"[len] tokenizer 路径不可用（{tokenizer_path}），跳过 token 长度统计")
        return
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    lens = sorted(len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts)
    n = len(lens)
    print(f"[len] tokenizer={tokenizer_path} vocab={len(tok)}")
    print(f"[len] prompt token: median={lens[n // 2]} p90={lens[int(n * 0.9)]} "
          f"p99={lens[int(n * 0.99)]} max={lens[-1]}")
    for cap in (512, 768, 1024, 1280):
        kept = sum(1 for x in lens if x <= cap)
        print(f"       prompt_length={cap} -> 保留 {kept}/{n} = {100.0 * kept / n:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer", default=os.environ.get("QGEN_PRETRAIN"),
                    help="同一基础模型的 tokenizer，仅用于统计长度")
    ap.add_argument("--limit", type=int, default=0, help=">0 时每个 split 只导这么多行（冒烟用）")
    args = ap.parse_args()

    desc = "industrial and scientific"
    out_dir = args.out_dir
    limit = args.limit or None

    all_prompts = []
    for split, csv_path, out_name in (("train", args.train_csv, "rl_train.jsonl"),
                                     ("valid", args.val_csv, "rl_val.jsonl"),
                                     ("test", args.test_csv, "rl_test.jsonl")):
        assert os.path.isfile(csv_path), f"缺文件: {csv_path}"
        ps = export(csv_path, os.path.join(out_dir, out_name), desc, limit=limit)
        if split == "train":
            all_prompts = ps

    print("\n--- prompt 样例（train 第 1 条）---")
    print(all_prompts[0])
    print("--- 样例结束 ---\n")
    if args.tokenizer:
        report_token_lengths(all_prompts, os.path.abspath(args.tokenizer))


if __name__ == "__main__":
    main()
