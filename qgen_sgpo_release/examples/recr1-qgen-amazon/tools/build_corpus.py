
import argparse
import ast
import html
import json
import os
import re

def clean_text(raw):
    if raw is None:
        return ""
    if isinstance(raw, list):
        parts = [str(x) for x in raw]
    elif isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                val = ast.literal_eval(s)
                parts = [str(x) for x in val] if isinstance(val, (list, tuple)) else [s]
            except (ValueError, SyntaxError):
                parts = [s]
        else:
            parts = [s]
    else:
        parts = [str(raw)]

    text = " ".join(parts)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build(item_json_path, out_path, fields, max_chars):
    with open(item_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_written, n_empty, lens = 0, 0, []
    with open(out_path, "w", encoding="utf-8") as fout:
        for key in sorted(items.keys(), key=lambda x: int(x)):
            meta = items[key]
            pieces = [clean_text(meta.get(fld)) for fld in fields]
            metadata = " ".join(p for p in pieces if p).strip()
            if max_chars > 0:
                metadata = metadata[:max_chars]
            if not metadata:
                n_empty += 1
            lens.append(len(metadata))
            rec = {
                "item_id": str(key),
                "metadata": metadata,
                "title": clean_text(meta.get("title")),
                "category": clean_text(meta.get("categories")),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    lens.sort()
    print(f"[corpus] {out_path}")
    print(f"  items={n_written}  empty_metadata={n_empty}  fields={fields}")
    if lens:
        print(
            f"  metadata chars: median={lens[len(lens)//2]} "
            f"p90={lens[int(len(lens)*0.9)]} max={lens[-1]}"
        )
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item_json", required=True, help="Amazon 商品元数据 JSON 路径")
    ap.add_argument("--out_dir", required=True, help="重建语料的输出目录")
    ap.add_argument("--fields", default="title,description,brand",
                    help="拼进 metadata 的字段，逗号分隔，顺序即拼接顺序")
    ap.add_argument("--max_chars", type=int, default=0,
                    help=">0 时截断 metadata；BM25 对长文档已有 b=0.75 的长度惩罚，默认不截")
    args = ap.parse_args()

    item_json = args.item_json
    if not os.path.exists(item_json):
        raise FileNotFoundError(f"找不到 item.json：{item_json}")

    out_path = os.path.join(args.out_dir, "corpus.jsonl")
    build(item_json, out_path, [f.strip() for f in args.fields.split(",") if f.strip()],
          args.max_chars)


if __name__ == "__main__":
    main()
