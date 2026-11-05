
import json
import math
import re
from collections import defaultdict

import numpy as np

__all__ = [
    "STOPWORDS", "tokenize", "parse_lucene_query", "BM25Index", "load_corpus",
    "dcg_at_k", "ndcg_at_k", "extract_solution", "validate_response_structure",
    "check_json_format", "calculate_answer_score", "compute_score", "DEFAULT_TOP_K",
]

DEFAULT_TOP_K = 100

STOPWORDS = frozenset(
    """a an and are as at be but by for if in into is it no not of on or such
    that the their then there these they this to was will with""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BOOL_OPS = frozenset({"and", "or", "not"})

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ASSISTANT_HEADERS = ("<|im_start|>assistant", "Assistant:")


def _light_stem(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("sses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    return w


def tokenize(text, stem="light"):
    toks = _TOKEN_RE.findall(text.lower())
    out = []
    for t in toks:
        if t in STOPWORDS:
            continue
        out.append(_light_stem(t) if stem == "light" else t)
    return out


def parse_lucene_query(query, stem="light", mode="or_all"):
    cleaned = re.sub(r"\bcontents\s*:", " ", query, flags=re.I)
    cleaned = re.sub(r"[()\[\]{}\"~^*?+\\/:]", " ", cleaned)

    if mode == "or_all":
        return tokenize(cleaned, stem=stem), []

    raw_tokens = re.findall(r"[A-Za-z0-9]+", cleaned)
    must, should, pending_and = [], [], False
    for tok in raw_tokens:
        if tok in ("AND",):
            pending_and = True
            continue
        if tok in ("OR", "NOT"):
            pending_and = False
            continue
        low = tok.lower()
        if low in STOPWORDS or low in _BOOL_OPS:
            continue
        term = _light_stem(low) if stem == "light" else low
        if pending_and:
            must.append(term)
            if should and should[-1] not in must:
                must.append(should.pop())
            pending_and = False
        else:
            should.append(term)
    return should, must


class BM25Index:

    def __init__(self, k1=1.2, b=0.75, stem="light"):
        self.k1 = k1
        self.b = b
        self.stem = stem
        self.doc_ids = []
        self.doc_texts = []
        self.doc_len = None
        self.avgdl = 0.0
        self.postings = {}
        self.idf = {}

    def build(self, doc_ids, doc_texts):
        n_docs = len(doc_ids)
        self.doc_ids = list(doc_ids)
        self.doc_texts = list(doc_texts)

        tf_lists = defaultdict(list)
        doc_len = np.zeros(n_docs, dtype=np.float32)
        for i, text in enumerate(doc_texts):
            toks = tokenize(text, stem=self.stem)
            doc_len[i] = len(toks)
            counts = defaultdict(int)
            for t in toks:
                counts[t] += 1
            for t, c in counts.items():
                tf_lists[t].append((i, c))

        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if n_docs else 0.0

        for term, pairs in tf_lists.items():
            idx = np.fromiter((p[0] for p in pairs), dtype=np.int32, count=len(pairs))
            tf = np.fromiter((p[1] for p in pairs), dtype=np.float32, count=len(pairs))
            dl = doc_len[idx]
            denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            df = len(pairs)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            self.postings[term] = (idx, (idf * tf * (self.k1 + 1.0) / denom).astype(np.float32))
            self.idf[term] = idf
        return self

    def search(self, query, top_k=DEFAULT_TOP_K, mode="or_all"):
        should, must = parse_lucene_query(query, stem=self.stem, mode=mode)
        if not should and not must:
            return []

        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        matched = False
        for term in should + must:
            post = self.postings.get(term)
            if post is None:
                continue
            idx, contrib = post
            np.add.at(scores, idx, contrib)
            matched = True
        if not matched:
            return []

        if must:
            keep = np.ones(len(self.doc_ids), dtype=bool)
            for term in set(must):
                post = self.postings.get(term)
                if post is None:
                    return []
                hit = np.zeros(len(self.doc_ids), dtype=bool)
                hit[post[0]] = True
                keep &= hit
            scores = np.where(keep, scores, 0.0)

        n_hit = int((scores > 0).sum())
        if n_hit == 0:
            return []
        k = min(top_k, n_hit)
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part], kind="stable")]
        return [(self.doc_ids[i], self.doc_texts[i], float(scores[i])) for i in order]

    def batch_search(self, queries, top_k=DEFAULT_TOP_K, mode="or_all"):
        return {q: self.search(q, top_k=top_k, mode=mode) for q in queries}

    @classmethod
    def from_corpus(cls, path, **kwargs):
        doc_ids, doc_texts = load_corpus(path)
        return cls(**kwargs).build(doc_ids, doc_texts)


def load_corpus(path):
    doc_ids, doc_texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            doc_ids.append(str(d["item_id"]))
            doc_texts.append(d["metadata"].strip())
    return doc_ids, doc_texts


def dcg_at_k(retrieved, target, k):
    retrieved = retrieved[:k]
    gains = [1.0 if item == target else 0.0 for item in retrieved]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(retrieved, target, k):
    dcg = dcg_at_k(retrieved, target, k)
    ideal_dcg = dcg_at_k([target], target, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0



def extract_solution(solution_str):
    processed_str = solution_str
    for header in _ASSISTANT_HEADERS:
        if header in solution_str:
            processed_str = solution_str.split(header, 1)[1].strip()
            break

    matches = _ANSWER_RE.findall(processed_str)
    if matches:
        return matches[-1].strip(), processed_str
    return None, processed_str


def validate_response_structure(processed_str):
    tags = {
        "think_start": ("<think>", 1),
        "think_end": ("</think>", 1),
        "answer_start": ("<answer>", 1),
        "answer_end": ("</answer>", 1),
    }
    passed = True
    positions = {}
    for name, (tag, expected) in tags.items():
        positions[name] = processed_str.find(tag)
        if processed_str.count(tag) != expected:
            passed = False

    if (positions["think_start"] > positions["think_end"]
            or positions["think_end"] > positions["answer_start"]
            or positions["answer_start"] > positions["answer_end"]):
        passed = False
    return passed


def check_json_format(json_str):
    if not json_str:
        return False
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and "query" in data


def calculate_answer_score(json_str, label, top_k, retriever, mode="or_all", strict=False):
    try:
        query = json.loads(json_str)["query"]
        if not isinstance(query, str):
            query = str(query)
        hits = retriever.search(query, top_k=top_k, mode=mode)
        item_ids = [h[0] for h in hits]
        return ndcg_at_k(item_ids, label, top_k)
    except Exception:
        if strict:
            raise
        return -2.0


def _resolve_label(ground_truth):
    if isinstance(ground_truth, dict):
        return str(ground_truth["target"])
    return str(ground_truth)


def compute_score(solution_str, ground_truth, retriever=None, format_reward=0.1,
                  top_k=DEFAULT_TOP_K, mode="or_all", strict=False, return_detail=False):
    if retriever is None:
        raise ValueError("retriever 不能为空：缺检索器会让所有 reward 静默退化，必须显式传入")

    label = _resolve_label(ground_truth)

    answer_text, processed_str = extract_solution(solution_str)
    format_correct = validate_response_structure(processed_str) and check_json_format(answer_text)
    format_score = format_reward if format_correct else -2.0

    answer_score = 0.0
    if format_correct and answer_text:
        answer_score = calculate_answer_score(answer_text, label, top_k, retriever,
                                              mode=mode, strict=strict)

    if answer_score > 0:
        total = format_score + answer_score
    elif format_score > 0:
        total = 0.0
    else:
        total = format_score

    if return_detail:
        return {"total": float(total), "format": float(format_score),
                "answer": float(answer_score), "top_k": top_k,
                "format_correct": format_correct, "answer_text": answer_text}
    return float(total)
