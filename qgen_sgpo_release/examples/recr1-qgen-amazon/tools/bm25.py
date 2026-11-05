
import importlib.util
import os
import sys

_REAL = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "qgen_sgpo", "qgen_bm25.py",
))
assert os.path.isfile(_REAL), f"找不到实现文件: {_REAL}"

_spec = importlib.util.spec_from_file_location("_qgen_bm25_impl", _REAL)
_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_qgen_bm25_impl", _mod)
_spec.loader.exec_module(_mod)

STOPWORDS = _mod.STOPWORDS
DEFAULT_TOP_K = _mod.DEFAULT_TOP_K
tokenize = _mod.tokenize
parse_lucene_query = _mod.parse_lucene_query
BM25Index = _mod.BM25Index
load_corpus = _mod.load_corpus
dcg_at_k = _mod.dcg_at_k
ndcg_at_k = _mod.ndcg_at_k
