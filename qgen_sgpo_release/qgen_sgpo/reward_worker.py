import os

import torch

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from qgen_sgpo.qgen_bm25 import DEFAULT_TOP_K, BM25Index, compute_score
from roll.utils.logging import get_logger

logger = get_logger()

class QgenBM25RewardWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.retriever = None
        self.top_k = None
        self.mode = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        corpus = os.environ.get("QGEN_CORPUS", "")
        assert corpus and os.path.isfile(corpus), (
            f"QGEN_CORPUS 必须指向 build_corpus.py 产出的 corpus.jsonl，当前 = {corpus!r}"
        )
        self.top_k = int(os.environ.get("QGEN_TOP_K", str(DEFAULT_TOP_K)))
        self.mode = os.environ.get("QGEN_MATCH_MODE", "or_all")

        self.retriever = BM25Index.from_corpus(corpus)
        logger.info(
            f"[qgen-reward] 索引就绪: docs={len(self.retriever.doc_ids)} "
            f"vocab={len(self.retriever.postings)} avgdl={self.retriever.avgdl:.1f} "
            f"| top_k={self.top_k} ({100.0 * self.top_k / max(len(self.retriever.doc_ids), 1):.1f}% "
            f"of pool) mode={self.mode}"
        )

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        batch_size = len(response_text_list)
        ground_truths = data.non_tensor_batch["ground_truth"]

        totals, ndcgs = [], []
        n_format_ok = n_retrieved = 0
        for response, gt in zip(response_text_list, ground_truths):
            d = compute_score(response, gt, retriever=self.retriever,
                              top_k=self.top_k, mode=self.mode, return_detail=True)
            totals.append(d["total"])
            ndcgs.append(max(d["answer"], 0.0))
            if d["format_correct"]:
                n_format_ok += 1
            if d["answer"] > 0:
                n_retrieved += 1

        mean_ndcg = sum(ndcgs) / max(batch_size, 1)
        logger.info(
            f"[qgen-reward] rows={batch_size} format_ok={n_format_ok}/{batch_size} "
            f"hit@{self.top_k}={n_retrieved}/{batch_size} "
            f"mean_ndcg={mean_ndcg:.4f} "
            f"mean_total={sum(totals) / max(batch_size, 1):.4f}"
        )

        output_tensors = {
            "token_level_rewards": torch.zeros_like(data.batch["responses"], dtype=torch.float16),
            "response_level_rewards": torch.tensor(totals, dtype=torch.float16),
            "scores": torch.tensor(ndcgs, dtype=torch.float16),
        }
        return DataProto.from_dict(tensors=output_tensors)
