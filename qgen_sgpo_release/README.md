# Query Generation for Document Retrieval: GRPO / GSPO / SGPO

**GRPO / GSPO / SGPO** for LLM query generation (Rec-R1 style). Given a user's interaction history, the model generates a query for BM25 product retrieval and learns from the retrieval reward.

Base model: **Qwen2.5-3B-Instruct**.

## Task Setup

- Data: a query generation task built from the Amazon **Industrial_and_Scientific** review data
  (corpus of 3,686 products / training samples in `data/rl_train.jsonl` / validation set of 4,532 examples)
- Generation format: `<think>...</think><answer>{"query": "..."}</answer>`; each of the four tags must occur exactly once, in that order, and the answer must be valid JSON containing `query`.
- Reward (single source of truth: `qgen_sgpo/qgen_bm25.py`):
  - format score: +0.1 for correct format, −2.0 otherwise;
  - answer score: parse the query, retrieve with BM25 (`top_k=100`, `or_all`), and compute the NDCG@100 of the gold document;
  - **total reward**: −2 for invalid format; 0 for valid format without the gold document in the retrieved results; otherwise `0.1 + NDCG@100`.
- Retriever: NumPy Okapi BM25 (`k1=1.2`, `b=0.75`) with light stemming, shared by all three methods.
- Evaluation protocol: greedy decoding, full 4,532-example validation set, BM25 top100/or_all, single relevant item (IDCG=1)

## The Three Methods

| Arm | Config | Description |
|---|---|---|
| GRPO | `qgen_grpo_3b_2gpu.yaml` | token-level importance ratio baseline |
| GSPO | `qgen_gspo_3b_2gpu_s42.yaml` | sequence-level ratio (`importance_sampling: seq`); differs from GRPO only in the ratio granularity |
| SGPO | `qgen_sgpo_tg_c4_a200_3b_2gpu.yaml` | confidence-guided token-level advantage reshaping, formula below |

Core of SGPO (`qgen_sgpo/sgpo.py`):

```
A_t = A_s + sign(A_s) · λ_k · u_t
λ_k = |A_s| · sqrt((c_n − 1) / (mean(u_t²) + ε))
c_n = max(c_min, c_max − (c_max − c_min) · n / N)     # linear annealing
```

- `u_t`: log π_new minus its sequence mean over valid response tokens (stop-gradient), using `SGPO_SIGNAL=confidence`
- `c_max=4.0 → c_min=1.0`, annealed over `N=200` training steps; at `c_n=1`, the task gradient matches GSPO under the same masks and aggregation.
- sequence ratio in the forward pass, per-token gradients (`SGPO_TOKEN_RATIO=0, SGPO_TOKEN_GRAD=1`)
- all shared hyperparameters (lr=1e-6 / KL 0.001 / ppo_epochs=1 / 600 steps / rollout 32×8)
  are identical across all arms and seeds; the only variables are the seed and the algorithm

SGPO uses relative token confidence to reshape advantages: higher-confidence tokens receive larger positive advantages in positive-advantage sequences and more negative advantages in negative-advantage sequences.

## Results (four seeds, step599, val 4532)

All three metrics were recomputed from the saved checkpoint-599 rank records, with all 4,532 validation examples in the denominator; missing results count as zero. Every arm uses the same code snapshot and the same hyperparameters. **Bold** marks the best method for a given metric and cutoff within a seed.

### NDCG@K

| seed | arm | @1 | @2 | @3 | @5 | @10 | @20 | @30 | @50 | @100 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| s42 | GRPO | 0.04568 | 0.06711 | 0.07649 | 0.08480 | 0.09471 | 0.10167 | 0.10604 | 0.11031 | 0.11656 |
| s42 | GSPO | 0.04634 | 0.06903 | 0.07775 | 0.08476 | 0.09575 | 0.10217 | 0.10593 | 0.11013 | **0.11719** |
| s42 | SGPO | **0.04898** | **0.07070** | **0.07854** | **0.08673** | **0.09661** | **0.10271** | **0.10616** | **0.11058** | 0.11689 |
| s43 | GRPO | 0.05693 | 0.07433 | 0.08349 | **0.09098** | 0.09979 | 0.10701 | 0.10977 | 0.11406 | 0.12040 |
| s43 | GSPO | 0.04214 | 0.06247 | 0.07218 | 0.08060 | 0.09090 | 0.09839 | 0.10206 | 0.10655 | 0.11374 |
| s43 | SGPO | **0.05825** | **0.07593** | **0.08377** | 0.09081 | **0.10085** | **0.10711** | **0.11079** | **0.11512** | **0.12177** |
| s1234 | GRPO | 0.04700 | 0.06955 | 0.07672 | 0.08430 | 0.09310 | 0.10086 | 0.10429 | 0.10868 | 0.11609 |
| s1234 | GSPO | 0.04104 | 0.06234 | 0.07260 | 0.08073 | 0.09018 | 0.09851 | 0.10267 | 0.10727 | 0.11421 |
| s1234 | SGPO | **0.04987** | **0.07145** | **0.07884** | **0.08668** | **0.09595** | **0.10336** | **0.10617** | **0.11077** | **0.11663** |
| s12345 | GRPO | 0.04523 | 0.06807 | 0.07733 | 0.08467 | 0.09538 | 0.10237 | 0.10588 | 0.11057 | 0.11734 |
| s12345 | GSPO | 0.04259 | 0.06444 | 0.07459 | 0.08236 | 0.09200 | 0.09971 | 0.10310 | 0.10787 | 0.11387 |
| s12345 | SGPO | **0.05406** | **0.07313** | **0.08152** | **0.08805** | **0.09792** | **0.10445** | **0.10824** | **0.11188** | **0.11774** |

### MRR@K

| seed | arm | @1 | @2 | @3 | @5 | @10 | @20 | @30 | @50 | @100 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| s42 | GRPO | 0.04568 | 0.06267 | 0.06892 | 0.07350 | 0.07754 | 0.07950 | 0.08033 | 0.08092 | 0.08145 |
| s42 | GSPO | 0.04634 | 0.06432 | 0.07013 | 0.07404 | 0.07853 | 0.08032 | 0.08103 | 0.08160 | 0.08221 |
| s42 | SGPO | **0.04898** | **0.06620** | **0.07142** | **0.07597** | **0.08007** | **0.08176** | **0.08242** | **0.08302** | **0.08356** |
| s43 | GRPO | 0.05693 | 0.07072 | 0.07682 | 0.08095 | 0.08452 | 0.08651 | 0.08703 | 0.08761 | 0.08816 |
| s43 | GSPO | 0.04214 | 0.05825 | 0.06472 | 0.06938 | 0.07360 | 0.07570 | 0.07639 | 0.07700 | 0.07763 |
| s43 | SGPO | **0.05825** | **0.07226** | **0.07749** | **0.08137** | **0.08546** | **0.08721** | **0.08791** | **0.08850** | **0.08907** |
| s1234 | GRPO | 0.04700 | 0.06487 | 0.06965 | 0.07388 | 0.07748 | 0.07962 | 0.08027 | 0.08087 | 0.08151 |
| s1234 | GSPO | 0.04104 | 0.05792 | 0.06476 | 0.06924 | 0.07312 | 0.07548 | 0.07625 | 0.07688 | 0.07747 |
| s1234 | SGPO | **0.04987** | **0.06697** | **0.07190** | **0.07627** | **0.08009** | **0.08214** | **0.08267** | **0.08330** | **0.08380** |
| s12345 | GRPO | 0.04523 | 0.06333 | 0.06951 | 0.07357 | 0.07787 | 0.07982 | 0.08048 | 0.08111 | 0.08169 |
| s12345 | GSPO | 0.04259 | 0.05991 | 0.06667 | 0.07101 | 0.07495 | 0.07713 | 0.07777 | 0.07843 | 0.07894 |
| s12345 | SGPO | **0.05406** | **0.06917** | **0.07476** | **0.07838** | **0.08239** | **0.08421** | **0.08492** | **0.08542** | **0.08593** |

### HitRate@K

| seed | arm | @1 | @2 | @3 | @5 | @10 | @20 | @30 | @50 | @100 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| s42 | GRPO | 0.04568 | 0.07966 | 0.09841 | 0.11871 | 0.14960 | **0.17674** | **0.19726** | **0.21999** | 0.25883 |
| s42 | GSPO | 0.04634 | 0.08230 | **0.09974** | 0.11673 | **0.15093** | 0.17608 | 0.19373 | 0.21624 | **0.25993** |
| s42 | SGPO | **0.04898** | **0.08341** | 0.09907 | **0.11893** | 0.14938 | 0.17343 | 0.18954 | 0.21315 | 0.25221 |
| s43 | GRPO | 0.05693 | 0.08451 | **0.10282** | **0.12114** | 0.14872 | **0.17718** | 0.19020 | 0.21315 | 0.25243 |
| s43 | GSPO | 0.04214 | 0.07436 | 0.09378 | 0.11430 | 0.14629 | 0.17564 | **0.19285** | **0.21690** | **0.26125** |
| s43 | SGPO | **0.05825** | **0.08628** | 0.10194 | 0.11915 | **0.15049** | 0.17498 | 0.19219 | 0.21536 | 0.25662 |
| s1234 | GRPO | 0.04700 | 0.08274 | 0.09709 | 0.11540 | 0.14276 | 0.17343 | 0.18954 | 0.21293 | 0.25883 |
| s1234 | GSPO | 0.04104 | 0.07480 | 0.09532 | 0.11518 | 0.14453 | **0.17696** | **0.19660** | **0.22109** | **0.26434** |
| s1234 | SGPO | **0.04987** | **0.08407** | **0.09885** | **0.11783** | **0.14651** | 0.17564 | 0.18888 | 0.21337 | 0.24978 |
| s12345 | GRPO | 0.04523 | 0.08142 | 0.09996 | **0.11783** | **0.15159** | **0.17895** | **0.19550** | **0.22065** | **0.26258** |
| s12345 | GSPO | 0.04259 | 0.07723 | 0.09753 | 0.11628 | 0.14629 | 0.17630 | 0.19219 | 0.21756 | 0.25485 |
| s12345 | SGPO | **0.05406** | **0.08429** | **0.10106** | 0.11695 | 0.14784 | 0.17343 | 0.19131 | 0.21072 | 0.24691 |

### Mean delta over four seeds (%, SGPO vs GRPO)

| metric | @1 | @2 | @3 | @5 | @10 | @20 | @30 | @50 | @100 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NDCG | +8.80 | +4.42 | +2.79 | +2.23 | +2.20 | +1.41 | +1.27 | +1.07 | +0.55 |
| MRR | +8.80 | +5.07 | +3.82 | +3.42 | +3.39 | +3.08 | +3.03 | +2.98 | +2.90 |
| HitRate | +8.80 | +2.98 | +0.68 | -0.02 | +0.30 | -1.23 | -1.34 | -1.59 | -2.59 |

Each row is the mean of the four per-seed relative deltas. Averaged over four seeds, SGPO improves NDCG and MRR over GRPO at all nine cutoffs. HitRate is higher at K=1/2/3/10, marginally lower at K=5, and lower from K=20 onward.

The small-K gains and large-K HitRate decreases are consistent with a trade-off between front-of-ranking quality and deeper retrieval coverage for this confidence-guided method.

## Reproduction

### Environment

ROLL **v0.3.0**, commit `7f9d4d3873a87679c8d070085a4bd3bdeea7e589`. Run from the repository root:

```bash
export RELEASE_ROOT="$PWD"
export ROLL_ROOT="$HOME/roll-qgen-v030"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch v0.3.0 https://github.com/alibaba/ROLL.git "$ROLL_ROOT"
python3 tools/prepare_roll.py --roll-root "$ROLL_ROOT" --apply
```

Training environment: **Linux x86_64 / Python 3.10 / CUDA**.

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
python -m pip install -r requirements.txt --no-build-isolation
python -m pip install --no-deps -e "$ROLL_ROOT/mcore_adapter" -e "$ROLL_ROOT" -e "$RELEASE_ROOT"
python -m pip check
python tools/prepare_roll.py --runtime --report environment.json
```

Dependencies: PyTorch 2.8.0, vLLM 0.10.2, Transformers 4.57.0, Megatron-Core 0.15.0, and Ray 2.48.0. FlashAttention and Transformer Engine require CUDA headers and a compiler.

Hardware: **2× NVIDIA A100-80GB GPUs**, DP=2, TP=1. Megatron-Core training and vLLM rollout generation share the two GPUs. Micro batch 4 × gradient accumulation 32 × DP 2 = 256 rows per optimization step (32 prompts × 8 samples).

### Data

The `data/` directory already contains all data used in the experiments:

| File | Description |
|---|---|
| `data/corpus.jsonl` | BM25 corpus (3,686 products) |
| `data/rl_train.jsonl` | RL training set |
| `data/rl_val.jsonl` | validation set (4,532 examples; the main-table protocol) |
| `data/rl_test.jsonl` | test set |

Splits: 36,259 train / 4,532 validation / 4,533 test. The JSONL prompts are already chat-templated.

Data construction tools: `build_corpus.py` takes item JSON; `export_rl_jsonl.py` takes sequence CSV files.

### Training

Download the base model, set the reward parameters, and run each method separately:

```bash
python tools/download_base_model.py --output "$HOME/qwen25-3b-qgen-base"
export QGEN_PRETRAIN="$HOME/qwen25-3b-qgen-base"
export QGEN_CORPUS="$RELEASE_ROOT/data/corpus.jsonl"
export QGEN_TOP_K=100 QGEN_MATCH_MODE=or_all
export CUDA_VISIBLE_DEVICES=0,1

# Run each arm separately, from the release root.
SGPO_ENABLE=0 python examples/start_rl_qgen.py --config_name qgen_grpo_3b_2gpu
SGPO_ENABLE=0 python examples/start_rl_qgen.py --config_name qgen_gspo_3b_2gpu_s42
SGPO_ENABLE=1 SGPO_SIGNAL=confidence SGPO_C_MIN=1 SGPO_C_MAX=4 SGPO_ANNEAL_STEPS=200 \
    SGPO_TOKEN_RATIO=0 SGPO_TOKEN_GRAD=1 \
    python examples/start_rl_qgen.py --config_name qgen_sgpo_tg_c4_a200_3b_2gpu
```

Checkpoints: `checkpoints/<exp_name>/checkpoint-599/`. Add `--dry_run` for a configuration check.

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python examples/start_eval_qgen.py \
    --model_name ./checkpoints/qgen-2gpu-sgpo-tg-c4-a200/checkpoint-599 \
    --ckpt_ids checkpoint-599 \
    --data_file ./data/rl_val.jsonl --corpus ./data/corpus.jsonl \
    --out_dir ./eval_out_sgpo --limit 0
```

Set `--model_name` to a local HF checkpoint directory and use a separate `--out_dir` for each method.

Evaluation uses vLLM greedy decoding (temperature=0, n=1, max_new_tokens=256) on one A100-80GB. Metrics are averaged over all 4,532 validation examples at K=1/2/3/5/10/20/30/50/100, with zero for misses. HitRate@K is `count(0 < rank <= K) / n`, stored as `hit@K`. Training and evaluation share the BM25 scorer.

### Tests

```bash
python -m pip install -r requirements-cpu.txt
python tools/verify_release.py
ROLL_ROOT="$ROLL_ROOT" python tests/utils/test_sgpo.py
python examples/recr1-qgen-amazon/tools/selfcheck_qgen.py --tokenizer "$QGEN_PRETRAIN"
```

## Build

```bash
python -m pip install build==1.2.2.post1
python -m build --sdist --outdir ../qgen_dist
```

## Repository Layout

```
qgen_sgpo/            SGPO, BM25 scorer, and ROLL reward-worker extension
patches/              two-file patch + fixed-commit/file-hash manifest
tools/                framework preparation, base-model download, integrity checks
examples/
  start_rl_qgen.py    training entry point
  start_eval_qgen.py  evaluation entry point
  recr1-qgen-amazon/  three-arm hydra configs + data construction tools
tests/utils/          SGPO CPU regression tests
data/                 experiment data (corpus + train/validation/test sets)
```

## Acknowledgments

This implementation builds on the open-source [ROLL](https://github.com/alibaba/ROLL) reinforcement learning training framework;
the task and reward protocols follow Rec-R1.

## File and checkpoint map

```text
qgen_sgpo_release/
├── README.md
├── qgen_sgpo/                       # objectives and BM25 reward workers
├── examples/
│   ├── start_rl_qgen.py             # RL launch
│   ├── start_eval_qgen.py           # checkpoint evaluation
│   └── recr1-qgen-amazon/*.yaml     # GRPO/GSPO/SGPO configs
├── data/                            # corpus.jsonl,rl_train/val/test.jsonl
├── patches/                         # ROLL integration patch
├── tools/                           # model download,setup,release checks
├── requirements*.txt                # CPU/GPU dependencies
├── tests/                           # objective checks
└── checkpoints/<exp_name>/          # created by training
    └── checkpoint-599/              # final model directory for evaluation
```

The base model is loaded from `QGEN_PRETRAIN`. Training checkpoint roots are set by `checkpoint_config.output_dir` in the YAML configuration, defaulting to `./checkpoints/${exp_name}`. With the included2GPU configs, experiment directories are `qgen-2gpu-grpo`, `qgen-2gpu-gspo-s42`,and `qgen-2gpu-sgpo-tg-c4-a200`. The final model is evaluated by passing its full `checkpoint-599/` directory to `start_eval_qgen.py --model_name`.
