# SGPO on the ReSID GR backbone: Office

Office_Products uses the same parallel6×16 SID Transformer encoder–decoder and RL protocol as the Industrial experiment. The release includes all data/SIDs, one supervised starting checkpoint, and 15 actual 600-step RL checkpoints for GRPO/GSPO/SGPO, seeds42–46.

## Reproduce evaluation and statistics

```bash
cd resid_sgpo_release/office
python -m pip install -r requirements.txt
python verify.py
python evaluate.py --all --split valid --output runs/evaluation
python statistics.py --evaluated runs/evaluation/metrics.json --output runs/statistics
```

Evaluate the actual600-step checkpoints on all 4,866 validation requests, beam 20. The evaluator checks HR/NDCG/MRR@5/10/20 against `references/expected_metrics.json`. Full metrics include exact/partial matching and all/head/tail groups at1/5/10/20/50/100/200.

## Reproduce training

```bash
python train.py --method grpo --seed 42 --output runs/grpo-seed42
python train.py --method gspo --seed 42 --output runs/gspo-seed42
python train.py --method sgpo --seed 42 --output runs/sgpo-seed42
python run_all.py --output runs/all-seeds
```

The five-seed comparison uses actual step-600 weights. Each run also retains every 100-step checkpoint and validation metrics. Training checkpoint selection uses mean VALID all/tail Recall@50/100/200,with selected TEST evaluation stored separately.

## Configuration

38,924 training / 4,866 validation / 4,866 test requests; 3,459 items. Shared fixed Office base weights and item SID assignment. Batch size 32, 8 beam rollouts, 600 updates, learning rate 1e-5, 20-step warmup and cosine decay, AdamW weight decay 0, clip 0.1, KL weight 0.001, gradient clip 0.5, one update per fresh group. Reward is fraction of 6 correct SID positions. Dropout disabled for rollout and optimization. Exact categorical KL(current||initial) on visited prefixes. SGPO uses c=4→1 over 200 steps with sequence-forward/token-gradient routing. RL seed changes only random state and TRAIN order; base model/SIDs remain fixed.

Metrics and statistical protocol match Industrial: single-target HR=Recall; NDCG andMRR at5/10/20; five-seed mean±sample std; paired two-sided t-test best vs runner-up,n=5, df=4, unadjusted p<0.05. The 4,866 requests is each metric denominator; seed is paired repeat unit. TRAIN groups tail1–6,head≥12,middle7–11,zero separate.

`references/results.md` contains the complete per-seed table,mean/std,and t/p values. `references/*-ranks.npz` stores each request's ranks. `python statistics.py --output runs/reference-statistics` recomputes the table from saved ranks.

## Directory and checkpoint map

```text
./
├── train.py / run_all.py            # RL training entrypoints
├── evaluate.py / statistics.py      # inference metrics and five-seed tests
├── verify.py / manifest.json        # file and reference checks
├── runtime.py / config.json         # loading and hyperparameters
├── resid_gr/                       # encoder–decoder,beam search,matching
├── data/                           # fixed SID map and train/valid/test CSV
├── checkpoints/
│   ├── index.json                  # checkpoint metadata and hashes
│   ├── base/gr.pt                  # pretrained supervised backbone
│   └── seed42/…seed46/             # grpo.pt,gspo.pt,sgpo.pt in each
├── references/                     # results.md,expected metrics,request ranks
├── tests/                          # objective checks
└── runs/                           # new outputs created by the run commands
```

Shipped models: `checkpoints/seed42/grpo.pt`, `gspo.pt`, `sgpo.pt`,and the same files under seeds43–46. Each is an actual600-step checkpoint. The supervised initialization is `checkpoints/base/gr.pt`.

Example fresh run: `python train.py --method sgpo --seed 42 --output runs/sgpo-seed42`. Its checkpoints are `runs/sgpo-seed42/sgpo-step100.pt` through `sgpo-step600.pt`; `sgpo-best.pt` is the separately selected model. Validation history is `valid-checkpoints.json`, and the training log is `sgpo.jsonl` in that same output directory.
