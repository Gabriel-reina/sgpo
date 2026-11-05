# SGPO on the ReSID GR backbone

The experiment uses a Transformer encoder–decoder with fixed parallel 6×16 SIDs on Amazon Industrial_and_Scientific. The release contains the backbone, fixed data/SIDs, its supervised checkpoint, and the final600-step GRPO/GSPO/SGPO weights for RL seeds42–46.

## Environment

```bash
cd resid_sgpo_release/industrial
python -m pip install -r requirements.txt
```

Reference runtime: Python3.9, PyTorch2.8.0, NumPy2.0.2, SciPy1.13.1, CPU FP32,4threads.

## Verify the release

```bash
python verify.py
python -m unittest discover -s tests
```

## Evaluate all fifteen checkpoints

```bash
python evaluate.py --all --split valid --output runs/evaluation
python statistics.py --evaluated runs/evaluation/metrics.json --output runs/statistics
```

The evaluator checks HR/NDCG/MRR@5/10/20 against the reference values and writes all@1/5/10/20/50/100/200 accuracy and coverage metrics for exact and partial matching. Every shipped RL checkpoint is the actual600-step checkpoint. The main table uses all4532 validation requests.

## Reproduce training

```bash
python train.py --method grpo --seed 42 --output runs/grpo-seed42
python train.py --method gspo --seed 42 --output runs/gspo-seed42
python train.py --method sgpo --seed 42 --output runs/sgpo-seed42
```

Run all fifteen combinations sequentially with `python run_all.py --output runs/training`. Repeat individual commands with seeds43,44,45,46. Both training and evaluation resolve inputs relative to this directory. Each run saves every100-step checkpoint and validation metrics. The supervised backbone weights and fixed SID assignment are shared across all methods/seeds.

## Training protocol

Each update uses32TRAIN requests and8catalog-constrained beam candidates per request. Reward is the fraction of the6SID positions agreeing with the training target. Group-standardized advantages drive token-ratio GRPO, sequence-ratio GSPO, or sequence-forward/token-gradient SGPO. A batch receives one optimizer update. The exact categorical KL(current||fixed initial policy) is averaged across visited prefixes. Dropout is disabled during both rollout and optimization. AdamW uses lr1e-5,warmup20,cosine decay over600steps,weight_decay0,clip0.1,KL0.001,gradient_clip0.5. SGPO anneals c4→1 over200steps. Beam evaluation uses20complete SIDs. Partial matching ranks by minimum Hamming distance, earliest supporting beam, then itemID.

Training checkpoint selection uses mean VALID all/tail Recall@50/100/200. The five-seed comparison evaluates the actual600-step checkpoints. Frequency groups use TRAIN targets: tail1–6,middle7–11,head≥12,zero separate. TEST is a separately available split.

## Metrics and statistics

Single-target HitRate equals Recall. NDCG@K averages1/log2(rank+1) for hits withinK; MRR@K averages1/rank. Effective coverage counts distinct correctly retrieved targets; exposure coverage counts distinct exposed catalog items. The five-seed table reports mean±sample standard deviation. Paired two-sided t-tests use5RL-seed differences,df4,p<0.05,with unadjusted p-values. Per-request ranks and reference metrics are included in `references/`. Recompute their statistics with:

```bash
python statistics.py --output runs/reference-statistics
```

## Files

- `resid_gr/`: backbone,data loader,beam search,matching and metrics.
- `checkpoints/base/gr.pt`: shared supervised starting point.
- `checkpoints/seed42/` through `seed46/`: GRPO,GSPO,SGPO final weights.
- `data/`:36,259TRAIN /4,532VALID /4,533TEST requests;3,686item SIDs.
- `references/`: saved ranks, numerical reference and experiment report.

The backbone and SID construction follow ReSID; its broader framework is described in the ReSID paper.

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
