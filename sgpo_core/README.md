# SGPO: Sequence-level Group Policy Optimization for SID-based Generative Recommendation

This repository releases the RL post-training objectives used in our work, implemented in
[`sgpo/rl.py`](sgpo/rl.py). It contains a single entry point, `compute_grpo_loss`, which
implements **SGPO** and a family of group-based policy optimization baselines behind
mutually exclusive flags. All objectives share the same rollout format, reward, and
group-normalized advantage, and differ only in how the importance ratio and the
advantage are shaped.


## Method overview

For each user, the policy produces a group of `K = grpo_topk` beam-search rollouts, each
a semantic-ID (SID) token sequence of length `L = token_level`. Row 0 of the group is the
ground-truth sequence; when `use_old >= 1`, it is used only as the reward reference and
the remaining `K-1` rollouts form the training group.

- **Reward** (`dpo_reward_func`): number of token positions of a rollout that match the
  ground-truth SID sequence, optionally thresholded by `dpo_reward_mode`.
- **Advantage**: group-normalized (GRPO-style), `A = (r - mean_group) / (std_group + eps)`.
- **Ratio**: `ratio = pi_new / pi_old` per token, where `pi_old` are the behavior-policy
  probabilities recorded at rollout time and `pi_new` come from re-scoring the rollout
  sequences with the current policy (teacher forcing via the decoder).
- **Optional KL** regularizer against the old policy (`use_kl`, weight `beta`), and PPO
  clipping with `epsilon` where applicable.

## RL objective branches

Exactly one of the following flags should be enabled (all off falls back to GRPO):

| Flag | Objective | Ratio granularity | Advantage shaping |
|---|---|---|---|
| *(none, default)* | **GRPO** | token-level | sequence-level advantage broadcast to tokens, PPO clip |
| `use_gspo` | **GSPO** | sequence-level (geometric mean of token ratios) | sequence-level advantage, PPO clip |
| `use_sapo` | **SAPO** | token-level | soft sigmoid gate `sigmoid(tau * (ratio - 1))` instead of hard clipping, asymmetric temperatures `tau_pos` / `tau_neg` for positive / negative advantages |
| `use_gtpo` | **GTPO** | token-level | token entropy-based reward redistribution: high-entropy tokens in correct rollouts get bonus, low-entropy tokens in wrong rollouts get stronger penalty |
| `use_grpo_s` | **GRPO-S** | token-level | sequence entropy-based reward reshaping (per-sequence mean entropy variant of GTPO) |
| `use_aepo` | **AEPO** | token-level | advantage modulated by standardized token entropy: `A_t = A * (1 + a * ΔH_t)` |
| `use_sgpo` | **SGPO (ours)** | sequence-level | token-level advantage reshaping `A_t = A_s ± λ · u_t`, where `u_t` is the centered token log-prob and `λ` is scaled by an annealed coefficient `c(n)` from `sgpo_c_max` to `sgpo_c_min` over `sgpo_anneal_steps` |

SGPO additionally supports `sgpo_token_grad`:

- `sgpo_token_grad = 1` (default): the sequence-level ratio keeps its forward value but
  gradients are routed to each token individually, yielding
  `grad ≈ s_i * Σ_t A_t * ∇ log π(y_t)`.
- `sgpo_token_grad = 0` (legacy): the sequence ratio enters as a plain scalar; the
  reshaping acts only through the clipping gate.

## Usage

`compute_grpo_loss(samples, user_encoder_out_last, item_seq_mask)` expects:

- `samples['predict_token_list']` — rollout SID tokens, reshaped to `[B, K, L]`
  (row 0 = ground truth).
- `samples['predict_probs_list']` — old-policy per-token probabilities, same shape.
- `samples['predict_valid_item_id']`, `predict_hit_*` — auxiliary per-rollout ids used by
  reward variants, `[B, K]`.
- `user_encoder_out_last` `[B, seq_len, d]` and `item_seq_mask` `[B, seq_len, 1]` —
  encoder outputs, repeated per rollout for teacher-forced re-scoring.

Host-class requirements (attributes / methods referenced via `self`):

- Config: `grpo_topk`, `token_level`, `seq_len`, `input_feature_size`, `epsilon`, `beta`,
  `use_kl`, `use_old`, `dpo_weight`, `dpo_reward_mode`, one objective flag from the table,
  and for SGPO/SAPO the extra hyperparameters listed above.
- Decoder: `token_decoder_v2(...)` must teacher-force the rollout token sequences and
  populate `self.dpo_new_probs` (per-level probabilities of the taken tokens) and
  `self.dpo_full_probs` (full vocabulary distributions, used for entropy).

The returned scalar is `total_loss * dpo_weight`; add it to your training loss.

## Plugging into any SID-based generative retrieval / recommendation model

The objectives are model-agnostic: nothing in `rl.py` depends on a specific backbone.
Any SID-based generative recommender (e.g., TIGER- or REG4Rec-style encoder-decoder)
can adopt them by providing three things:

1. **Rollouts** — beam-search (or sample) `K` SID sequences per user from the current
   policy, and record their per-token probabilities as the old policy.
2. **Re-scoring** — a decoder call that teacher-forces those sequences under the current
   parameters and exposes per-token taken-probabilities and full distributions.
3. **Reward reference** — the ground-truth SID sequence of the target item.

Set `token_level` / vocabulary size to match your SID scheme (e.g., 4 levels × 512 codes
for a TIGER-style setting, 8 levels × 300 codes for a REG4Rec-style setting), and the
loss applies unchanged.

## File map

```text
sgpo_core/
├── README.md      # objective definitions and host-model interface
└── sgpo/rl.py     # reward and policy-optimization implementation
```

This module receives rollout tokens/probabilities and decoder outputs from its host model. Checkpoint creation is handled by that model's training entrypoint. ReSID checkpoint locations are documented in `../resid_sgpo_release/README.md`.
