import math
import os

__all__ = ['sgpo_enabled', 'compute_sgpo', 'sgpo_metrics_empty']


def sgpo_enabled():
    return os.environ.get('SGPO_ENABLE', '0') == '1'


def _sgpo_cfg():
    cfg = {
        "c_max": float(os.environ.get("SGPO_C_MAX", "2.0")),
        "c_min": float(os.environ.get("SGPO_C_MIN", "1.0")),
        "anneal_steps": int(os.environ.get("SGPO_ANNEAL_STEPS", "200")),
        "token_ratio": os.environ.get("SGPO_TOKEN_RATIO", "0") == "1",
        "token_grad": os.environ.get("SGPO_TOKEN_GRAD", "0") == "1",
        "signal": os.environ.get("SGPO_SIGNAL", "confidence"),
    }
    for name in ("SGPO_TOKEN_RATIO", "SGPO_TOKEN_GRAD"):
        if os.environ.get(name, "0") not in {"0", "1"}:
            raise ValueError(f"{name} 必须为 0 或 1")
    if not (math.isfinite(cfg["c_max"]) and math.isfinite(cfg["c_min"])
            and cfg["c_max"] >= cfg["c_min"] >= 1.0):
        raise ValueError("SGPO 要求有限的 c_max >= c_min >= 1")
    if cfg["anneal_steps"] <= 0:
        raise ValueError("SGPO_ANNEAL_STEPS 必须为正训练步数")
    if cfg["token_ratio"] and cfg["token_grad"]:
        raise ValueError("SGPO_TOKEN_GRAD=1 只用于 SGPO_TOKEN_RATIO=0")
    if cfg["signal"] != "confidence":
        raise ValueError("SGPO_SIGNAL 必须为 confidence")
    return cfg


def sgpo_metrics_empty():
    return {
        "sgpo/c_n": 0.0,
        "sgpo/global_step": 0.0,
        "sgpo/token_grad": 0.0,
        "sgpo/token_ratio": 0.0,
        "sgpo/lambda_k_mean": 0.0,
        "sgpo/lambda_k_max": 0.0,
        "sgpo/u_t_sq_mean": 0.0,
        "sgpo/A_t_pos_mean": 0.0,
        "sgpo/A_t_neg_mean": 0.0,
        "sgpo/advantage_shift": 0.0,
    }


def compute_sgpo(log_probs, old_log_probs, advantages, response_mask, global_step):
    import torch

    cfg = _sgpo_cfg()
    c_max = cfg["c_max"]
    c_min = cfg["c_min"]
    N = cfg["anneal_steps"]
    n = int(global_step)
    if n < 0 or n != global_step:
        raise ValueError("global_step 必须为非负整数训练步数")
    token_ratio = cfg["token_ratio"]

    c_n = max(c_min, c_max - (c_max - c_min) * n / N)

    log_ratio = torch.where(response_mask.bool(), log_probs - old_log_probs.detach(), 0.0)

    mask = response_mask.float()
    lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

    log_p_new = torch.where(response_mask.bool(), log_probs.detach(), 0.0)
    log_p_mean = (log_p_new * mask).sum(dim=-1, keepdim=True) / lengths
    u_t = (log_p_new - log_p_mean) * mask

    u_t_sq_mean_per_seq = (u_t ** 2).sum(dim=-1, keepdim=True) / lengths
    eps = 1e-8

    A_s = (advantages.detach() * mask).sum(dim=-1, keepdim=True) / lengths
    advantages_abs = A_s.abs()
    u_sq = u_t_sq_mean_per_seq
    lambda_k = torch.where(
        u_sq > 0,
        advantages_abs * torch.sqrt(max(0.0, c_n - 1.0) / (u_sq + eps)),
        torch.zeros_like(advantages_abs),
    )

    sign_A = torch.sign(A_s)
    A_t = (A_s + sign_A * lambda_k * u_t) * mask

    if token_ratio:
        ratio = torch.exp(log_ratio)
    else:
        masked_log_ratio = (log_ratio * mask).sum(dim=-1, keepdim=True) / lengths
        sequence_ratio = torch.exp(masked_log_ratio)
        if cfg["token_grad"]:
            ratio = sequence_ratio.detach() * torch.exp(log_ratio - log_ratio.detach())
        else:
            ratio = sequence_ratio.expand_as(log_ratio)
    ratio = ratio * mask + (1.0 - mask)

    pos_mask = (A_t * mask > 0).float()
    neg_mask = (A_t * mask < 0).float()
    pos_sum = pos_mask.sum().clamp(min=1.0)
    neg_sum = neg_mask.sum().clamp(min=1.0)
    A_t_pos_mean = (A_t * pos_mask).sum() / pos_sum
    A_t_neg_mean = (A_t * neg_mask).sum() / neg_sum

    metrics = {
        "sgpo/c_n": float(c_n),
        "sgpo/global_step": float(n),
        "sgpo/token_grad": float(cfg["token_grad"]),
        "sgpo/token_ratio": float(token_ratio),
        "sgpo/lambda_k_mean": lambda_k.mean().detach().item(),
        "sgpo/lambda_k_max": lambda_k.max().detach().item(),
        "sgpo/u_t_sq_mean": u_t_sq_mean_per_seq.mean().detach().item(),
        "sgpo/A_t_pos_mean": A_t_pos_mean.detach().item(),
        "sgpo/A_t_neg_mean": A_t_neg_mean.detach().item(),
        "sgpo/advantage_shift": ((A_t - advantages) * mask).abs().mean().detach().item(),
    }

    return A_t, ratio, metrics
