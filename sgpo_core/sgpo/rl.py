import torch
import torch.nn.functional as F

def dpo_reward_func(self, predict_token_list, real_dpo_row_vec, predict_dpo_row_vec,
                        predict_valid_item_id, predict_hit_item_id, predict_hit_attr_a_id,
                        predict_hit_attr_b_id, predict_hit_attr_c_id):

        real_token = (predict_token_list * real_dpo_row_vec).sum(dim=1, keepdim=True)
        self.print_node("real_token:", real_token, 16)

        token_hit_num = predict_token_list - real_token  # [B, grpo_topk, token_level]
        zero_mask = (token_hit_num == 0.0).float()
        zero_count = zero_mask.sum(dim=-1, keepdim=True)  # [B, grpo_topk, 1]
        self.print_node("hit_mask:", zero_mask, 16)
        self.print_node("hit_num:", zero_count, 16)

        if self.use_old >= 1:
            zero_count = zero_count * predict_dpo_row_vec

        neg_thresholds = {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0}
        threshold = neg_thresholds.get(self.dpo_reward_mode, 4.0)

        neg_mask = (zero_count < threshold).float()
        neg_reward = zero_count * neg_mask

        if self.use_old >= 1:
            return zero_count[:, 1:self.grpo_topk, :]  # [B, grpo_topk-1, 1]
        else:
            return zero_count

def compute_grpo_loss(self, samples, user_encoder_out_last, item_seq_mask):
    predict_token_list = samples['predict_token_list'].view(-1, self.grpo_topk, self.token_level)
    predict_probs_list = samples['predict_probs_list'].view(-1, self.grpo_topk, self.token_level)

    predict_valid_item_id = samples['predict_valid_item_id'].view(-1, self.grpo_topk)
    predict_hit_item_id = samples['predict_hit_item_id'].view(-1, self.grpo_topk)
    predict_hit_attr_a_id = samples['predict_hit_attr_a_id'].view(-1, self.grpo_topk)
    predict_hit_attr_b_id = samples['predict_hit_attr_b_id'].view(-1, self.grpo_topk)
    predict_hit_attr_c_id = samples['predict_hit_attr_c_id'].view(-1, self.grpo_topk)

    # print(f"samples['predict_token_list'].full_shape: {samples['predict_token_list'].shape}")

    device = predict_token_list.device
    real_dpo_row_vec = torch.zeros(1, self.grpo_topk, 1, device=device)
    real_dpo_row_vec[0, 0, 0] = 1.0
    predict_dpo_row_vec = 1.0 - real_dpo_row_vec

    self.print_node("dpo_row_vec:", real_dpo_row_vec)
    self.print_node("predict_token_list:", predict_token_list, 10)
    self.print_node("predict_probs_list:", predict_probs_list, 64)

    grpo_reward_selected = self.dpo_reward_func(
        predict_token_list, real_dpo_row_vec, predict_dpo_row_vec,
        predict_valid_item_id, predict_hit_item_id, predict_hit_attr_a_id,
        predict_hit_attr_b_id, predict_hit_attr_c_id
    )  # [B, K, 1]

    self.print_node("grpo_reward_selected:", grpo_reward_selected)

    if self.use_old >= 1:
        actual_topk = self.grpo_topk - 1
        predict_token_list = predict_token_list[:, 1:, :]
        old_probs = predict_probs_list[:, 1:, :].reshape(-1, self.token_level)
    else:
        actual_topk = self.grpo_topk
        old_probs = predict_probs_list.reshape(-1, self.token_level)

    # group-normalized advantage within each rollout group (Eq. 2)
    mean_reward = grpo_reward_selected.mean(dim=1, keepdim=True)
    std_reward = grpo_reward_selected.std(dim=1, keepdim=True) + 1e-4
    grpo_advantages = ((grpo_reward_selected - mean_reward) / std_reward).detach()
    self.print_node("mean_grpo_reward:", mean_reward, 1)
    self.print_node("std_grpo_reward:", std_reward, 1)
    self.print_node("grpo_advantages:", grpo_advantages)
    # print("advantages range:", grpo_advantages.min().item(), grpo_advantages.max().item())

    batch_size = user_encoder_out_last.size(0)
    user_encoder_out_last_dpo = user_encoder_out_last.unsqueeze(1).repeat(
        1, actual_topk, 1, 1).view(-1, self.seq_len, self.input_feature_size)
    item_seq_mask_dpo = item_seq_mask.unsqueeze(1).repeat(
        1, actual_topk, 1, 1).view(-1, self.seq_len, 1)

    self.print_node("user_encoder_out_last_dpo:", user_encoder_out_last_dpo, 1)
    self.print_node("item_seq_mask_dpo:", item_seq_mask_dpo, 1)

    predict_level_target = [
        predict_token_list[:, :, i].reshape(-1, 1)
        for i in range(self.token_level)
    ]
    # print("compute_grpo_loss")
    # print(f"batch_size (original): {samples['predict_token_list'].size(0)}")
    # print(f"user_encoder_out_last_dpo.shape: {user_encoder_out_last_dpo.shape}")

    token_level_ids_grpo, token_level_probs_grpo = self.token_decoder_v2(
        user_encoder_out_last_dpo, item_seq_mask_dpo,
        is_consist=2, is_train=True,
        token_level_target=predict_level_target,
        level_idx=[0] * self.token_level,
        samples=samples
    )

    # assert len(self.dpo_new_probs) == self.token_level, f"dpo_new_probs len={len(self.dpo_new_probs)} != token_level={self.token_level}"

    dpo_new_probs = self.dpo_new_probs[0]
    for i in range(1, self.token_level):
        dpo_new_probs = torch.cat([dpo_new_probs, self.dpo_new_probs[i]], dim=-1)
    # dpo_new_probs: [B*actual_topk, L]

    # average log prob of correct/incorrect tokens (token-level, new policy)
    all_token_list = samples['predict_token_list'].view(-1, self.grpo_topk, self.token_level)  # [B, grpo_topk, L]
    gt_tokens = all_token_list[:, 0:1, :]  # [B, 1, L]
    neg_tokens = all_token_list[:, 1:, :]  # [B, actual_topk, L]

    new_probs_reshaped = dpo_new_probs.view(-1, actual_topk, self.token_level)  # [B, actual_topk, L]
    hit_mask = (neg_tokens == gt_tokens).float()  # [B, actual_topk, L]
    miss_mask = 1.0 - hit_mask

    # advantage sign masks, shape [B, actual_topk, L]
    adv_pos_mask = (grpo_advantages.view(-1, actual_topk, 1).expand(-1, -1, self.token_level) > 0).float()
    adv_neg_mask = (grpo_advantages.view(-1, actual_topk, 1).expand(-1, -1, self.token_level) <= 0).float()

    log_new_probs = new_probs_reshaped.log()

    # overall statistics
    hit_log_prob = (log_new_probs * hit_mask).sum() / hit_mask.sum().clamp(min=1.0)
    miss_log_prob = (log_new_probs * miss_mask).sum() / miss_mask.sum().clamp(min=1.0)
    self.metrics_to_log['grpo_hit_token_log_prob'] = hit_log_prob.item()
    self.metrics_to_log['grpo_miss_token_log_prob'] = miss_log_prob.item()
    self.metrics_to_log['grpo_hit_miss_log_prob_gap'] = (hit_log_prob - miss_log_prob).item()
    self.dprint("grpo_hit_token_log_prob: ", hit_log_prob.item())
    self.dprint("grpo_miss_token_log_prob: ", miss_log_prob.item())
    self.dprint("grpo_hit_miss_log_prob_gap: ", (hit_log_prob - miss_log_prob).item())

    # statistics for advantage > 0
    hit_pos = hit_mask * adv_pos_mask
    miss_pos = miss_mask * adv_pos_mask
    if hit_pos.sum() > 0:
        hit_log_prob_pos = (log_new_probs * hit_pos).sum() / hit_pos.sum().clamp(min=1.0)
        miss_log_prob_pos = (log_new_probs * miss_pos).sum() / miss_pos.sum().clamp(min=1.0)
        self.metrics_to_log['grpo_hit_token_log_prob_adv_pos'] = hit_log_prob_pos.item()
        self.metrics_to_log['grpo_miss_token_log_prob_adv_pos'] = miss_log_prob_pos.item()
        self.metrics_to_log['grpo_hit_miss_log_prob_gap_adv_pos'] = (hit_log_prob_pos - miss_log_prob_pos).item()
        self.dprint("grpo_hit_token_log_prob_adv_pos: ", hit_log_prob_pos.item())
        self.dprint("grpo_miss_token_log_prob_adv_pos: ", miss_log_prob_pos.item())
        self.dprint("grpo_hit_miss_log_prob_gap_adv_pos: ", (hit_log_prob_pos - miss_log_prob_pos).item())

    # statistics for advantage <= 0
    hit_neg = hit_mask * adv_neg_mask
    miss_neg = miss_mask * adv_neg_mask
    if hit_neg.sum() > 0:
        hit_log_prob_neg = (log_new_probs * hit_neg).sum() / hit_neg.sum().clamp(min=1.0)
        miss_log_prob_neg = (log_new_probs * miss_neg).sum() / miss_neg.sum().clamp(min=1.0)
        self.metrics_to_log['grpo_hit_token_log_prob_adv_neg'] = hit_log_prob_neg.item()
        self.metrics_to_log['grpo_miss_token_log_prob_adv_neg'] = miss_log_prob_neg.item()
        self.metrics_to_log['grpo_hit_miss_log_prob_gap_adv_neg'] = (hit_log_prob_neg - miss_log_prob_neg).item()
        self.dprint("grpo_hit_token_log_prob_adv_neg: ", hit_log_prob_neg.item())
        self.dprint("grpo_miss_token_log_prob_adv_neg: ", miss_log_prob_neg.item())
        self.dprint("grpo_hit_miss_log_prob_gap_adv_neg: ", (hit_log_prob_neg - miss_log_prob_neg).item())

    ratio = (dpo_new_probs / (old_probs + 1e-5)).clamp(min=1e-8)
    self.print_node("dpo_new_probs", dpo_new_probs, 64)
    self.print_node("old_probs", old_probs, 64)
    self.print_node("ratio_new/old", ratio, 64)

    # 3. per-level hit/miss log prob statistics (overall / positive / negative)
    # for level_idx in range(self.token_level):
    #     level_hit_mask = hit_mask[:, :, level_idx]    # [B, actual_topk]
    #     level_miss_mask = miss_mask[:, :, level_idx]  # [B, actual_topk]
    #     level_log_new_probs = log_new_probs[:, :, level_idx]  # [B, actual_topk]
    #     level_adv_2d = grpo_advantages.view(-1, actual_topk, 1).squeeze(-1)  # [B, actual_topk]
    #     level_adv_pos_mask = (level_adv_2d > 0).float()   # [B, actual_topk]
    #     level_adv_neg_mask = (level_adv_2d <= 0).float()  # [B, actual_topk]
    #
    #     # overall
    #     level_hit_lp = (level_log_new_probs * level_hit_mask).sum() / level_hit_mask.sum().clamp(min=1.0)
    #     level_miss_lp = (level_log_new_probs * level_miss_mask).sum() / level_miss_mask.sum().clamp(min=1.0)
    #     self.metrics_to_log[f'grpo_hit_log_prob_level_{level_idx + 1}'] = level_hit_lp.item()
    #     self.metrics_to_log[f'grpo_miss_log_prob_level_{level_idx + 1}'] = level_miss_lp.item()
    #     self.metrics_to_log[f'grpo_hit_miss_gap_level_{level_idx + 1}'] = (level_hit_lp - level_miss_lp).item()
    #
    #     # advantage > 0
    #     level_hit_pos = level_hit_mask * level_adv_pos_mask
    #     level_miss_pos = level_miss_mask * level_adv_pos_mask
    #     if level_hit_pos.sum() > 0:
    #         level_hit_lp_pos = (level_log_new_probs * level_hit_pos).sum() / level_hit_pos.sum().clamp(min=1.0)
    #         level_miss_lp_pos = (level_log_new_probs * level_miss_pos).sum() / level_miss_pos.sum().clamp(min=1.0)
    #         self.metrics_to_log[f'grpo_hit_log_prob_level_{level_idx + 1}_adv_pos'] = level_hit_lp_pos.item()
    #         self.metrics_to_log[f'grpo_miss_log_prob_level_{level_idx + 1}_adv_pos'] = level_miss_lp_pos.item()
    #         self.metrics_to_log[f'grpo_hit_miss_gap_level_{level_idx + 1}_adv_pos'] = (level_hit_lp_pos - level_miss_lp_pos).item()
    #
    #     # advantage <= 0
    #     level_hit_neg = level_hit_mask * level_adv_neg_mask
    #     level_miss_neg = level_miss_mask * level_adv_neg_mask
    #     if level_hit_neg.sum() > 0:
    #         level_hit_lp_neg = (level_log_new_probs * level_hit_neg).sum() / level_hit_neg.sum().clamp(min=1.0)
    #         level_miss_lp_neg = (level_log_new_probs * level_miss_neg).sum() / level_miss_neg.sum().clamp(min=1.0)
    #         self.metrics_to_log[f'grpo_hit_log_prob_level_{level_idx + 1}_adv_neg'] = level_hit_lp_neg.item()
    #         self.metrics_to_log[f'grpo_miss_log_prob_level_{level_idx + 1}_adv_neg'] = level_miss_lp_neg.item()
    #         self.metrics_to_log[f'grpo_hit_miss_gap_level_{level_idx + 1}_adv_neg'] = (level_hit_lp_neg - level_miss_lp_neg).item()

    full_probs_stacked = torch.stack(self.dpo_full_probs, dim=1)  # [B*K, L, V]
    token_entropy = -(full_probs_stacked * torch.clamp(full_probs_stacked, 1e-8).log()).sum(dim=-1).detach()  # [B*K, L]
    avg_token_entropy = token_entropy.mean().item()

    # clip fraction
    eps = self.epsilon
    ratio_vec = ratio.view(-1)
    clipped = ((ratio_vec - 1.0).abs() > eps).float()
    clip_frac = clipped.mean().item()

    # store metrics
    self.metrics_to_log['grpo_token_entropy'] = avg_token_entropy
    self.metrics_to_log['grpo_clip_frac'] = clip_frac

    self.dprint("grpo_token_entropy: ", avg_token_entropy)
    self.dprint("grpo_clip_frac: ", clip_frac)

    # ============ grouped statistics by advantage sign and level ============
    # prepare data
    ratio_reshaped = ratio.view(-1, actual_topk, self.token_level)  # [B, K, L]
    advantages_flat = grpo_advantages.view(-1, actual_topk, 1)  # [B, K, 1]
    advantages_broadcast = advantages_flat.expand(-1, -1, self.token_level).reshape(-1)  # [B*K*L]

    # group by advantage sign
    pos_mask = advantages_broadcast > 0
    neg_mask = advantages_broadcast <= 0

    # flatten token entropy and ratio
    token_entropy_flat = token_entropy.view(-1)  # [B*K*L]
    ratio_flat = ratio_reshaped.reshape(-1)  # [B*K*L]

    # 1. overall statistics for advantage > 0
    if pos_mask.sum() > 0:
        pos_token_entropy = token_entropy_flat[pos_mask].mean().item()
        pos_clipped = ((ratio_flat[pos_mask] - 1.0).abs() > eps).float()
        pos_clip_frac = pos_clipped.mean().item()
        self.metrics_to_log['grpo_token_entropy_adv_pos'] = pos_token_entropy
        self.metrics_to_log['grpo_clip_frac_adv_pos'] = pos_clip_frac

        self.dprint("grpo_token_entropy_adv_pos: ", pos_token_entropy)
        self.dprint("grpo_clip_frac_adv_pos: ", pos_clip_frac)

    # 2. overall statistics for advantage <= 0
    if neg_mask.sum() > 0:
        neg_token_entropy = token_entropy_flat[neg_mask].mean().item()
        neg_clipped = ((ratio_flat[neg_mask] - 1.0).abs() > eps).float()
        neg_clip_frac = neg_clipped.mean().item()
        self.metrics_to_log['grpo_token_entropy_adv_neg'] = neg_token_entropy
        self.metrics_to_log['grpo_clip_frac_adv_neg'] = neg_clip_frac

        self.dprint("grpo_token_entropy_adv_neg: ", neg_token_entropy)
        self.dprint("grpo_clip_frac_adv_neg: ", neg_clip_frac)

    # # 3. per-level statistics (overall / positive / negative)
    # for level_idx in range(self.token_level):
    #     # extract data for this level
    #     level_entropy = token_entropy[:, level_idx]  # [B*K]
    #     level_ratio = ratio_reshaped[:, :, level_idx].reshape(-1)  # [B*K]
    #     level_adv = advantages_flat.squeeze(-1).reshape(-1)  # [B*K]
    #
    #     # overall statistics for this level
    #     level_entropy_mean = level_entropy.mean().item()
    #     level_clipped = ((level_ratio - 1.0).abs() > eps).float()
    #     level_clip_frac = level_clipped.mean().item()
    #     self.metrics_to_log[f'grpo_token_entropy_level_{level_idx + 1}'] = level_entropy_mean
    #     self.metrics_to_log[f'grpo_clip_frac_level_{level_idx + 1}'] = level_clip_frac
    #
    #     # positive-sample statistics for this level
    #     level_pos_mask = level_adv > 0
    #     if level_pos_mask.sum() > 0:
    #         level_pos_entropy = level_entropy.reshape(-1)[level_pos_mask].mean().item()
    #         level_pos_clipped = ((level_ratio[level_pos_mask] - 1.0).abs() > eps).float()
    #         level_pos_clip = level_pos_clipped.mean().item()
    #         self.metrics_to_log[f'grpo_token_entropy_level_{level_idx + 1}_adv_pos'] = level_pos_entropy
    #         self.metrics_to_log[f'grpo_clip_frac_level_{level_idx + 1}_adv_pos'] = level_pos_clip
    #
    #     # negative-sample statistics for this level
    #     level_neg_mask = level_adv <= 0
    #     if level_neg_mask.sum() > 0:
    #         level_neg_entropy = level_entropy.reshape(-1)[level_neg_mask].mean().item()
    #         level_neg_clipped = ((level_ratio[level_neg_mask] - 1.0).abs() > eps).float()
    #         level_neg_clip = level_neg_clipped.mean().item()
    #         self.metrics_to_log[f'grpo_token_entropy_level_{level_idx + 1}_adv_neg'] = level_neg_entropy
    #         self.metrics_to_log[f'grpo_clip_frac_level_{level_idx + 1}_adv_neg'] = level_neg_clip

    if self.use_gspo:
        log_ratio = ratio.log()
        log_ratio_mean = log_ratio.view(-1, actual_topk, self.token_level).mean(dim=-1)
        ratio_mean = log_ratio_mean.exp().unsqueeze(-1)
        # self.print_node("ratio_mean", ratio_mean, 64)

        # log max/min of log_ratio - log_ratio_mean
        log_ratio_reshaped = log_ratio.view(-1, actual_topk, self.token_level)
        log_ratio_diff = log_ratio_reshaped - log_ratio_mean.unsqueeze(-1)
        self.print_node("log_ratio_diff_max", log_ratio_diff.max(), 1)
        self.print_node("log_ratio_diff_min", log_ratio_diff.min(), 1)

        # log max/min of log(old_probs) - mean(log(old_probs))
        log_old_probs = old_probs.log().view(-1, actual_topk, self.token_level)
        log_old_probs_mean = log_old_probs.mean(dim=-1, keepdim=True)
        log_old_probs_diff = log_old_probs - log_old_probs_mean
        self.print_node("log_old_probs_diff_max", log_old_probs_diff.max(), 1)
        self.print_node("log_old_probs_diff_min", log_old_probs_diff.min(), 1)

        ratio_inv = (old_probs / (dpo_new_probs + 1e-5)).clamp(min=1e-8)
        log_ratio_inv = ratio_inv.log()
        log_ratio_inv_mean = log_ratio_inv.view(-1, actual_topk, self.token_level).mean(dim=-1)
        ratio_inv_mean = log_ratio_inv_mean.exp().unsqueeze(-1)

        advantages_flat = grpo_advantages.view(-1, actual_topk, 1)
        loss_unclipped = ratio_mean * advantages_flat
        loss_clipped = torch.clamp(ratio_mean, 1 - self.epsilon, 1 + self.epsilon) * advantages_flat
        ppo_loss = -torch.min(loss_unclipped, loss_clipped).view(-1, actual_topk)
        # self.print_node("advantages_flat", advantages_flat,64)
        # self.print_node("loss_unclipped", loss_unclipped,64)
        # self.print_node("loss_clipped",loss_clipped,64)
        # self.print_node("ppo_loss", ppo_loss,64)
        self.print_node("ppo_loss", ppo_loss)

        kl_divergence = torch.zeros_like(ppo_loss)
        if self.use_kl:
            kl_divergence = (ratio_inv_mean - ratio_inv_mean.log() - 1).view(-1, actual_topk) * self.beta
            self.print_node("kl_div", kl_divergence)
            self.metrics_to_log['kl_divergence_mean'] = kl_divergence.mean().item()

        per_sample_loss = ppo_loss + kl_divergence

    elif self.use_sapo:
        ratio_reshaped = ratio.view(-1, actual_topk, self.token_level)
        ratio_inv = (old_probs / (dpo_new_probs + 1e-5)).clamp(min=1e-8)
        ratio_inv_reshaped = ratio_inv.view(-1, actual_topk, self.token_level)
        advantages_flat = grpo_advantages  # [B, K, 1]

        tau = torch.where(advantages_flat > 0,
                            torch.tensor(self.tau_pos, device=device),
                            torch.tensor(self.tau_neg, device=device))
        gate = torch.sigmoid(tau * (ratio_reshaped - 1.0)) * (4.0 / tau)
        ppo_loss_token = -(gate * advantages_flat)

        self.print_node("tau", tau)
        self.print_node("gate", gate)
        self.print_node("ppo_loss_token", ppo_loss_token)

        kl_divergence_token = torch.zeros_like(ppo_loss_token)
        if self.use_kl:
            kl_divergence_token = (ratio_inv_reshaped - ratio_inv_reshaped.log() - 1) * self.beta
            self.print_node("kl_div_token", kl_divergence_token)

        per_sample_loss = (ppo_loss_token + kl_divergence_token).mean(dim=-1)  # [B, K]
    elif self.use_gtpo:
        H_i_t_reshaped = token_entropy.view(-1, actual_topk, self.token_level)  # [B, K, L]

        advantages_flat_seq = grpo_advantages.view(-1, actual_topk, 1)  # [B, K, 1]
        is_correct = (advantages_flat_seq.squeeze(-1) > 0).float()  # [B, K]
        is_wrong = (advantages_flat_seq.squeeze(-1) <= 0).float()  # [B, K]

        is_correct_expanded = is_correct.unsqueeze(-1)  # [B, K, 1]
        H_correct_masked = H_i_t_reshaped * is_correct_expanded  # [B, K, L]
        H_sum_correct = H_correct_masked.sum(dim=1, keepdim=True)  # [B, 1, L]
        d_t = is_correct.sum(dim=1, keepdim=True).unsqueeze(-1).expand(-1, -1, self.token_level)  # [B, 1, L]

        adv_i_plus = advantages_flat_seq.expand(-1, -1, self.token_level)  # [B, K, L]
        alpha1 = 1.0
        alpha2 = 0.5
        H_ratio_correct = H_i_t_reshaped / (H_sum_correct + 1e-8)
        reward_plus = (alpha1 * adv_i_plus + alpha2 * H_ratio_correct * d_t) * is_correct_expanded

        # token reward reshaping for incorrect sequences
        inv_H_j_t = 1.0 / (H_i_t_reshaped.clamp(min=1e-8))  # [B, K, L]
        is_wrong_expanded = is_wrong.unsqueeze(-1)  # [B, K, 1]
        inv_H_wrong_masked = inv_H_j_t * is_wrong_expanded
        inv_H_sum_wrong = inv_H_wrong_masked.sum(dim=1, keepdim=True)  # [B, 1, L]
        h_t = is_wrong.sum(dim=1, keepdim=True).unsqueeze(-1).expand(-1, -1, self.token_level)  # [B, 1, L]

        inv_H_ratio_wrong = inv_H_j_t / (inv_H_sum_wrong + 1e-8)
        reward_minus = (alpha1 * (-1) + alpha2 * inv_H_ratio_wrong * h_t * (-1)) * is_wrong_expanded

        # combine rewards
        gtpo_advantages = reward_plus + reward_minus  # [B, K, L]

        self.print_node("gtpo_advantages:", gtpo_advantages)

        # compute loss
        ratio_reshaped = ratio.view(-1, actual_topk, self.token_level)
        ratio_inv = (old_probs / (dpo_new_probs + 1e-5)).clamp(min=1e-8)
        ratio_inv_reshaped = ratio_inv.view(-1, actual_topk, self.token_level)

        loss_unclipped = ratio_reshaped * gtpo_advantages
        loss_clipped = torch.clamp(ratio_reshaped, 1 - self.epsilon, 1 + self.epsilon) * gtpo_advantages
        ppo_loss_token = -torch.min(loss_unclipped, loss_clipped)

        kl_divergence_token = torch.zeros_like(ppo_loss_token)
        if self.use_kl:
            kl_divergence_token = (ratio_inv_reshaped - ratio_inv_reshaped.log() - 1) * self.beta
            self.print_node("kl_div_token", kl_divergence_token)

        per_sample_loss = (ppo_loss_token + kl_divergence_token).mean(dim=-1)  # [B, K]

    elif self.use_grpo_s:
        H_i_t_reshaped = token_entropy.view(-1, actual_topk, self.token_level)  # [B, K, L]
        H_hat_k = H_i_t_reshaped.mean(dim=-1, keepdim=True)  # [B, K, 1]

        advantages_flat_seq = grpo_advantages.view(-1, actual_topk, 1)  # [B, K, 1]
        is_correct = (advantages_flat_seq.squeeze(-1) > 0).float()  # [B, K]
        is_wrong = (advantages_flat_seq.squeeze(-1) <= 0).float()  # [B, K]

        # reward reshaping for correct sequences
        is_correct_expanded = is_correct.unsqueeze(-1)  # [B, K, 1]
        H_hat_correct_masked = H_hat_k * is_correct_expanded  # [B, K, 1]
        H_hat_sum_correct = H_hat_correct_masked.sum(dim=1, keepdim=True)  # [B, 1, 1]
        n = is_correct.sum(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]

        alpha1 = 1.0
        alpha2 = 0.5
        H_hat_ratio_correct = H_hat_k / (H_hat_sum_correct + 1e-8)  # [B, K, 1]
        reward_plus = (alpha1 * advantages_flat_seq + alpha2 * H_hat_ratio_correct * n) * is_correct_expanded

        # reward reshaping for incorrect sequences
        inv_H_hat_j = 1.0 / (H_hat_k.clamp(min=1e-8))  # [B, K, 1]
        is_wrong_expanded = is_wrong.unsqueeze(-1)  # [B, K, 1]
        inv_H_hat_wrong_masked = inv_H_hat_j * is_wrong_expanded
        inv_H_hat_sum_wrong = inv_H_hat_wrong_masked.sum(dim=1, keepdim=True)  # [B, 1, 1]
        m = is_wrong.sum(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]

        inv_H_hat_ratio_wrong = inv_H_hat_j / (inv_H_hat_sum_wrong + 1e-8)
        reward_minus = (alpha1 * (-1) + alpha2 * inv_H_hat_ratio_wrong * m * (-1)) * is_wrong_expanded

        grpo_s_advantages = reward_plus + reward_minus  # [B, K, 1]
        self.print_node("grpo_s_advantages:", grpo_s_advantages)

        ratio_reshaped = ratio.view(-1, actual_topk, self.token_level)
        ratio_inv = (old_probs / (dpo_new_probs + 1e-5)).clamp(min=1e-8)
        ratio_inv_reshaped = ratio_inv.view(-1, actual_topk, self.token_level)

        loss_unclipped = ratio_reshaped * grpo_s_advantages
        loss_clipped = torch.clamp(ratio_reshaped, 1 - self.epsilon, 1 + self.epsilon) * grpo_s_advantages
        ppo_loss_token = -torch.min(loss_unclipped, loss_clipped)

        kl_divergence_token = torch.zeros_like(ppo_loss_token)
        if self.use_kl:
            kl_divergence_token = (ratio_inv_reshaped - ratio_inv_reshaped.log() - 1) * self.beta
            self.print_node("kl_div_token", kl_divergence_token)

        per_sample_loss = (ppo_loss_token + kl_divergence_token).mean(dim=-1)  # [B, K]

    elif self.use_aepo:
        A_Acc = grpo_advantages.view(-1, actual_topk, 1).expand(-1, -1, self.token_level)  # [B, K, L]

        H_t = token_entropy.view(-1, actual_topk, self.token_level)  # [B, K, L]
        H_t_mean = H_t.mean()
        H_t_std = H_t.std() + 1e-4
        A_Delta_H = (H_t - H_t_mean) / H_t_std  # [B, K, L]

        a = 0.3  # entropy reshaping coefficient, public-dataset setting (0.5 on the industrial dataset)
        aepo_advantages = A_Acc * (1.0 + a * A_Delta_H)  # [B, K, L]

        self.print_node("A_Acc (from grpo_advantages):", A_Acc)
        self.print_node("A_Delta_H (entropy):", A_Delta_H)
        self.print_node("aepo_advantages:", aepo_advantages)

        ratio_reshaped = ratio.view(-1, actual_topk, self.token_level)
        ratio_inv = (old_probs / (dpo_new_probs + 1e-5)).clamp(min=1e-8)
        ratio_inv_reshaped = ratio_inv.view(-1, actual_topk, self.token_level)

        loss_unclipped = ratio_reshaped * aepo_advantages
        loss_clipped = torch.clamp(ratio_reshaped, 1 - self.epsilon, 1 + self.epsilon) * aepo_advantages
        ppo_loss_token = -torch.min(loss_unclipped, loss_clipped)

        kl_divergence_token = torch.zeros_like(ppo_loss_token)
        if self.use_kl:
            kl_divergence_token = (ratio_inv_reshaped - ratio_inv_reshaped.log() - 1) * self.beta
            self.print_node("kl_div_token", kl_divergence_token)

        per_sample_loss = (ppo_loss_token + kl_divergence_token).mean(dim=-1)  # [B, K]

    elif self.use_sgpo:
        log_ratio = ratio.log()
        log_ratio_mean = log_ratio.view(-1, actual_topk, self.token_level).mean(dim=-1)
        ratio_mean = log_ratio_mean.exp().unsqueeze(-1)  # [B, K, 1]

        log_p_new_grad = dpo_new_probs.log().view(-1, actual_topk, self.token_level)  # [B, K, L], keeps grad
        log_p_new = log_p_new_grad.detach()  # A_t is treated as a stop-gradient term
        log_p_mean = log_p_new.mean(dim=-1, keepdim=True)  # [B, K, 1]
        u_t = log_p_new - log_p_mean  # [B, K, L]

        c_max = self.sgpo_c_max
        c_min = self.sgpo_c_min
        N = self.sgpo_anneal_steps
        n = self.global_step
        c_n = max(c_min, c_max - (c_max - c_min) * n / N)
        if self.global_step % self.log_steps == 1:
            print(f"c:{c_n}")

        # λ^(k) = |A_s^(k)| * sqrt((c(n)-1) / (mean(u_t^2) + eps))
        u_t_sq_mean = (u_t ** 2).mean(dim=-1, keepdim=True)  # [B, K, 1]
        advantages_abs = grpo_advantages.abs().view(-1, actual_topk, 1)  # [B, K, 1]
        lambda_k = advantages_abs * torch.sqrt((c_n - 1.0) / (u_t_sq_mean + 1e-8))  # [B, K, 1]

        # A_t^(k) = A_s^(k) + λ^(k) * u_t^(k)  if A_s > 0
        # A_t^(k) = A_s^(k) - λ^(k) * u_t^(k)  if A_s <= 0
        advantages_seq = grpo_advantages.view(-1, actual_topk, 1)  # [B, K, 1]
        is_positive = (advantages_seq > 0).float()  # [B, K, 1]

        A_t = advantages_seq + is_positive * lambda_k * u_t - (1.0 - is_positive) * lambda_k * u_t  # [B, K, L]

        # log_p_new = dpo_new_probs.log().view(-1, actual_topk, self.token_level)  # [B, K, L]
        #
        # # mean over the same level
        # # log_p_mean = log_p_new.mean(dim=(0, 1), keepdim=True)  # [1, 1, L]
        # log_p_mean = log_p_new.mean()
        # u_t = log_p_new - log_p_mean  # [B, K, L]
        #
        # c_max_pos = 4.0
        # c_min_pos = 1.0
        # c_max_neg = 4.0
        # c_min_neg = 1.0
        # N = self.sgpo_anneal_steps
        # n = self.global_step
        # c_n_pos = c_max_pos - (c_max_pos - c_min_pos) * n / N
        # c_n_neg = c_max_neg - (c_max_neg - c_min_neg) * n / N
        # # c_n_neg = 1.0
        # if self.global_step % self.log_steps == 1:
        #     print(f"c_pos:{c_n_pos}, c_neg:{c_n_neg}")
        #
        # # λ^(k) = |A_s^(k)| * sqrt((c(n)-1) / (mean(u_t^2) + eps))
        # advantages_abs = grpo_advantages.abs().view(-1, actual_topk, 1)  # [B, K, 1]
        # advantages_seq = grpo_advantages.view(-1, actual_topk, 1)  # [B, K, 1]
        # is_positive = (advantages_seq > 0).float()  # [B, K, 1]
        #
        # # separate c_n for positive/negative samples
        # c_n = is_positive * c_n_pos + (1.0 - is_positive) * c_n_neg  # [B, K, 1]
        # lambda_k = advantages_abs * torch.sqrt(c_n - 1.0)  # [B, K, 1]
        #
        # A_t = advantages_seq + is_positive * lambda_k * u_t - (1.0 - is_positive) * lambda_k * u_t  # [B, K, L]

        self.print_node("u_t (reshaping term):", u_t)
        self.print_node("lambda_k:", lambda_k)
        self.print_node("A_t (reshaped advantages):", A_t)

        # sgpo_token_grad=1 (default): token-level ratio whose forward value still equals s_i,
        #   but whose gradient is routed to each token individually,
        #     ratio_token = s_i * exp(l_t - stopgrad(l_t))  ->  d/dtheta = s_i * grad log pi(y_t)
        #   giving grad J ~ s_i * sum_t A_t * grad log pi(y_t) (Eq. 13).
        # sgpo_token_grad=0 (legacy): s_i enters as a plain per-sequence scalar; the reshaping
        #   then only acts through the sign(A_t) gating of torch.min() outside the trust
        #   region, because sum_t u_t == 0 makes mean_t(A_t) == A_s.
        if self.sgpo_token_grad:
            eff_ratio = ratio_mean.detach() * torch.exp(log_p_new_grad - log_p_new_grad.detach())
        else:
            eff_ratio = ratio_mean
        self.print_node("eff_ratio:", eff_ratio, 64)

        loss_unclipped = eff_ratio * A_t  # [B, K, L]
        loss_clipped = torch.clamp(eff_ratio, 1 - self.epsilon, 1 + self.epsilon) * A_t
        ppo_loss_token = -torch.min(loss_unclipped, loss_clipped)  # [B, K, L]

        self.print_node("sgpo_loss_token:", ppo_loss_token)

        # KL divergence
        ratio_inv = (old_probs / (dpo_new_probs + 1e-5)).clamp(min=1e-8)
        log_ratio_inv = ratio_inv.log()
        log_ratio_inv_mean = log_ratio_inv.view(-1, actual_topk, self.token_level).mean(dim=-1)
        ratio_inv_mean = log_ratio_inv_mean.exp().unsqueeze(-1)

        kl_divergence_token = torch.zeros_like(ppo_loss_token)
        if self.use_kl:
            kl_divergence_token = (ratio_inv_mean - ratio_inv_mean.log() - 1).expand(-1, -1,
                                                                                        self.token_level) * self.beta
            self.print_node("kl_div_token", kl_divergence_token)
            self.metrics_to_log['kl_divergence_mean'] = kl_divergence_token.mean().item()

        per_sample_loss = (ppo_loss_token + kl_divergence_token).mean(dim=-1)  # [B, K]

    else:
        ratio_reshaped = ratio.view(-1, actual_topk, self.token_level)

        ratio_inv = (old_probs / (dpo_new_probs + 1e-5)).clamp(min=1e-8)
        ratio_inv_reshaped = ratio_inv.view(-1, actual_topk, self.token_level)
        advantages_flat = grpo_advantages  # [B, K, 1]

        loss_unclipped = ratio_reshaped * advantages_flat
        loss_clipped = torch.clamp(ratio_reshaped, 1 - self.epsilon, 1 + self.epsilon) * advantages_flat
        ppo_loss_token = -torch.min(loss_unclipped, loss_clipped)

        kl_divergence_token = torch.zeros_like(ppo_loss_token)
        if self.use_kl:
            kl_divergence_token = (ratio_inv_reshaped - ratio_inv_reshaped.log() - 1) * self.beta
            self.print_node("kl_div_token", kl_divergence_token)

        per_sample_loss = (ppo_loss_token + kl_divergence_token).mean(dim=-1)  # [B, K]
    self.print_node("per_sample_loss", per_sample_loss)
    total_loss = per_sample_loss.mean()

    # self.dprint("per_token_loss.item():",total_loss.item())
    self.print_node("per_token_loss", total_loss)
    # print(f"GRPO: r={mean_reward.item():.1f}, L={total_loss.item():.2f}")

    grad = torch.autograd.grad(total_loss, dpo_new_probs, retain_graph=True, allow_unused=True)[0]
    if grad is not None:
        grad_flat = grad.view(-1)  # [B*K*L]
        self.metrics_to_log['grpo_grad_norm'] = grad_flat.norm().item()
        self.dprint("grpo_grad_norm:", grad_flat.norm().item())

        if pos_mask.sum() > 0:
            self.metrics_to_log['grpo_grad_norm_adv_pos'] = grad_flat[pos_mask].norm().item()
            self.dprint("grpo_grad_norm_adv_pos:", grad_flat[pos_mask].norm().item())
        if neg_mask.sum() > 0:
            self.metrics_to_log['grpo_grad_norm_adv_neg'] = grad_flat[neg_mask].norm().item()
            self.dprint("grpo_grad_norm_adv_neg:", grad_flat[neg_mask].norm().item())

    return total_loss * self.dpo_weight