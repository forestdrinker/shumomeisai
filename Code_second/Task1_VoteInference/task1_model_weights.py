"""
Task 1 模型修正版 - 估计粉丝/评委权重
=====================================

【核心修改】
原代码: b = rJ + rF (硬编码1:1)
修改后: b = w_J * rJ + w_F * rF, 其中 w_J, w_F 是估计参数

【输出】
- w_judge: 评委权重 (0到1)
- w_fan: 粉丝权重 (= 1 - w_judge)
- 这回答了"粉丝和评委的实际影响力占比"
"""

import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from jax import random, nn


def probabilistic_model_with_weights(
    n_pairs,
    n_weeks,
    active_mask,
    observed_scores,
    judge_percents,
    elim_events,
    final_ranking,
    rule_segment,
    
    # 超参数
    tau_u=1.0,
    sigma_u=0.1,
    lambda_pl=2.0,
    lambda_fin=2.0,
    gamma_save=5.0,
    rho=0.05,
    
    # 是否估计权重（设为False则使用固定0.5）
    estimate_weights=True,
    fixed_weight_judge=0.5,  # 如果不估计，使用这个固定值
):
    """
    DWTS投票推断模型 - 带权重估计
    
    【新增输出】
    - w_judge: 评委的实际权重 (0到1之间)
    - w_fan: 粉丝的实际权重 (= 1 - w_judge)
    """
    
    # ======================
    # 1. 估计或固定权重参数
    # ======================
    if estimate_weights:
        # 使用Beta先验: 均值0.5，但允许从数据中学习
        # Beta(2, 2) 在0.5附近，但有一定方差
        w_judge = numpyro.sample("w_judge", dist.Beta(2.0, 2.0))
    else:
        w_judge = fixed_weight_judge
    
    w_fan = 1.0 - w_judge
    
    # 记录权重（即使是固定的也记录）
    numpyro.deterministic("w_judge_value", w_judge)
    numpyro.deterministic("w_fan_value", w_fan)
    
    # ======================
    # 2. 人气潜变量 u
    # ======================
    with numpyro.plate("pairs", n_pairs, dim=-1):
        u_init_raw = numpyro.sample("u_init_raw", dist.Normal(0, 1))
        u_init = numpyro.deterministic("u_init", u_init_raw * tau_u)
    
    with numpyro.plate("time_innovation", n_weeks - 1, dim=-2):
        with numpyro.plate("pairs_inn", n_pairs, dim=-1):
            u_innov_raw = numpyro.sample("u_innov_raw", dist.Normal(0, 1))
            u_innov = u_innov_raw * sigma_u
    
    u_init_expanded = jnp.expand_dims(u_init, axis=0)
    u_innov_full = jnp.concatenate([u_init_expanded, u_innov], axis=0)
    u = jnp.cumsum(u_innov_full, axis=0)
    
    numpyro.deterministic("u", u)
    
    # ======================
    # 3. 投票份额 v
    # ======================
    huge_neg = -1e9
    u_masked = jnp.where(active_mask, u, huge_neg)
    v = nn.softmax(u_masked, axis=-1)
    
    numpyro.deterministic("v", v)
    
    # ======================
    # 4. 计算 Badness（使用估计的权重！）
    # ======================
    
    kappa_J = 1.0
    kappa_F = 0.1
    kappa_C = 0.1
    
    def soft_rank(scores, kappa, mask):
        s_col = scores[:, None]
        s_row = scores[None, :]
        diff = s_row - s_col
        sig = nn.sigmoid(diff / kappa)
        valid_k = mask[None, :]
        sig_masked = jnp.where(valid_k, sig, 0.0)
        r = 1.0 + jnp.sum(sig_masked, axis=1) - 0.5
        return jnp.where(mask, r, 0.0)
    
    if rule_segment == 'percent':
        # 【修改】加权组合
        # 原: C = judge_percents + v
        # 新: C = w_judge * judge_percents + w_fan * v
        C = w_judge * judge_percents + w_fan * v
        
        def compute_week_b_percent(week_idx):
            return soft_rank(C[week_idx], kappa_C, active_mask[week_idx])
        
        b = jnp.stack([compute_week_b_percent(t) for t in range(n_weeks)])
        
    else:  # rank or rank_save
        # 【修改】加权排名
        # 原: b = rJ + rF
        # 新: b = w_judge * rJ + w_fan * rF
        def compute_week_b_rank(week_idx):
            rJ = soft_rank(observed_scores[week_idx], kappa_J, active_mask[week_idx])
            rF = soft_rank(v[week_idx], kappa_F, active_mask[week_idx])
            return w_judge * rJ + w_fan * rF  # 加权！
        
        b = jnp.stack([compute_week_b_rank(t) for t in range(n_weeks)])
    
    numpyro.deterministic("b", b)
    
    # ======================
    # 5. 似然：淘汰事件（同原代码）
    # ======================
    for t, eliminated_indices in elim_events:
        if len(eliminated_indices) == 0:
            continue
        
        b_t = b[t]
        mask_t = active_mask[t]
        
        if rule_segment == 'rank_save' and len(eliminated_indices) == 1:
            e = eliminated_indices[0]
            logits = lambda_pl * b_t
            logits = jnp.where(mask_t, logits, -1e9)
            
            rJ_t = soft_rank(observed_scores[t], kappa_J, mask_t)
            
            exp_b = jnp.exp(logits)
            sum_exp = jnp.sum(exp_b)
            
            def body_fn(j, val):
                p_e = exp_b[e] / sum_exp
                p_j_given_e = exp_b[j] / (sum_exp - exp_b[e])
                p_j = exp_b[j] / sum_exp
                p_e_given_j = exp_b[e] / (sum_exp - exp_b[j])
                prob_bottom2 = (p_e * p_j_given_e) + (p_j * p_e_given_j)
                
                logits_judge = gamma_save * rJ_t
                p_judge_elim_e = jnp.exp(logits_judge[e]) / (
                    jnp.exp(logits_judge[e]) + jnp.exp(logits_judge[j]))
                
                term = prob_bottom2 * p_judge_elim_e
                valid = (j != e) & mask_t[j]
                return val + jnp.where(valid, term, 0.0)
            
            total_prob = jax.lax.fori_loop(0, n_pairs, body_fn, 0.0)
            
            log_p_save = jnp.log(total_prob + 1e-10)
            n_active_count = jnp.sum(mask_t)
            log_p_uniform = -jnp.log(n_active_count + 1e-10)
            
            log_p_mixed = jnp.logaddexp(
                jnp.log(1.0 - rho) + log_p_save,
                jnp.log(rho) + log_p_uniform
            )
            
            numpyro.factor(f"elim_save_{t}", log_p_mixed)
        
        else:
            current_mask = mask_t
            for e_idx in eliminated_indices:
                logit_e = lambda_pl * b_t[e_idx]
                logits_all = lambda_pl * b_t
                logits_all_masked = jnp.where(current_mask, logits_all, -1e9)
                lse = nn.logsumexp(logits_all_masked)
                
                log_p_main = logit_e - lse
                n_active_count = jnp.sum(current_mask)
                log_p_uniform = -jnp.log(n_active_count + 1e-10)
                
                log_p_mixed = jnp.logaddexp(
                    jnp.log(1.0 - rho) + log_p_main,
                    jnp.log(rho) + log_p_uniform
                )
                
                numpyro.factor(f"elim_pl_{t}_{e_idx}", log_p_mixed)
                current_mask = current_mask.at[e_idx].set(False)
    
    # ======================
    # 6. 似然：最终排名
    # ======================
    T_final = n_weeks - 1
    b_final = b[T_final]
    a_final = -b_final
    
    current_mask_fin = active_mask[T_final]
    
    for rank_i, idx in enumerate(final_ranking):
        logit_i = lambda_fin * a_final[idx]
        logits_all = lambda_fin * a_final
        logits_all_masked = jnp.where(current_mask_fin, logits_all, -1e9)
        lse = nn.logsumexp(logits_all_masked)
        
        numpyro.factor(f"final_rank_{rank_i}", logit_i - lse)
        current_mask_fin = current_mask_fin.at[idx].set(False)


# ======================
# 提取权重结果的辅助函数
# ======================
def extract_weight_results(samples_dict, season_info):
    """
    从MCMC样本中提取权重估计结果
    
    Returns:
        dict: {
            'season': ...,
            'rule': ...,
            'w_judge_mean': 评委权重后验均值,
            'w_judge_std': 评委权重后验标准差,
            'w_judge_ci_low': 2.5%分位数,
            'w_judge_ci_high': 97.5%分位数,
            'w_fan_mean': 粉丝权重后验均值,
            ...
        }
    """
    import numpy as np
    
    w_judge_samples = samples_dict.get('w_judge', samples_dict.get('w_judge_value'))
    
    if w_judge_samples is None:
        return None
    
    w_judge_samples = np.array(w_judge_samples)
    w_fan_samples = 1.0 - w_judge_samples
    
    return {
        'season': season_info.get('season'),
        'rule_segment': season_info.get('rule_segment'),
        
        # 评委权重
        'w_judge_mean': np.mean(w_judge_samples),
        'w_judge_std': np.std(w_judge_samples),
        'w_judge_median': np.median(w_judge_samples),
        'w_judge_ci_low': np.percentile(w_judge_samples, 2.5),
        'w_judge_ci_high': np.percentile(w_judge_samples, 97.5),
        
        # 粉丝权重
        'w_fan_mean': np.mean(w_fan_samples),
        'w_fan_std': np.std(w_fan_samples),
        'w_fan_ci_low': np.percentile(w_fan_samples, 2.5),
        'w_fan_ci_high': np.percentile(w_fan_samples, 97.5),
    }


# ======================
# 使用说明
# ======================
"""
【如何使用】

1. 替换 task1_runner.py 中的模型调用:
   
   原: from task1_model import probabilistic_model
   新: from task1_model_weights import probabilistic_model_with_weights as probabilistic_model

2. 运行MCMC后，样本中会包含:
   - w_judge: 评委权重的后验样本
   - w_judge_value: 评委权重值
   - w_fan_value: 粉丝权重值

3. 分析结果:
   weights = extract_weight_results(samples, {'season': 28, 'rule_segment': 'rank_save'})
   print(f\"评委实际权重: {weights['w_judge_mean']:.1%} [{weights['w_judge_ci_low']:.1%}, {weights['w_judge_ci_high']:.1%}]\")


【预期结果】

如果估计的 w_judge ≈ 0.5:
  → 官方声称的50:50是真实的

如果 w_judge > 0.5:
  → 评委的实际影响力大于粉丝

如果 w_judge < 0.5:
  → 粉丝的实际影响力大于评委


【不同规则的预期】

rank规则 (早期赛季):
  - 可能接近50:50
  - 但实际影响力取决于分数分布

percent规则 (中期赛季):
  - 官方是50:50
  - 估计结果应该接近0.5

rank_save规则 (近期赛季):
  - 评委有额外的救援权
  - 可能 w_judge > 0.5
"""
