"""
Task 1 概率模型 - 修正版
=========================
主要修改：
1. 降低 lambda_pl 从 5.0 到 2.0 (增加不确定性)
2. 降低 lambda_fin 从 5.0 到 2.0
3. 添加详细注释说明参数选择理由
"""

import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from jax import random, nn


def probabilistic_model(
    n_pairs,
    n_weeks,
    active_mask,      # (n_weeks, n_pairs) Boolean/0-1 mask
    observed_scores,  # (n_weeks, n_pairs) raw sum-scores S_it (for rank rule)
    judge_percents,   # (n_weeks, n_pairs) pJ_it (for percent rule)
    elim_events,      # List of (week_idx, eliminated_indices_list)
    final_ranking,    # (n_finalists,) indices of finalists in order 1st, 2nd...
    rule_segment,     # String: 'rank', 'percent', 'rank_save'
    
    # ============== 修正后的超参数 ==============
    tau_u=1.0,        # 初始人气标准差
    sigma_u=0.1,      # 随机游走漂移标准差
    
    # 【修正】降低harshness以增加不确定性
    lambda_pl=2.0,    # 淘汰harshness (原5.0→2.0)
    lambda_fin=2.0,   # 最终排名harshness (原5.0→2.0)
    
    gamma_save=5.0,   # 评委save强度
    rho=0.05,         # 随机淘汰概率
    
    # Soft-rank scales
    kappa_J=1.0,      # 评委分数scale
    kappa_F=0.1,      # 投票份额scale
    kappa_C=0.1,      # 组合百分比scale
):
    """
    DWTS投票推断的NUTS模型
    
    【参数选择理由】
    - lambda_pl=2.0: 降低这个值使淘汰概率分布更平缓，
      增加后验不确定性，避免过度拟合。
    - sigma_u=0.1: 随机游走步长，控制人气变化速度
    - rho=0.05: 允许5%的"爆冷"淘汰，增加模型鲁棒性
    
    Dimensions:
      T = n_weeks
      N = n_pairs
    """
    
    # =====================================================
    # 1. 先验 / 潜变量
    # =====================================================
    
    # 初始人气 u_{i,0}
    with numpyro.plate("pairs", n_pairs, dim=-1):
        u_init_raw = numpyro.sample("u_init_raw", dist.Normal(0, 1))
        u_init = numpyro.deterministic("u_init", u_init_raw * tau_u)
    
    # 随机游走增量 eps_{i,t}
    with numpyro.plate("time_innovation", n_weeks - 1, dim=-2):
        with numpyro.plate("pairs_inn", n_pairs, dim=-1):
            u_innov_raw = numpyro.sample("u_innov_raw", dist.Normal(0, 1))
            u_innov = u_innov_raw * sigma_u
    
    # 构建完整的u轨迹
    u_init_expanded = jnp.expand_dims(u_init, axis=0)
    u_innov_full = jnp.concatenate([u_init_expanded, u_innov], axis=0)
    u = jnp.cumsum(u_innov_full, axis=0)
    
    numpyro.deterministic("u", u)
    
    # =====================================================
    # 2. 投票份额 v_{i,t}
    # =====================================================
    
    # 对非活跃选手设置极小值
    huge_neg = -1e9
    u_masked = jnp.where(active_mask, u, huge_neg)
    
    # Softmax得到投票份额
    v = nn.softmax(u_masked, axis=-1)
    
    numpyro.deterministic("v", v)
    
    # =====================================================
    # 3. 计算 "Badness" b_{i,t}
    # =====================================================
    
    def soft_rank(scores, kappa, mask):
        """软排名函数"""
        s_col = scores[:, None]
        s_row = scores[None, :]
        diff = s_row - s_col
        
        sig = nn.sigmoid(diff / kappa)
        
        valid_k = mask[None, :]
        sig_masked = jnp.where(valid_k, sig, 0.0)
        
        r = 1.0 + jnp.sum(sig_masked, axis=1) - 0.5
        
        return jnp.where(mask, r, 0.0)
    
    # 根据规则计算badness
    if rule_segment == 'percent':
        # Combined = pJ + v
        C = judge_percents + v
        
        def compute_week_b_percent(week_idx):
            return soft_rank(C[week_idx], kappa_C, active_mask[week_idx])
        
        b = jnp.stack([compute_week_b_percent(t) for t in range(n_weeks)])
        
    else:  # rank or rank_save
        # b = rJ + rF
        def compute_week_b_rank(week_idx):
            rJ = soft_rank(observed_scores[week_idx], kappa_J, active_mask[week_idx])
            rF = soft_rank(v[week_idx], kappa_F, active_mask[week_idx])
            return rJ + rF
        
        b = jnp.stack([compute_week_b_rank(t) for t in range(n_weeks)])
    
    numpyro.deterministic("b", b)
    
    # =====================================================
    # 4. 似然：淘汰事件
    # =====================================================
    
    for t, eliminated_indices in elim_events:
        if len(eliminated_indices) == 0:
            continue
        
        b_t = b[t]
        mask_t = active_mask[t]
        
        # rank_save 规则的特殊处理
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
            
            # 混合模型：(1-rho)*P_save + rho*P_uniform
            log_p_save = jnp.log(total_prob + 1e-10)
            n_active_count = jnp.sum(mask_t)
            log_p_uniform = -jnp.log(n_active_count + 1e-10)
            
            log_p_mixed = jnp.logaddexp(
                jnp.log(1.0 - rho) + log_p_save,
                jnp.log(rho) + log_p_uniform
            )
            
            numpyro.factor(f"elim_save_{t}", log_p_mixed)
        
        else:
            # 标准 Plackett-Luce
            current_mask = mask_t
            for e_idx in eliminated_indices:
                logit_e = lambda_pl * b_t[e_idx]
                
                logits_all = lambda_pl * b_t
                logits_all_masked = jnp.where(current_mask, logits_all, -1e9)
                lse = nn.logsumexp(logits_all_masked)
                
                log_p_main = logit_e - lse
                
                # 混合模型
                n_active_count = jnp.sum(current_mask)
                log_p_uniform = -jnp.log(n_active_count + 1e-10)
                
                log_p_mixed = jnp.logaddexp(
                    jnp.log(1.0 - rho) + log_p_main,
                    jnp.log(rho) + log_p_uniform
                )
                
                numpyro.factor(f"elim_pl_{t}_{e_idx}", log_p_mixed)
                
                current_mask = current_mask.at[e_idx].set(False)
    
    # =====================================================
    # 5. 似然：最终排名
    # =====================================================
    
    T_final = n_weeks - 1
    b_final = b[T_final]
    a_final = -b_final  # goodness = -badness
    
    current_mask_fin = active_mask[T_final]
    
    for rank_i, idx in enumerate(final_ranking):
        logit_i = lambda_fin * a_final[idx]
        
        logits_all = lambda_fin * a_final
        logits_all_masked = jnp.where(current_mask_fin, logits_all, -1e9)
        lse = nn.logsumexp(logits_all_masked)
        
        numpyro.factor(f"final_rank_{rank_i}", logit_i - lse)
        
        current_mask_fin = current_mask_fin.at[idx].set(False)


# =====================================================
# 超参数敏感性分析建议
# =====================================================
"""
建议测试以下lambda_pl值并报告结果：

lambda_pl = 0.5  # 非常平缓，高不确定性
lambda_pl = 1.0  # 平缓
lambda_pl = 2.0  # 推荐值 (当前)
lambda_pl = 5.0  # 较陡峭
lambda_pl = 10.0 # 非常陡峭，低不确定性

预期效果：
- lambda_pl越小，Coverage越高但可能>95%
- lambda_pl越大，Accuracy越高但Coverage可能<85%
- 选择使Coverage≈90%的值
"""
