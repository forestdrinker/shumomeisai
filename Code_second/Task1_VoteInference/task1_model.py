
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
    
    # Hyperparameters (can be fixed or passed as args)
    tau_u=1.0,        # Initial popularity scale
    sigma_u=0.1,      # Random walk drift scale
    kappa_c=0.1,      # Soft-rank scale generic (we might need specific ones)
    lambda_pl=5.0,    # Elimination harshness (Relaxed from 10.0)
    lambda_fin=5.0,   # Final placement harshness (Relaxed from 10.0)
    gamma_save=5.0,   # Judges save strength
    rho=0.05          # Shock/Random elimination probability
):
    """
    NUTS Model for DWTS Vote Inference.
    Dimensions:
      T = n_weeks
      N = n_pairs
    """
    
    # 1. Priors / Latent Variables
    # u_{i,t} Random Walk
    # We can model u as shape (T, N)
    
    # Initial state u_{i,0} (Week 1 is index 0 in python)
    # u_0 ~ Normal(0, tau_u) -> Re-parameterized: u_init_raw * tau_u
    with numpyro.plate("pairs", n_pairs, dim=-1):
        u_init_raw = numpyro.sample("u_init_raw", dist.Normal(0, 1))
        u_init = numpyro.deterministic("u_init", u_init_raw * tau_u)
        
    # Innovations for t=1..T-1
    # eps_{i,t} ~ Normal(0, sigma_u) -> Re-parameterized
    with numpyro.plate("time_innovation", n_weeks - 1, dim=-2): 
        with numpyro.plate("pairs_inn", n_pairs, dim=-1):     
            u_innov_raw = numpyro.sample("u_innov_raw", dist.Normal(0, 1))
            u_innov = u_innov_raw * sigma_u
            
    # Construct full u trajectory via cumsum
    # u_innov shape: (T-1, N)
    # Prepend u_init: (1, N)
    # result: (T, N)
    
    # Need to properly shape u_init to (1, N)
    u_init_expanded = jnp.expand_dims(u_init, axis=0) 
    u_innov_full = jnp.concatenate([u_init_expanded, u_innov], axis=0) # (T, N)
    u = jnp.cumsum(u_innov_full, axis=0)
    
    numpyro.deterministic("u", u) # Record u
    
    # 2. Vote Share v_{i,t} via Softmax (masked by active set)
    # We need to ensure we only softmax over ACTIVE contestants.
    # Inactive entries should effectively have -inf score or be masked out later.
    # But for calculation of v_{i,t} for active ones, we need log_sum_exp over active only.
    
    # Trick: Set u of inactive to -1e9 so exp(u) is 0.
    # active_mask is likely 1 for active, 0 for inactive.
    huge_neg = -1e9
    u_masked = jnp.where(active_mask, u, huge_neg)
    
    # Compute v
    # v = exp(u) / sum(exp(u)) per week
    v = nn.softmax(u_masked, axis=-1)
    
    # Verify: if active_mask is all 0 for a row (shouldn't happen), v might be nan
    # We assume data is prepared such that at least 1 person is active.
    
    numpyro.deterministic("v", v)
    
    # 3. Compute "Badness" b_{i,t} based on rule
    # b_{i,t} should be high for those likely to be eliminated.
    # Formulae depend on rule_segment.
    
    # For computation, we need the "soft rank" function.
    # rank(x_i) = 1 + sum_k sigma((x_k - x_i)/kappa)
    # If x is "good score" (bigger is better), then smaller rank is better.
    # Badness usually aligns with Rank (Index). Bigger Rank (e.g. 10th place) = Bad.
    
    def soft_rank(scores, kappa, mask):
        # scores: (N,)
        # mask: (N,) boolean
        # returns: (N,) rank approximation (1..M) masked
        
        # Broadcast difference: x_k - x_i
        # s_col: (N, 1), s_row: (1, N)
        s_col = scores[:, None]
        s_row = scores[None, :]
        diff = s_row - s_col # diff[i, k] = x_k - x_i
        
        # Sigmoid
        sig = nn.sigmoid(diff / kappa) # High if x_k > x_i (k beats i)
        
        # Sum over k != i and k is active
        # Mask logic: only count k if k is active.
        # Also result i is only valid if i is active.
        valid_k = mask[None, :] # (1, N)
        
        # Diagonal should be 0 (k!=i), sigmoid(0)=0.5, usually we subtract 0.5 or just handle diag?
        # Formula: sum_{k!=i} ...
        # sigmoid(0) = 0.5. rank of self vs self is not counted.
        # We can subtract 0.5 * mask from sum, or manually mask diag.
        # simpler: rank = 1 + sum_{k} sigmoid(...) - 0.5 (since k=i gives 0.5)
        # But only sum over active k.
        
        # Mask inactive k in the sum:
        sig_masked = jnp.where(valid_k, sig, 0.0)
        
        # Sum rows
        r = 1.0 + jnp.sum(sig_masked, axis=1) - 0.5 # Sub self-comparison
        
        return jnp.where(mask, r, 0.0) # Return 0 for inactive

    # Decide Badness b
    # rule: rank, percent, rank_save
    
    # We operate week by week or vectorized?
    # soft_rank on (T, N) matrix requires (T, N, N) memory if vectorized fully Naively. 
    # N ~ 13, T ~ 10. 10*13*13 is tiny. Vectorization is fine.
    
    # Scale kappa usually depends on std of scores, but we use fixed small number or hyperparam for now.
    # The plan says kappa ~ c / Delta. 
    # For implementation simplicity in core model, we use passed scalar or constant.
    kappa_J = 1.0 # Score scale ~ 30. Relaxed from 0.5 to 1.0
    kappa_F = 0.1 # Vote share ~ 0.1. Relaxed from 0.05 to 0.1
    kappa_C = 0.1 # Combined percent. Relaxed from 0.05 to 0.1
    
    b = jnp.zeros((n_weeks, n_pairs))
    
    if rule_segment == 'percent':
        # Combined = pJ + v
        C = judge_percents + v
        # Badness = SoftRank(C)
        # Vectorized soft_rank?
        # Let's map over weeks
        def compute_week_b_percent(week_idx):
             return soft_rank(C[week_idx], kappa_C, active_mask[week_idx])
             
        b = jnp.stack([compute_week_b_percent(t) for t in range(n_weeks)])
        
    else: # rank or rank_save
        # rJ = SoftRank(S)
        # rF = SoftRank(v)
        # b = rJ + rF
        def compute_week_b_rank(week_idx):
            rJ = soft_rank(observed_scores[week_idx], kappa_J, active_mask[week_idx])
            rF = soft_rank(v[week_idx], kappa_F, active_mask[week_idx])
            return rJ + rF
            
        b = jnp.stack([compute_week_b_rank(t) for t in range(n_weeks)])
        
    numpyro.deterministic("b", b)

    # 4. Likelihood: Elimination Events
    # elim_events is list of tuples: (t, [e1, e2...])
    
    for t, eliminated_indices in elim_events:
        # t is int, eliminated_indices is list of ints
        if len(eliminated_indices) == 0:
            continue
            
        # Current week badness
        b_t = b[t] # (N,)
        mask_t = active_mask[t] # (N,)
        
        # "Bottom 2 + Save" Logic vs "Lowest Combined" Logic
        # If 'rank_save' AND (we are in bottom 2 scenario?), the prompt says:
        # "For save week: first pick bottom-two, then judges pick one to eliminate"
        # We need to know if this specific week was a "Save" week. 
        # Usually checking `rule_segment` is generally applied to season.
        # But specific week logic (like double elim) matters.
        # For simplicity, if rule_segment == 'rank_save', we use save logic for ALL eliminations?
        # Or only if standard elim? 
        # The prompt Execution Core says: "If save enabled: use Judges' Save marginal likelihood".
        # We will assume if rule_segment is 'rank_save' and n_elim == 1, it might be a save. 
        # But Save usually forces Bottom 2.
        
        # Logic: 
        # If segment == 'rank_save': Use Save Likelihood
        # Else: Use Standard Plackett-Luce on b_t
        
        if rule_segment == 'rank_save':
             # SAVE LOGIC
             # Only implemented for single elimination effectively in the formula (bottom 2 -> 1 out)
             # If multiple eliminated, maybe fallback to PL?
             if len(eliminated_indices) == 1:
                 e = eliminated_indices[0] # The one eliminated
                 
                 # We need to sum over all possible "other" bottom-2 person j
                 # P(e | b) = sum_{j!=e} P(z={e,j}) * P(e | {e,j}, Judges)
                 
                 # 1. P(z={e,j}) probability that {e,j} are the bottom 2.
                 # Using PL to pick 2 "worst" (highest b).
                 # P(z1=e, z2=j) = exp(lam*b_e)/Sum * exp(lam*b_j)/Sum_excl_e
                 
                 # To be efficient, calculate logprobs
                 logits = lambda_pl * b_t
                 # Mask inactive by setting logits to -inf
                 logits = jnp.where(mask_t, logits, -1e9)
                 
                 # We need to sum over j != e
                 
                 # Calculate rJ_t once (constant for the loop)
                 rJ_t = soft_rank(observed_scores[t], kappa_J, mask_t)
                 
                 exp_b = jnp.exp(logits)
                 sum_exp = jnp.sum(exp_b)
                 
                 # We iterate over all j using fori_loop (static bounds) with masking
                 
                 def body_fn(j, val):
                      # if j == e or inactive, skip (prob 0)
                      # Implementing via `where` masks
                      
                      # Case 1: First e, then j
                      p_e = exp_b[e] / sum_exp
                      p_j_given_e = exp_b[j] / (sum_exp - exp_b[e])
                      
                      # Case 2: First j, then e
                      p_j = exp_b[j] / sum_exp
                      p_e_given_j = exp_b[e] / (sum_exp - exp_b[j])
                      
                      prob_bottom2 = (p_e * p_j_given_e) + (p_j * p_e_given_j)
                      
                      # Judge Decision P(out=e | {e,j})
                      # Depends on Judge Rank rJ
                      # P = exp(gamma * rJ_e) / (exp(gamma * rJ_e) + exp(gamma * rJ_j))
                      
                      logits_judge = gamma_save * rJ_t
                      p_judge_elim_e = jnp.exp(logits_judge[e]) / (jnp.exp(logits_judge[e]) + jnp.exp(logits_judge[j]))
                      
                      term = prob_bottom2 * p_judge_elim_e
                      
                      # Mask if j==e or j inactive
                      # Check valid: j != e AND j is active
                      valid = (j != e) & mask_t[j]
                      return val + jnp.where(valid, term, 0.0)

                 total_prob = jax.lax.fori_loop(0, n_pairs, body_fn, 0.0)
                 
                 # Mixture: (1-rho)*P_save + rho*P_unif
                 log_p_save = jnp.log(total_prob + 1e-10)
                 
                 n_active_count = jnp.sum(mask_t)
                 log_p_uniform = -jnp.log(n_active_count + 1e-10)
                 
                 log_p_mixed = jnp.logaddexp(
                     jnp.log(1.0 - rho) + log_p_save,
                     jnp.log(rho) + log_p_uniform
                 )
                 
                 numpyro.factor(f"elim_save_{t}", log_p_mixed)
                 
             else:
                 # Fallback to standard PL if multiple Elim (Double Elim usually just lowest votes)
                 # Or assumption: Double Elim is purely score based? 
                 # Let's stick to PL on 'b' (Badness).
                 pass # Go to PL block below
                 
        if rule_segment != 'rank_save' or len(eliminated_indices) > 1:
            # Standard PL on Badness b
            # Iterative removal: P(e1|Active) * P(e2|Active\{e1}) ...
            
            # Helper for PL
            # eliminated_indices is ordered? Usually standard PL doesn't care about order of set, 
            # but PL formula DOES imply an order (pick worst, then 2nd worst).
            # We treat the list in `elim_events` as the order of elimination?
            # Or we sum over permutations? 
            # The prompt says: "elimination set writing as (e_1...e_m)" and gives product formula.
            # This implies we treat the observed set as a sequence or just product of conditional probs.
            # If we don't know the order, technically should sum factorial. 
            # But usually we just assume one order or that they are "bottom m". 
            # "Select m worst simultaneous" -> Top-m Plackett Luce used for ranking?
            # Simplified: Just product of conditionals in stored order.
            
            current_mask = mask_t
            for e_idx in eliminated_indices:
                # Logit for e_idx
                logit_e = lambda_pl * b_t[e_idx]
                
                # LogSumExp over current active
                # Mask: current_mask
                logits_all = lambda_pl * b_t
                logits_all_masked = jnp.where(current_mask, logits_all, -1e9)
                lse = nn.logsumexp(logits_all_masked)
                
                # Factor: log P = logit_e - lse
                log_p_main = logit_e - lse
                
                # Mixture
                n_active_count = jnp.sum(current_mask)
                log_p_uniform = -jnp.log(n_active_count + 1e-10)
                
                log_p_mixed = jnp.logaddexp(
                     jnp.log(1.0 - rho) + log_p_main,
                     jnp.log(rho) + log_p_uniform
                )
                
                numpyro.factor(f"elim_pl_{t}_{e_idx}", log_p_mixed)
                
                # Remove e from mask for next term
                current_mask = current_mask.at[e_idx].set(False)

    # 5. Likelihood: Final Placement
    # final_ranking is list of indices [1st_place_idx, 2nd_place_idx, ...]
    # We model this as PL on 'Goodness' a = -b_{T}
    # Or explicitly on Goodness defined by b_{T}.
    # Formula says: a_{i, T} = -b_{i, T}.
    # PL on 'a'. Maximizing 'a' means minimizing 'b'.
    
    T_final = n_weeks - 1 # Last index
    b_final = b[T_final]
    a_final = -b_final
    
    current_mask_fin = active_mask[T_final]
    
    for rank_i, idx in enumerate(final_ranking):
        # idx is the person at rank_i (1st, 2nd...)
        # Prob(idx is best among remaining)
        
        logit_i = lambda_fin * a_final[idx]
        
        logits_all = lambda_fin * a_final
        logits_all_masked = jnp.where(current_mask_fin, logits_all, -1e9)
        lse = nn.logsumexp(logits_all_masked)
        
        numpyro.factor(f"final_rank_{rank_i}", logit_i - lse)
        
        # Remove from set
        current_mask_fin = current_mask_fin.at[idx].set(False)

