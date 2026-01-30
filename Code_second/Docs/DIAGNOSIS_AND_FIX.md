# Task 1 MCMC 收敛问题诊断与修复

## 📋 问题总结

你的输出显示 **Season 29 严重不收敛**：
- R-hat 最高达 **2.04**（标准要求 <1.05）
- n_eff 最低仅 **3.19**（标准要求 >100）
- Divergences: **18**（应该为 0）

这是 **模型设计 + 编程实现的复合问题**，不仅仅是采样量不足。

---

## 🔍 根因分析

### 问题1：数值溢出（最严重 ⚠️）

**原代码：**
```python
lambda_pl = 10.0   # 淘汰尖锐度
gamma_save = 5.0   # 评委偏好强度
```

**为什么有问题：**
- `b` (badness rank) 范围约 1-14
- `logit = lambda_pl * b = 10 * 14 = 140`
- `exp(140) = Infinity` → **数值溢出**

同样，`gamma_save * rJ = 5 * 14 = 70` → `exp(70)` 也会溢出。

### 问题2：soft-rank 温度太低

**原代码：**
```python
kappa_F = 0.05  # vote share 范围 ~0.1
```

**为什么有问题：**
- 当两人票差 0.05 时：`sigmoid(0.05/0.05) = 0.73`
- 当票差 0.1 时：`sigmoid(0.1/0.05) = 0.88`
- soft-rank 几乎变成 hard-rank，**梯度消失**

### 问题3：Save 逻辑的 fori_loop

**原代码：**
```python
total_prob = jax.lax.fori_loop(0, n_pairs, body_fn, 0.0)
```

**为什么有问题：**
- `fori_loop` 内部累加 `exp(...)` 值
- 循环中的梯度计算不稳定
- 容易出现数值下溢（多个小概率相乘）

### 问题4：mask 值太极端

**原代码：**
```python
huge_neg = -1e9
u_masked = jnp.where(active_mask, u, huge_neg)
```

**为什么有问题：**
- `exp(-1e9) = 0` 是对的，但 `-1e9` 在某些计算中可能导致数值问题
- 梯度在边界处可能不稳定

---

## ✅ 修复方案

### 修复1：降低超参数

```python
# 原来
lambda_pl = 10.0
gamma_save = 5.0

# 修复后
lambda_pl = 2.0   # 降低5倍
gamma_save = 1.0  # 降低5倍
```

**原理**：更平滑的似然函数让 MCMC 更容易探索后验空间。

### 修复2：自适应 kappa

```python
def soft_rank_adaptive(scores, mask, base_kappa=0.1):
    # 计算分数标准差
    std_score = jnp.std(scores[mask])
    
    # kappa 至少是 std 的 30%，保持 soft
    kappa = jnp.maximum(base_kappa, 0.3 * std_score)
    
    # ... 其余计算
```

**原理**：kappa 自动适应数据尺度，避免 hard-rank 导致的梯度消失。

### 修复3：向量化 Save 逻辑 + Log-space 计算

```python
# 原来（有问题）
total_prob = jax.lax.fori_loop(0, n_pairs, body_fn, 0.0)
numpyro.factor(..., jnp.log(total_prob + 1e-10))

# 修复后
log_contribution = log_p_bottom2 + log_p_judge_elim_e  # (N,)
log_total_prob = nn.logsumexp(log_contribution_masked)
numpyro.factor(..., log_total_prob)
```

**原理**：
1. 向量化避免循环梯度问题
2. Log-space 计算避免 exp 溢出/下溢

### 修复4：温和的 mask 值

```python
# 原来
huge_neg = -1e9

# 修复后  
huge_neg = -30.0  # exp(-30) ≈ 0，但数值更稳定
```

---

## 📁 修复后的文件

1. **task1_model_v2.py**：修复后的模型定义
2. **task1_runner_v2.py**：修复后的运行器

### 使用方法

```bash
# 测试运行（Season 1 和 29）
python task1_runner_v2.py --test-run

# 正式运行所有赛季
python task1_runner_v2.py --all --warmup 1000 --samples 2000

# 如果还有问题，使用简化模型调试
python task1_runner_v2.py --seasons 29 --simple
```

---

## 📊 预期改善

| 指标 | 修复前 (S29) | 预期修复后 |
|------|--------------|------------|
| max(R-hat) | 2.04 | < 1.05 |
| min(n_eff) | 3.19 | > 100 |
| Divergences | 18 | 0 |

---

## 🔧 如果仍有问题

### 进一步调优选项

1. **继续降低 lambda_pl**：
   ```python
   lambda_pl = 1.0  # 甚至更小
   ```

2. **增加采样量**：
   ```bash
   python task1_runner_v2.py --warmup 2000 --samples 4000
   ```

3. **使用多链**：
   修改 `num_chains=4`，更好地诊断收敛。

4. **对特别困难的赛季使用 SVI warm start**：
   ```python
   from numpyro.infer import SVI, Trace_ELBO
   from numpyro.infer.autoguide import AutoNormal
   
   # 先跑 SVI
   guide = AutoNormal(model_fn)
   svi = SVI(model_fn, guide, ...)
   svi_result = svi.run(...)
   
   # 用 SVI 结果初始化 MCMC
   init_params = guide.median(svi_result.params)
   ```

---

## 💡 关键洞察

这个问题的本质是：

> **贝叶斯隐变量反演是一个"可识别性弱"的问题**

观众票 `v` 从淘汰结果反推，存在多解。当：
- 似然函数太尖锐 → 后验多峰 → MCMC 难以跨峰采样
- soft-rank 太硬 → 梯度消失 → NUTS 无法有效导航

修复的核心思路是 **让后验更平滑**，让 MCMC 更容易探索，同时通过贝叶斯先验提供正则化。
