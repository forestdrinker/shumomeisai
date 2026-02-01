
# Visual Design Plan: High-Impact Hero Figures (Nature/Science Style)

为了回答核心问题（Q1-Q4）并达到“美观吸睛”的效果，我们设计了以下 4 张“Hero Figures”。设计风格对标 *Nature/Science*：简约、高信噪比、使用语义化配色（Semantic Color Mapping）。

## 🎨 Aesthetic Guidelines (设计美学)
- **Palette**: 
  - **Certainty/Probabilistic**: `Viridis` or `Rocket` (Continuous).
  - **Categorical (Judge vs Fan)**: `RdBu` (Red=Judge, Blue=Fan) or `PutOr` (Purple/Orange).
  - **Pareto**: `Grey` for dominated points, `Crimson` for Pareto Front, `Gold` for Recommendations.
- **Font**: Sans-serif (Arial/Helvetica), Size 10-12pt for axis, 14pt for titles.
- **Elements**: 
  - Thin grid lines (alpha=0.3).
  - Translucent confidence bands (alpha=0.2).
  - Minimalist spines (remove top/right).

---

## 🌊 Figure 1: The Hidden Current (Rank-Flow Divergence Chart)
> **Question**: "What is the 'True' popularity hidden beneath the Judges' scores? How much did they disagree?"

- **Visual Form**: **Probabilistic Bump Chart (Dual-Layer Rank Flow)**
- **X-Axis**: Week ($t$)
- **Y-Axis**: Rank (1 = Top, N = Bottom). *Inverted Y-axis.*
- **Data Source**: `Results/posterior_samples/season_{s}.npz` + `processed/panel.csv`
- **Information Density (High)**:
  1.  **Fan Rank (The Current)**: 
      - **Position**: Median Rank of the posterior vote share.
      - **Thickness**: 95% CI of the Rank (Visualizes Uncertainty width).
      - **Color**: Candidate Identity (Consistent) or Certainty intensity.
  2.  **Judge Rank (The Surface)**: 
      - **Mark**: Sharp, high-contrast dots/lines overlaying the flow.
      - **Insight**: The vertical graphical gap between the Dot (Judge) and the Ribbon (Fan) is the **"Controversy Gap"**.
  3.  **Elimination Events**: 
      - **Mark**: "X" or "Skull" icon at the end of a ribbon.
      - **Context**: If an "X" appears on a high-ranking ribbon -> **Shock Elimination**.
- **Visual Metaphor**: "Judges are the visible rocks; Fans are the hidden current moving around them."
- **Why it works**: Far denser than a simple ribbon. It simultaneously shows **Popularity Evolution**, **Judge-Fan Disagreement**, **Inference Uncertainty**, and **Survival Dynamics** in a single view.

## 🎻 Figure 2: The Violin of Bias (Task 2: Rule Comparison)
> **Question**: "Does the Percent Rule favor fan-favorites more than the Rank Rule?"

- **Visual Form**: **Split Violin Plot (Raincloud Plot)**
- **X-Axis**: Season Group (e.g., Early vs Late) or Rule Type.
- **Y-Axis**: Fan Bias Metric ($\rho_{Fan} - \rho_{Judge}$) or Survival Advantage.
- **Data Source**: `Results/task2_metrics.csv`
- **Key Elements**:
  - **Left Half**: Rank Rule distribution.
  - **Right Half**: Percent Rule distribution.
  - **Inner Quartiles**: Box plot inside the violin.
  - **Color Code**: Blue (Percent), Grey (Rank).
- **Why it works**: Shows the *distribution* of fairness, not just averages. Proves if one rule is consistently more biased.

## 🐝 Figure 3: The Forest of Influence (Task 3: Attribution)
> **Question**: "Is it the Star or the Partner? Who drives the votes?"

- **Visual Form**: **Coefficient Forest Plot (with Bayes Intervals)**
- **Y-Axis**: Predictor Names (sorted by magnitude).
- **X-Axis**: Effect Size ($\beta$ or Partial Dependence).
- **Data Source**: `Results/task3_baseline_lmm_summary.txt` (Baseline) or `task3_analysis/shap_values.csv` (GBDT).
- **Key Elements**:
  - **Points**: Point estimate of effect.
  - **Error Bars**: 95% CI (from LMM or Bootstrap SHAP).
  - **Color**: Red (Judge Factors: Technique), Blue (Fan Factors: Charisma/Partner).
  - **Split**: Show "Judge Model" coeffs on Left, "Fan Model" coeffs on Right.
- **Why it works**: Directly compares the *drivers* of two different scores.

## 📉 Figure 4: The Pareto Trade-off (Task 4: Optimization)
> **Question**: "Can we make everyone happy? No, but here is the best trade-off."

- **Visual Form**: **Multi-Objective Scatter Plot**
- **X-Axis**: Viewer Alignment (Obj_F).
- **Y-Axis**: Judge Alignment (Obj_J) or Drama (Obj_D).
- **Data Source**: `Results/task4_pareto_front.csv`.
- **Key Elements**:
  - **Grey Dots**: All evaluated trial rules (LHS samples).
  - **Red Line**: The **Pareto Frontier** (Non-dominated methods).
  - **Gold Star**: The **Knee Point** (Recommendation).
  - **Bubble Size**: Robustness (Obj_R) or Drama.
- **Why it works**: Visually proves "Optimality". The standard chart for Mechanism Design.

