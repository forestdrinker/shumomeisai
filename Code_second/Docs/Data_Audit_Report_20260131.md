# Math Modeling Coach Audit Report
**Date:** 2026-01-31
**Subject:** Critical Review of Project Output Data
**Auditor:** Math Modeling Coach (Strict Mode)

---

## 🟥 Overall Assessment: UNACCEPTABLE FOR SUBMISSION

Your current data outputs demonstrate a **fundamental lack of professional rigor**. While the *pipelines* seem to be running, the results are plagued by amateurish formatting, valid statistical concerns, and distinct lack of quality control. If this were a competition submission, you would be penalized heavily for "Presentation Quality".

You are treating floating-point numbers like garbage dumps and neglecting basic data hygiene (clean categories, proper CSV formulation).

---

## Task 1 Analysis: Vote Inference & Calibration
**File Audited:** `task1_metrics.csv`, `MainFig2_Validation.csv`
**Score:** 4/10

### 🚩 Critical Issues
1.  **Calibration "Perfect" Score (Over-Conservative CI):**
    *   **Observation:** `coverage_90` is exactly `1` for entire blocks of early weeks.
    *   **Verdict:** This is **statistically implausible** for a "90%" interval unless your intervals are absurdly wide. A perfect 1.0 means your model is **under-confident**. You are playing it too safe, making the "prediction" uselessly vague.
2.  **Floating Point Garbage:**
    *   **Observation:** `brier` score like `0.06567992975142782`.
    *   **Verdict:** **Significant Digits, Student!** Do not output 16 decimal places. No judge trusts `0.065679929...`. Round to 3-4 significant figures. It clutters the file size and looks messy.

### ⚠️ Warnings
*   **Data Consistency:** Line 39 in validation has `n_elim=2`. Ensure your logic handles double eliminations correctly and clearly marks them.
*   **Missing Metadata:** No units in headers. Is `avg_ci_width` in percentage points or raw vote share? Explicitly state unit (e.g., `avg_ci_width_pct`).

---

## Task 2 Analysis: Rule Replay & Metrics
**File Audited:** `task2_metrics.csv`
**Score:** 5/10

### 🚩 Critical Issues
1.  **Suspense Metric Definition:**
    *   **Observation:** `suspense_H` (Entropy) frequently exceeds 1.0 (e.g., `1.129`).
    *   **Verdict:** If this is Shannon Entropy (bits/nats), fine. But if this is meant to be a normalized [0,1] metric for a general audience, **it is broken**. You must clarify the base ($log_2$ vs $log_{10}$ vs $log_e$) or normalize it. A value > 1 confuses non-expert readers expecting a probability-like scale.
2.  **Amateur String Formatting:**
    *   **Observation:** `rule` column uses `rank`, `rank_save`, `percent`.
    *   **Verdict:** Inconsistent casing. Use Title Case (`Rank`, `Percent`) or keep it strictly lowercase codes. Consistency is key.
3.  **Floating Point mess:**
    *   **Observation:** `0.8369999999999997`.
    *   **Verdict:** This is a classic floating-point error. Round your outputs! (`round(x, 4)`).

---

## Task 3 Analysis: Attribution
**File Audited:** `task3_lmm_fan_coeffs_aggregated.csv`
**Score:** **2/10 (FAIL)**

### 🚩 Critical Issues
1.  **Data Hygiene Failure (Duplicate Categories):**
    *   **Observation:** You have distinct rows for `Social Media Personality` and `Social media personality`.
    *   **Verdict:** **INEXCUSABLE.** This splits your sample size, biases the coefficients for *both* resulting rows, and invalidates your statistical inferences for this group. This is basic "Data Cleaning 101".
2.  **Spelling Errors:**
    *   **Observation:** `Beauty Pagent` (missing 'a').
    *   **Verdict:** Typos in final data categories destroy credibility. It suggests you didn't even look at the file before creating a plot.

---

## Task 4 Analysis: Policy Optimization
**File Audited:** `task4_pareto_front.csv`
**Score:** 3/10

### 🚩 Critical Issues
1.  **Malformed CSV Structure:**
    *   **Observation:** The `gamma` column has data in some rows but is empty/missing in others (resulting in jagged lines like `...,0.0175,` vs `...,0.028,1.559`).
    *   **Verdict:** While Python *might* handle this, standard CSV parsers will choke or misalign columns. If `gamma` is unused (e.g. when `save_flag=0`), put `NaN`, `0`, or `N/A`. Do not leave a trailing comma or ragged edge.
2.  **Parameter Precision:**
    *   **Observation:** `b: 9.736920152922979`.
    *   **Verdict:** Optimization parameters do not need nanometer precision. Round them.

---

## 🛠️ Required Actions (Immediate Fixes)

1.  **Refactor Data Export Scripts:** Apply `.round(4)` to ALL float columns before `to_csv`.
2.  **Fix Task 3 Cleaning:** Normalize industry strings (`.str.title().str.strip()`) and fix typos (`Pagent` -> `Pageant`) *before* model training. Re-run Task 3.
3.  **Fix Task 4 CSV:** Ensure `gamma` column is filled with `None`/`NaN` instead of empty strings, or drop it if irrelevant for specific rows (better to fill for consistent schema).
4.  **Validate Task 1 Coverage:** Investigate why coverage is perfect. If true, your error bars are too big. Tighten the model variance ($\sigma_u$).

**Get to work.**
