# Task 2 Report: Fairness & Rule Sensitivity Analysis

## Executive Summary
We performed a counterfactual replay of all 34 seasons of DWTS, simulating what *would* have happened under alternative scoring rules ("Rank", "Percent", "Judges' Save"). By leveraging the latent vote shares ($v_{i,t}$) inferred in Task 1, we identified critical moments where the rulebook itself determined the winner, rather than the audience or judges alone.

## Key Findings

### 1. The "Percent Rule" is More Volatile but "Fairer"
-   **Reversal Rate**: If historical "Rank Rule" seasons (S1, S2, S28+) were replayed under the "Percent Rule", the champion would change in **~30%** of simulations.
-   **Season 2 Shock**: Season 2 shows a massive **46% probability of a different champion** under the Percent rule. The Rank rule compresses margins, potentially protecting a contestant who was mediocre in judges' scores but had strong (but not overwhelming) fan support, or vice versa.
-   **Sensitivity**: The Percent rule is more sensitive to "runaway favorites" (who get huge vote shares), whereas the Rank rule flattens a 50% vote share and a 20% vote share to just "Rank 1" vs "Rank 2".

### 2. The Judges' Save: A Safety Net or Drama Killer?
-   **Upset Reduction**: Introduction of the Judges' Save (simulated on early seasons) reduces the "Upset Rate" (elimination of a high-scoring couple) by **~15-20%**.
-   **Champion Stability**: In seasons like S32, adding the Save makes the outcome *more* predictable (Champion change prob < 15%), aligning the result closer to the "Judges' Choice".

### 3. "Robbed" Contestants
By comparing elimination weeks across rules, we identified contestants who were likely "robbed" by the specific mechanics of their season:
-   **Rank Rule Victims**: Contestants who had decent vote totals but got punished by the non-linear "Rank" conversion (e.g., being 2nd in votes gave them little advantage over 3rd, while a bad Judge score tanked them).
-   **Percent Rule Victims**: Contestants in "tight races" where a 1% difference in scores led to elimination, whereas the Rank rule would have saved them by treating them as equal to peers.

## Detailed Metrics (Sample)
| Season  | Rule Used | Prob. Champ Change (if Percent) | Prob. Champ Change (if Save) | Controversy (Drama D) |
| :------ | :-------- | :------------------------------ | :--------------------------- | :-------------------- |
| **S1**  | Rank      | 15%                             | 27%                          | 0.89 (High)           |
| **S2**  | Rank      | **46%**                         | 12%                          | 0.77                  |
| **S27** | Percent   | N/A (Baseline)                  | N/A                          | **1.02** (Extreme)    |
| **S29** | Rank+Save | 43% (if Percent)                | N/A                          | 0.79                  |

> **Note**: S27 (The "Bobby Bones" Season) shows extreme "Drama" ($D \approx 1.0$), confirming the historical controversy where audience votes violently disagreed with Judges' ranks.

## Artifacts Produced
-   `Results/task2_metrics.csv`: Full sensitivity table for all 34 seasons.
-   `Results/replay_results/`: Detailed week-by-week replay logs.
