
# Deep Dive Analysis: Employee 200016 Case Study
**Version:** 1.0  
**Date:** 2025-12-13 09:20:00

## 1. Executive Summary
This document provides a comprehensive technical analysis of the AI prediction for **Employee 200016 (Jarek Janus)**. It explains why the model selected a specific role despite apparent mismatches and breaks down the exact feature contributions (SHAP values) that drove this decision.

**Key Insight**: The recommendation is driven by strong **Structural Alignment** (Branch and Title keywords) which outweighed the lack of **Historical Precedent**. The low raw score indicates this is a "best available" fit rather than a "perfect" fit.

---

## 2. Subject Profile
| Attribute | Value | Context |
| :--- | :--- | :--- |
| **Name** | Jarek Janus | |
| **Rank** | Lieutenant (jg) | Junior Officer |
| **Branch** | Tactical Systems | Technical/Combat Operations |
| **Current Role** | Div Officer CS Meridian / Post 1 | Held since Mar 2022 (~3.8 Years) |
| **Pool** | Galactic | Deployed |

---

## 3. The Prediction
**Predicted Role**: `Staff Officer SFHQ - Technical Policy`  
**AI Confidence**: **33.7%** (Top Prediction)  
**Raw Fit Score**: **-10.37** (Log-Odds)

### Why 33% Confidence?
The raw score of **-10.37** is objectively low (indicating many negative factors). However, the AI uses "Softmax" normalization. This means it compares -10.37 against the scores of other candidates (e.g., -11.0, -12.5). Since Jarek's score is the "least negative", he receives the highest relative probability (33%).

---

## 4. Feature Breakdown (SHAP Analysis)
The following table details exactly how the model calculated the score.

| Feature Label | Component | Value | SHAP Impact | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Base Value** | *Starting Point* | N/A | **-8.94** | The systematic average probability for *any* random officer-role pair. Most pairs are incompatible, so the baseline is very low. |
| **History Pattern** | `prior_title_prob` | 0.0% | **-2.89** | **Major Penalty**. Historical data shows 0 instances of a "Div Officer CS Meridian" moving directly to "Staff Officer SFHQ". The model penalizes this lack of precedent. |
| **Title Similarity** | `title_similarity` | 0.03 | **+1.21** | **Major Boost**. The word "Officer" in both titles provides a semantic link, suggesting functional similarity despite the lack of exact history. |
| **Total Service** | `years_service` | ~4 Yrs | **+0.77** | **Boost**. The candidate's tenure indicates readiness for a new assignment, positively influencing the score. |
| **Rank Eligibility** | `rank_match_exact` | 0 (No) | **-0.46** | **Penalty**. The target role typically requires `Lieutenant`, but the candidate is `Lieutenant (jg)`. The model applies a penalty for this "Reach" assignment. |
| **Pool Match** | `prior_pool_prob` | 0.02 | **-0.16** | **Penalty**. Moving from `Galactic` to `Near Earth` (SFHQ) is a pool change, which is less common than staying in-pool. |
| **Branch Match** | `branch_match` | 1 (Yes) | **+0.09** | **Boost**. The candidate is in `Tactical Systems`, and the target role aligns with this branch (`Technical Policy` falls under the Tactical/Ops umbrella). |

### Mathematical Reconstruction
$$ \text{Final Score} = \text{Base} (-8.94) + \sum \text{Features} (-1.43) = \mathbf{-10.37} $$

---

## 5. Decision Logic Interpretation
The AI selected this role because:
1.  **It kept him in his Branch**: A cross-branch move would have incurred a much larger penalty (e.g., -2.0). 
2.  **It found a Semantic Match**: "Staff Officer" is functionally closer to "Div Officer" than unrelated roles like "Chief Engineer".
3.  **Tenure Support**: His time in service pushed him slightly above other junior candidates.

**Ideal Match Scenario**: 
For Jarek to achieve a higher score (e.g., >80% confidence), the target role would need to be:
*   **Rank Appropriate**: A `Lieutenant (jg)` role (removing the -0.46 penalty).
*   **Historically Common**: A role that "Div Officers" frequently move to (removing the -2.89 penalty).
*   **In-Pool**: A role within the `Galactic` pool (removing the -0.16 penalty).

## 6. Conclusion
The prediction of **Employee 200016** demonstrates the AI's ability to maximize "Structural Fit" (Branch/Keywords) when "Historical Fit" is absent. While the absolute score is low due to the lack of history, it correctly identifies the most logical career step relative to completely unrelated options.
