# Explainable AI (XAI) Implementation Guide for AI Appointer

**Version**: 1.0  
**Date**: 2025-12-13  
**Scope**: Technical & Operational Guide

---

## 1. Introduction: The "Black Box" Problem
In traditional AI, a model gives a score (e.g., "85% Match"), but users don't know *why*. This is dangerous for HR applications where fairness and logic are paramount. 

**Explainable AI (XAI)** solves this by treating the prediction as a "bill" that needs to be split among the participants. If the total bill is $100, who contributed what?
*   *Did the "Rank" feature pay $50?*
*   *Did "Bad History" reduce the bill by -$20?*

We use a Nobel-prize winning mathematical method called **SHAP (SHapley Additive exPlanations)** to calculate these contributions exactly.

---

## 2. Deep Dive: Understanding the Baseline (Base Value)
You will often see a large gray bar at the start of the visualizations labeled **"Baseline"** or **"Base Value"**. This is a critical concept to understand.

### What is it?
The Base Value is the **Average Score** of the model if it knew *nothing* about the specific candidate except that they exist in the universe of possible recommendations.
*   **Mathematical Definition**: It is the expected value $E[f(x)]$ of the model output over the entire training dataset.
*   **In Context**: Since there are thousands of possible roles and most officers are not a fit for most roles, the "average chance" of a random match is extremely low. This is why the Base Value is usually a negative number (e.g., -8.94).

### Where is it used?
1.  **Waterfall Chart (Start Point)**: It is the anchor. Every prediction starts at -8.94. If a candidate is perfect, positive features (Rank, Branch) add up from this hole to reach a positive score.
2.  **Force Plot (Center Line)**: In the "tug-of-war" visualization, the center point where the red and blue forces meet is this Base Value. It represents the "neutral" state.

### Why is it important?
It standardizes the explanations. Instead of asking "Why is this score 80%?", SHAP asks "Why is this score *higher than the average score*?". This difference is what the feature bars represent.

---

## 2. The Factors: What the AI Specificall Looks At
The AI Appointer uses a "Learning to Rank" (LTR) model. It doesn't just memorize rules; it learns patterns. Be aware of exactly what it is evaluating:

### A. Structural Factors (Hard Constraints)
| Feature Name | What it Measures | Why it Matters | Impact Trend |
| :--- | :--- | :--- | :--- |
| `rank_match_exact` | Does officer rank match role requirement? | Prevents Ensigns from becoming Admirals. | **High Negative** if mismatched. |
| `branch_match` | Does officer branch match target branch? | Prevents Engineers from taking Legal roles. | **High Positive** if matched. |
| `years_service` | Total time in service. | Proxy for seniority and experience. | Generally **Positive** as it increases. |

### B. Historical Factors (Pattern Matching)
| Feature Name | What it Measures | Why it Matters | Impact Trend |
| :--- | :--- | :--- | :--- |
| `prior_title_prob` | **The "Precedent" Score**. % of time this specific career move happened in the past. | If 0%, it's a "Pioneering" move. If 50%, it's "Standard". | **Dominant Factor**. High % = Huge Boost. |
| `prior_pool_prob` | Frequency of moving between these two Pools (e.g., Galactic -> Shore). | Captures flow dynamics (e.g., sea-to-shore rotation). | Moderate impact. |
| `title_similarity` | Text overlap between current and target title. | Catches "fuzzy" matches (e.g., "Ops Officer" $\approx$ "Operations Manager") even if history is missing. | **Safety Net**. Boosts scores when history is silent. |

---

## 3. Visual Tutorial: Reading the Charts
The app provides four key visualizations. Here is how to read them.

### 🔍 Local Explanation (Single Person)
Used in **Billet Lookup** and **Experiments Lab**.

#### 1. The Waterfall Chart (Decision Path)
*   **Goal**: See how we got from "Average" to "Generic Score".
*   **Reading It**:
    *   **Center Line**: The Baseline (Average Score).
    *   **Green Bars**: Features pushing the score **UP**.
    *   **Red Bars**: Features pushing the score **DOWN**.
    *   **End Value**: The final Log-Odds score (which converts to %).
*   **Example**: If you see a giant **Red Bar** for `prior_title_prob`, it means "We've never seen this move before, so we are skeptical."

#### 2. The Force Plot (Tug-of-War)
*   **Goal**: Visualize the balance of power.
*   **Reading It**:
    *   Think of it as a rope. Red team pulls left (lower score), Blue team pulls right (higher score).
    *   The width of the bar represents the strength of the feature.
    *   This is useful for seeing if a single massive negative is canceling out five small positives.

### 🌍 Global Explanation (System Wide)
Used in **Branch Analytics**.

#### 3. The Beeswarm Plot (Bias Detector)
*   **Goal**: See how a feature affects *everyone*.
*   **Reading It**:
    *   **Y-Axis**: Features (Rank, Branch, etc.).
    *   **X-Axis**: Impact (Left = Negative, Right = Positive).
    *   **Color**: Feature Value (Red = High, Blue = Low).
*   **Key Insight Example**: 
    *   Look at `years_service`.
    *   If the **Red** dots (High Years) are all on the **Right** (Positive Impact), it proves the model favors seniority.
    *   If you see High Values (Red) on the Left, the model might be ageist (penalizing seniority).

---

## 4. Real-World Case Study: Officer 200016
Let's analyze a real prediction to see the math in action.

**Subject**: Lt(jg) Jarek Janus (Tactical Systems)
**Proposed Role**: Staff Officer SFHQ - Technical Policy

### The "Bill" Breakdown (SHAP Values)
The mathematical formula for his score is: $$ \text{Base Value} + \sum (\text{Feature Impacts}) = \text{Final Log-Odds} $$

| Item | Value | Logic |
| :--- | :--- | :--- |
| **Baseline** | **-8.94** | The starting probability is near zero because specific roles are rare. |
| **History Pattern** | **-2.89** | *Penalty*. No one has ever moved from "Div Officer" to "Technical Policy". |
| **Rank Mismatch** | **-0.46** | *Penalty*. Role wants a full Lieutenant. He is Lt(jg). |
| **Title Keywords** | **+1.21** | *Boost*. "Officer" and "Technical" appear in his history. |
| **Total Service** | **+0.77** | *Boost*. He has good tenure stats. |
| **Branch Match** | **+0.09** | *Boost*. He is in the correct Tactical branch. |
| **FINAL SCORE** | **-10.37** | (Sum of the above) |

### Interpreting the Outcome
*   **Raw Score (-10.37)**: This looks like a negative number, but in probability math (Log-Odds), it's relative.
*   **Final % (33.7%)**: When compared to other impossible candidates (who scored -15 or -20), -10.37 was actually the **strongest score**.
*   **Conclusion**: The AI recommended him **NOT** because he was a perfect historical fit, but because his **Keywords** and **Branch** made him the "Least Bad" option among available candidates. This shows the AI can creatively fill gaps using semantic matching (`title_similarity`).

---

## 5. How to Use This for Debugging
1.  **If a Score is too Low**: Check the **Waterfall**. Is `prior_title_prob` huge and red? 
    *   *Fix*: This might be a missing data issue or a truly novel career move.
2.  **If a Candidate is "Qualified" but Ranked Low**: Check `rank_match_exact`.
    *   *Fix*: If it's Red, the strict rank requirement is blocking them. Use the **Rank Flexibility** slider to relax this.
3.  **If Bias is Suspected**: Go to **Branch Analytics**. Filter by the branch in question. Look at the **Beeswarm**.
    *   Does `branch_match` have a surprisingly low impact? The model might be confused about that branch's roles.

---
**Prepared by**: AntiGravity AI Team
