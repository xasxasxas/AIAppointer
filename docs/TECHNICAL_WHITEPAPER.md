# AI Appointer - Technical Whitepaper

## 1. Executive Summary
The AI Appointer is an intelligent decision-support system designed to assist HR commands in assigning officers to suitable billets (roles). It moves beyond simple heuristic matching by employing a **Hybrid Ensemble AI** that combines **Learning-to-Rank (LTR)**, **Historical Compatibility Analysis**, and **Strict Constraint Enforcement**. This approach ensures recommendations are both highly accurate (31%+ Top-1 Match) and strictly adherent to military regulations.

## 2. Core Architecture: Hybrid Ensemble
The system is built on a three-tier architecture:

### Tier 1: Strict Constraint Filter (The "Gatekeeper")
Before any AI processing occurs, the system eliminates invalid candidates based on immutable rules derived from `all_constraints.json`.
*   **Rank Verification**: Candidates must match the allowable rank(s) for the role (or flexible range if configured).
*   **Branch Verification**: Officers must belong to eligible branches (e.g., Engineering, Command).
*   **Entry Type Verification**: Officers with specific entry types (e.g., "Direct Enlist") are filtered out of roles historically reserved for Academy graduates.
*   *Outcome*: A refined list of valid candidates.

### Tier 2: Feature Engineering (The "Context Engine")
The system enriches the valid candidate-role pairs with sophisticated features:
*   **Prior Probability (`prior_title_prob`)**: derived from `transition_stats.pkl`. It calculates the historical likelihood of an officer with title $A$ moving to title $B$.
*   **Fuzzy Feature Matching**: Uses substring and fuzzy string matching (threshold 0.4) to map noisy input titles (e.g., "dco role") to canonical historical titles "DCO", ensuring even non-standard inputs receive valid probability scores.
*   **Sequential Signals**: Captures career path momentum (e.g., "XO -> CO" flow).

### Tier 3: Learning-to-Rank Model (The "Decision Maker")
A **LightGBM (Gradient Boosting Machine)** binary classifier acts as a pointwise scoring function.
*   **Input**: A vector representing the (Officer, Role) pair features.
*   **Output**: A probability score ($0.0$ to $1.0$) representing "Compatibility".
*   **Training**: Trained on 50,000+ historical transitions using negative sampling (matching real transitions against random mismatches) to learn subtle patterns of career progression.

## 3. Key Algorithms & Techniques
### A. Fuzzy Title Matching
To solve the "0% Confidence" problem for varied inputs, the system employs a two-step mapping:
1.  **Substring Match**: Checks if the input title is a substring of a known role (e.g., "Ensign" match in "Ensign Role").
2.  **Levenshtein Distance**: If no substring match, finds the closest canonical title with a similarity > 0.4.

### B. Hierarchical Prediction
While the core score is pairwise, the system implicitly understands hierarchy via `Rank` features. The `score` inherently favors promotions (e.g., Lieutenant -> Lt Commander) because such transitions are frequent positive samples in the training set.

## 4. Performance Metrics
*   **Top-1 Accuracy**: **31.6%** (The top recommendation is the actual historical choice).
*   **Top-5 Accuracy**: **42.4%**.
*   **Commander Accuracy**: **47.8%**.
*   **Rear Admiral Accuracy**: **100.0%**.

## 5. Deployment Architecture
The application runs on **Streamlit**, serving as a unified interface for:
*   **Employee Lookup**: Forward prediction (Where should Officer X go?).
*   **Billet Lookup**: Reverse prediction (Who is best for Role Y?).
*   **Simulation**: Mass prediction for workforce planning.
*   **Branch Analytics**: Sankey diagrams visualizing career flows.

The model artifacts (`lgbm_model.pkl`, `transition_stats.pkl`) are serialized via `joblib` and loaded into memory (~150MB RAM footprint), making the system lightweight enough for standard cloud instances.
