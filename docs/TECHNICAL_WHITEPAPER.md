# Technical Whitepaper: TalentSync AI

**Version:** 4.1  
**Date:** December 2025

## 1. Executive Summary

**TalentSync AI** is an advanced Learning-to-Rank (LTR) system designed to optimize internal talent mobility. Unlike traditional HR systems that rely on manual matching or simple keyword filters, TalentSync AI utilizes a gradient-boosted decision tree approach augmented by Markov Chain sequential modeling to predict the most probable and suitable next roles for officers.

## 2. System Performance (v4.0)

The latest release achieves state-of-the-art performance on the validation dataset:

-   **AUC (Area Under Curve)**: 99.98%
-   **Top-1 Accuracy**: 60.0% (+23.3% over baseline)
-   **Top-3 Accuracy**: 82.4%
-   **Median Rank**: 1.0 (The correct role is overwhelmingly the top prediction)
-   **Inference Speed**: <100ms per prediction

## 3. Key Innovations

### A. True Semantic AI Search
Restored in v4.1, this engine utilizes **Sentence-BERT (SBERT)** (`all-MiniLM-L6-v2`) to perform semantic vector-based matching. It allows users to search for "Officers with leadership experience in engineering" and find relevant candidates even if they don't have the exact keyword matches. This operates alongside a traditional hybrid keyword filter.

### B. Context-Aware Explainability (XAI)
TalentSync AI provides not just a score, but a rationale. The **SHAP-based XAI module** dynamically adjusts its explanations based on the context (e.g., highlighting "Leadership Experience" for a Command role vs "Technical Skills" for an Engineering role).

### C. Sequential Pattern Recognition
The system identifies 2nd and 3rd-order career patterns (e.g., *Role A -> Role B -> Role C*) using a custom Markov Engine, boosting prediction confidence for established career paths.

## 4. Core Architecture: Hybrid Ensemble
The system is built on a three-tier architecture:

### Tier 1: Strict Constraint Filter (The "Gatekeeper")
Before any AI processing occurs, the system eliminates invalid candidates based on immutable rules:
*   **Rank Verification**: Candidates must match the allowable rank(s).
*   **Branch Verification**: Officers must belong to eligible branches.
*   **Entry Type Verification**: Specific entry filters.
*   *Outcome*: A refined list of valid candidates.

### Tier 2: Feature Engineering (The "Context Engine")
The system enriches the valid candidate-role pairs with sophisticated features:
*   **Prior Probability (`prior_title_prob`)**: derived from `transition_stats.pkl`.
*   **Fuzzy Feature Matching**: Uses substring and fuzzy string matching (threshold 0.4) to map noisy input titles.
*   **Sequential Signals**: Captures career path momentum.

### Tier 3: Learning-to-Rank Model (The "Decision Maker")
A **LightGBM (Gradient Boosting Machine)** binary classifier acts as a pointwise scoring function.
*   **Input**: A vector representing the (Officer, Role) pair features.
*   **Output**: A probability score ($0.0$ to $1.0$) representing "Compatibility".

## 5. Visualization & UI Modules
The application runs on **Streamlit** with the following key modules:

1.  **Employee Lookup**: Forward prediction with SHAP Waterfall charts.
2.  **Billet Lookup**: Reverse prediction with ranked candidate lists.
3.  **Semantic AI Search**: Natural language search for officers and billets.
4.  **Analytics & Explorer**:
    *   **Network Graph**: Interactive force-directed graph of career paths.
    *   **Gantt Timeline**: Comprehensive temporal view of officer careers and billet occupancy.
    *   **Sankey Diagrams**: Visualization of branch flows.
5.  **Simulation**: What-if analysis for hypothetical scenarios.
6.  **Admin Console**: Model retraining and deployment management.

## 6. Deployment
The model artifacts (`lgbm_model.pkl`, `transition_stats.pkl`) are serialized via `joblib` and loaded into memory (~150MB RAM footprint). The system supports:
-   **Online**: Streamlit Cloud / Corporate Server.
-   **Offline/Air-Gapped**: Fully contained deployment using pre-downloaded wheels (see `DEPLOYMENT_GUIDE.md`).
