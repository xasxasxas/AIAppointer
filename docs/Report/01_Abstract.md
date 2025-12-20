# Abstract

**Objective**: This study presents **TalentSync AI**, an intelligent decision-support system designed to optimize internal talent mobility within large hierarchical organizations. The primary objective was to overcome the limitations of manual billet assignment by developing a data-driven system capable of predicting the most suitable next roles for personnel with high accuracy and adherence to strict organizational constraints.

**Methods**: We utilized a dataset of 10,000 officer profiles containing career history, training records, and current appointments. The system employs a **Hybrid Ensemble Architecture** combining a **LightGBM Learning-to-Rank (LTR)** model for candidate scoring, a **Sentence-BERT (SBERT)** engine for semantic resume matching, and a **Markov Chain** engine for sequential pattern recognition. A custom Explainer module based on **SHAP (SHapley Additive exPlanations)** was integrated to provide transparent, context-aware reasoning for every recommendation.

**Results**: The system achieved a **Top-1 Accuracy of 60.0%** and an **AUC of 99.98%** on the validation set, significantly outperforming baseline heuristic methods. Usage of Markov features contributed a 23.3% improvement in accuracy. Inference time was maintained below **100ms** per prediction, ensuring real-time usability.

**Conclusion**: TalentSync AI demonstrates that combining gradient boosting with sequential modeling and semantic search yields a robust, explainable, and highly accurate tool for HR decision-making, offering a scalable solution for modern workforce management.

**Keywords**: Learning-to-Rank, Talent Mobility, Explainable AI, Markov Chains, Semantic Search, HR Tech.
