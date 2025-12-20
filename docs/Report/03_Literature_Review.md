# 2. Literature Review

## 2.1 Recommender Systems in HR
The application of recommender systems to Human Resources (HR) has gained traction in recent years. Traditional approaches largely utilize **Content-Based Filtering** (CBF), matching candidate profiles to job descriptions using TF-IDF or simple keyword overlapping (Malinowski et al., 2006). While effective for exact matches, these systems struggle with the "vocabulary gap" problem where different terms imply similar skills.

## 2.2 Learning-to-Rank (LTR)
**Learning-to-Rank** has emerged as a superior paradigm for candidate selection compared to standard classification. Burges et al. (2005) demonstrated with RankNet that optimizing for pair-wise preferences yields better ranking results than point-wise regression. In the HR domain, LTR allows the model to learn the *relative* suitability of Candidate A vs. Candidate B for a specific role, rather than assigning an arbitrary "suitability score" in isolation.

## 2.3 Sequential Modeling & Career Trajectories
Modeling career paths requires understanding temporal dependencies. Determining the next role depends heavily on the *sequence* of prior roles. **Markov Chains** have been classically used to model such state transitions (Gagniuc, 2017). More recently, Meng et al. (2019) applied Long Short-Term Memory (LSTM) networks to career path prediction. However, for datasets of moderate size (<100k), higher-order Markov Chains often provide a better balance of performance and interpretability compared to deep neural networks.

## 2.4 Semantic Search & Embeddings
The advent of Transformer models like BERT (Devlin et al., 2018) revolutionized text understanding. **Sentence-BERT (SBERT)** (Reimers & Gurevych, 2019) optimized BERT for generating semantically meaningful sentence embeddings, enabling cosine similarity calculations that capture conceptual relatedness. This is crucial for matching diverse job titles (e.g., "Software Engineer" vs. "Application Developer") that strictly keyword-based systems might miss.

## 2.5 Explainable AI (XAI)
As AI systems are deployed in high-stakes domains like hiring, transparency is paramount. Lundberg & Lee (2017) introduced **SHAP (SHapley Additive exPlanations)**, a game-theoretic approach to feature attribution. SHAP values have become the gold standard for interpreting tree-based models (like XGBoost and LightGBM), providing both global feature importance and local instance-level explanations, which we heavily utilize in TalentSync AI.
