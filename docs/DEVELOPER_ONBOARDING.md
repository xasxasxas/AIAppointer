# 👩‍💻 Developer Onboarding Guide

Welcome to the **TalentSync AI** development team! This guide will help you get set up and understand the codebase.

## 1. Quick Start

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone
git clone https://github.com/your-org/TalentSyncAI.git
cd TalentSyncAI

# Virtual Env
python -m venv venv
# Windows: venv\Scripts\activate
# Linux: source venv/bin/activate

# Install
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run src/app.py
```

## 2. Project Structure

```
TalentSyncAI/
├── src/
│   ├── app.py                 # ENTRY POINT: Main Streamlit application
│   ├── predictor.py           # CORE: LightGBM inference logic
│   ├── semantic_engine.py     # CORE: SBERT & Search logic
│   ├── explainer.py           # CORE: SHAP interpretation
│   ├── markov_engine.py       # CORE: Career pattern analysis
│   ├── features_ltr.py        # Feature engineering pipeline
│   ├── xai_helpers.py         # UI helpers for XAI display
│   └── gantt_viz.py           # Visualization utilities
├── models/
│   ├── ltr/                   # Trained LightGBM models
│   └── all_constraints.json   # Role constraints (JSON)
├── data/                      # Dataset (CSV) - GitIgnored!
├── docs/                      # Documentation
└── scripts/                   # Training/Utility scripts
```

## 3. Key Workflows

### A. How prediction works (`predictor.py`)
1.  `app.py` sends `officer_row` to `predictor.predict()`.
2.  `predictor` generates a DataFrame of potential candidates (all roles).
3.  Constraints are applied (Rank/Branch filtering).
4.  LightGBM model scores valid candidates.
5.  Markov Engine boosts scores for recognized patterns.
6.  Top N results are returned with confidence scores.

### B. How Semantic Search works (`semantic_engine.py`)
1.  On startup, `SemanticAIEngine` loads SBERT model (lazy loaded).
2.  Index is built for `Appointment_history` and `Training_history`.
3.  Query is parsed for keywords (INCLUDE/EXCLUDE).
4.  Hybrid search combines keyword filtering + vector similarity.

## 4. Contributing
1.  Create a branch: `git checkout -b feature/amazing-feature`
2.  Make changes.
3.  Run the app to test: `streamlit run src/app.py`
4.  Commit and Push.

## 5. Troubleshooting
- **"Model not found"**: Run `python scripts/train_model.py` to regenerate models.
- **"SBERT download slow"**: First run takes time to download the model (~80MB).

Happy Coding! 🚀
