# AI Appointer - Intelligent Career Progression System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## \ud83d\ude80 Overview

**AI Appointer** is an intelligent career progression recommendation system that uses machine learning to predict optimal next roles for officers based on their career history, skills, and organizational patterns. The system leverages **LightGBM Learning-to-Rank (LTR)** models enhanced with **2nd and 3rd-order Markov Chain** sequential modeling to capture temporal career progression patterns.

## \u2728 Key Features

- \ud83d\udd2e **Predictive Career Recommendations**: AI-powered suggestions for next best roles using LightGBM LTR
- \ud83d\udd04 **Markov Chain Sequential Modeling**: Captures 2nd and 3rd-order career progression patterns
- \ud83d\udd0d **Explainable AI (XAI)**: SHAP-based explanations showing how predictions are made
- \ud83c\udfaf **Billet Lookup**: Reverse search to find best candidates for specific roles
- \ud83d\udcca **Interactive Analytics**: Career flow visualization, appointment timelines, and organizational insights
- \u26a1 **Real-time Predictions**: Sub-100ms inference time
- \ud83d\udee1\ufe0f **Constraint Compliance**: 100% adherence to rank, branch, and role requirements

## \ud83d\udcca Performance

- **Model Accuracy**: 99.98% AUC
- **Top-1 Accuracy**: 60% (23.3% improvement with Markov features)
- **Top-3 Accuracy**: 60% (16.7% improvement)
- **Median Rank**: 1.0 (78% improvement)
- **Inference Speed**: <100ms per prediction

## \ud83d\udee0\ufe0f Technology Stack

- **ML Framework**: LightGBM (Learning-to-Rank)
- **XAI**: SHAP (SHapley Additive exPlanations)
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Sequential Modeling**: Custom Markov Chain engine

## \ud83d\udcbb Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AIAppointer.git
cd AIAppointer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run src/app.py
```

4. **Access the app**
Open your browser and navigate to `http://localhost:8501`

## \ud83d\udcda Documentation

- **[User Guide](docs/USER_GUIDE.md)**: Complete guide for using all features
- **[Architecture](docs/ARCHITECTURE.md)**: System design and technical details
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Instructions for deploying to production
- **[XAI Documentation](docs/XAI_DOCUMENTATION.md)**: Understanding model explanations

## \ud83c\udfaf Features Overview

### 1. Employee Lookup
Find the best next role for an officer based on:
- Career history and progression patterns
- Skills and training
- Branch and rank requirements
- Historical precedents

### 2. Billet Lookup
Find the best candidates for a specific role:
- Ranked by fit score
- Career pattern indicators
- Detailed explanations for each candidate

### 3. Simulation Mode
Experiment with hypothetical officer profiles:
- Design custom career paths
- Test "what-if" scenarios
- Analyze AI predictions

### 4. Analytics & Explorer
Explore organizational data:
- Career flow visualization
- Temporal appointment timelines (Gantt chart)
- Organizational structure browser
- Statistical insights

### 5. Admin Console
Model management and training:
- Retrain models with new data
- Hyperparameter optimization
- Model versioning and rollback

## \ud83d\udd2c How It Works

### Sequential Modeling with Markov Chains

The system uses enhanced Markov Chain modeling to capture career progression patterns:

1. **1st-order**: Direct role-to-role transitions
2. **2nd-order**: Two-step career sequences
3. **3rd-order**: Three-step career sequences

These patterns are integrated as features in the LTR model, significantly improving prediction accuracy.

### Explainable AI

Every prediction comes with:
- **SHAP Waterfall Chart**: Shows how each factor influenced the score
- **Career Progression Pattern**: Displays recognized Markov sequences
- **Historical Precedents**: Officers who made similar transitions
- **Feature Breakdown**: Detailed explanation of all contributing factors

## \ud83d\udcca Model Performance

The enhanced model with Markov features shows significant improvements:

| Metric | Old Model | New Model | Improvement |
|--------|-----------|-----------|-------------|
| Top-1 Accuracy | 36.7% | **60.0%** | +23.3% |
| Top-3 Accuracy | 43.3% | **60.0%** | +16.7% |
| Median Rank | 4.5 | **1.0** | 78% |
| AUC | 99.96% | **99.98%** | +0.02% |

## \ud83d\udee1\ufe0f Constraints & Compliance

The system enforces hard constraints:
- **Rank Requirements**: Officers must meet minimum rank for roles
- **Branch Requirements**: Roles restricted to specific branches
- **Pool Requirements**: Career pool compatibility
- **Entry Type**: Considers how officers entered service

**Constraint Compliance**: 100% (all predictions meet requirements)

## \ud83d\udcdd Project Structure

```
AIAppointer/
\u251c\u2500\u2500 src/
\u2502   \u251c\u2500\u2500 app.py                 # Main Streamlit application
\u2502   \u251c\u2500\u2500 predictor.py           # Prediction engine
\u2502   \u251c\u2500\u2500 markov_engine.py       # Markov chain implementation
\u2502   \u251c\u2500\u2500 xai_helpers.py         # XAI display functions
\u2502   \u251c\u2500\u2500 gantt_viz.py           # Gantt chart visualization
\u2502   \u251c\u2500\u2500 features_ltr.py        # Feature engineering
\u2502   \u251c\u2500\u2500 train_ltr.py           # Model training
\u2502   \u2514\u2500\u2500 ...
\u251c\u2500\u2500 models/
\u2502   \u251c\u2500\u2500 ltr/                   # Trained models
\u2502   \u2514\u2500\u2500 all_constraints.json   # Role constraints
\u251c\u2500\u2500 data/                      # Dataset (not in repo)
\u251c\u2500\u2500 docs/                      # Documentation
\u251c\u2500\u2500 tests/                     # Unit tests
\u251c\u2500\u2500 requirements.txt           # Python dependencies
\u2514\u2500\u2500 README.md                  # This file
```

## \ud83e\udd1d Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## \ud83d\udcdc License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## \ud83d\udce7 Contact

For questions or support, please open an issue on GitHub.

## \ud83d\ude80 Deployment

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for instructions on deploying to:
- Streamlit Cloud
- Docker containers
- Local production servers

## \ud83d\udcda Citation

If you use this system in your research or project, please cite:

```bibtex
@software{ai_appointer_2025,
  title = {AI Appointer: Intelligent Career Progression System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/AIAppointer}
}
```

---

**Built with \u2764\ufe0f using Streamlit, LightGBM, and SHAP**
