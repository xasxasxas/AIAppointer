# 🎯 TalentSync AI - Intelligent Talent Placement System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/download)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [User Guide](docs/USER_GUIDE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Developer Onboarding](docs/DEVELOPER_ONBOARDING.md)
- [Technical Whitepaper](docs/TECHNICAL_WHITEPAPER.md)
- [Academic Report](docs/Report/01_Abstract.md)

## 🚀 Overview

**TalentSync AI** is an intelligent talent placement and career progression system for HR departments. It uses machine learning to predict optimal next roles for personnel based on career history, skills, and organizational patterns. The system leverages **LightGBM Learning-to-Rank (LTR)** models enhanced with **Markov Chain** sequential modeling.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏠 **Dashboard** | Interactive overview with stats, charts, and navigation guide |
| 👤 **Employee Lookup** | Find the best next role for any officer |
| 🎯 **Billet Lookup** | Find the best candidates for an open position |
| 🔍 **Semantic AI Search** | Natural language search with include/exclude filters |
| 📊 **Analytics & Explorer** | Career flow visualization and data insights |
| 🔄 **Simulation** | Test hypothetical scenarios ("what-if" analysis) |
| ⚙️ **Admin Console** | Retrain models and manage deployments |

## 📊 Performance

- **Model Accuracy**: 99.98% AUC
- **Top-1 Accuracy**: 60%
- **Inference Speed**: <100ms per prediction
- **Constraint Compliance**: 100%

## 💻 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/TalentSyncAI.git
cd TalentSyncAI

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run src/app.py
```

Access at `http://localhost:8501`

## 🛠️ Technology Stack

- **ML Framework**: LightGBM (Learning-to-Rank)
- **XAI**: SHAP (SHapley Additive exPlanations)
- **Frontend**: Streamlit
- **Sequential Modeling**: Custom Markov Chain engine
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, ECharts

## 🎯 Module Descriptions

### 🏠 Dashboard
Landing page with:
- System metrics (officers, billets, branches, ranks)
- Data breakdown charts
- Navigation guide for all modes

### 👤 Employee Lookup
Find the best next role for an officer:
- Filter by rank, branch, entry type
- AI-powered recommendations with confidence scores
- SHAP explanations for transparency

### 🎯 Billet Lookup
Find the best candidates for a position:
- Filter target roles by keyword
- Ranked candidates with fit scores
- Career pattern indicators

### 🔍 Semantic AI Search
Three search modes:
1. **Career Match**: INCLUDE/EXCLUDE filters for experience
2. **Billet Search**: Find billets by rank and branch constraints
3. **Similar Officer**: Find officers with similar career trajectories

### 📊 Analytics & Explorer
- Career flow Sankey diagrams
- Appointment timeline (Gantt chart)
- Dataset browser with filters

### 🔄 Simulation
Test hypothetical scenarios:
- Design custom officer profiles
- Adjust parameters
- Analyze AI predictions

### ⚙️ Admin Console
- Upload new HR data
- Retrain model
- Deploy to production
- Rollback capability

## 📝 Project Structure

```
TalentSyncAI/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── predictor.py        # Prediction engine
│   ├── explainer.py        # SHAP explainer
│   ├── semantic_engine.py  # Semantic search
│   ├── markov_engine.py    # Career patterns
│   └── ...
├── models/
│   ├── ltr/                # Trained models
│   └── all_constraints.json
├── data/                   # Dataset (not in repo)
├── requirements.txt
└── README.md
```

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy from `src/app.py`

### Local Production
```bash
streamlit run src/app.py --server.port 80
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

**Built with ❤️ using Streamlit, LightGBM, and SHAP**

*TalentSync AI v4.1*
