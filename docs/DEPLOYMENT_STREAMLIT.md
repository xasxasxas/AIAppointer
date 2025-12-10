# Deployment: Streamlit Cloud

This guide explains how to deploy the AI Appointer to [Streamlit Community Cloud](https://streamlit.io/cloud).

## Prerequisites
*   A **GitHub Account**.
*   This codebase pushed to a GitHub repository.

## Step-by-Step Deployment

1.  **Push Code to GitHub**:
    Ensure your repository structure looks like this:
    ```
    repo/
    ├── data/
    │   └── hr_star_trek_v4c_modernized_clean_modified_v4.csv
    ├── models/
    │   ├── ltr/
    │   │   ├── lgbm_model.pkl
    │   │   └── transition_stats.pkl (and others)
    │   └── all_constraints.json
    ├── src/
    │   ├── app.py
    │   └── predictor.py
    ├── requirements.txt
    └── README.md
    ```
    *Note: The `models` and `data` folders must be included. Ensure `lgbm_model.pkl` is <100MB.*

2.  **Login to Streamlit Cloud**:
    *   Go to [share.streamlit.io](https://share.streamlit.io/).
    *   Login with GitHub.

3.  **Create New App**:
    *   Click "New app".
    *   Select your Repository, Branch (main), and Main file path (`src/app.py`).
    
    *Important: Main file path is `src/app.py`, NOT `app.py`.*

4.  **Advanced Settings (Optional)**:
    *   Select Python 3.10+.
    *   No secrets are required for the base version.

5.  **Deploy**:
    *   Click "Deploy!".
    *   Wait ~2 minutes for dependencies to install.

## Troubleshooting
*   **Memory Limit**: If the app crashes on load, Streamlit Cloud Free Tier has a 1GB limit. The model footprint is ~200MB, so it should be fine. If it crashes, try closing other tabs or rebooting the app.
*   **Module Not Found**: Ensure `requirements.txt` includes `lightgbm` and `scikit-learn`.
