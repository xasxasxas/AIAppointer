# Air-Gapped (Offline) Deployment Guide

**Version**: 1.0  
**Date**: 2025-12-13

This guide details how to bundle and deploy the AI Appointer in a secure environment with **zero internet access**.

---

## 1. Preparation (On Internet-Connected Machine)
You need to download all dependencies as "wheels" (.whl files) before moving to the secure network.

### Step A: Bundle Dependencies
1.  Navigate to the project root.
2.  Create a strict lockfile (optional but recommended) or use `requirements.txt`.
3.  Run the following command to download all packages into a folder named `offline_packages`:
    ```bash
    pip download -r requirements.txt -d ./offline_packages
    ```
    *This will download pandas, streamlit, shap, lightgbm, etc., and all their sub-dependencies.*

### Step B: Bundle Project
1.  Copy the entire project folder, including:
    *   `src/`
    *   `models/` (Ensure your trained `.pkl` models are present!)
    *   `data/` (Ensure your CSV data is present)
    *   `offline_packages/` (The folder you just created)
    *   `requirements.txt`
    *   `run_app.bat` (if applicable)

### Step C: Transfer
Move this bundled folder to the secure air-gapped machine via USB or secure file transfer.

---

## 2. Installation (On Air-Gapped Machine)
### Step A: Install Python
Ensure Python 3.9+ is installed. If Python is not installed, you must provide the Python installer executable in your transfer bundle.

### Step B: Install Libraries Offline
1.  Open a terminal/command prompt in the project folder.
2.  Run the install command telling `pip` to look **only** in your local folder:
    ```bash
    pip install --no-index --find-links=./offline_packages -r requirements.txt
    ```

---

## 3. Configuration (Critical for Offline)
Streamlit tries to "phone home" for usage statistics and email prompts. You must disable this to prevent timeout errors or firewall alerts.

1.  **Check for `.streamlit/config.toml`** in your project folder. If it doesn't exist, create it.
2.  **Add the following lines**:

    ```toml
    [browser]
    gatherUsageStats = false

    [server]
    headless = true
    enableCORS = false
    enableXsrfProtection = false
    ```

---

## 4. Running the App
Run as normal:
```bash
streamlit run src/app.py
```

## 5. Troubleshooting
*   **"Missing Dependency"**: If `pip` fails saying it can't find a package, it means the dependency resolution on the internet machine differed from the target machine (e.g., Windows vs Linux). ensure you run the `pip download` command on the **same OS architecture** as the target machine.
*   **"Model not found"**: Ensure `models/ltr/` contains `lgbm_ranker.pkl` and other files. These are not installed via pip; they are part of your repo files.
