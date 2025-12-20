# 🚀 Deployment Guide for TalentSync AI

This comprehensive guide covers all deployment scenarios for **TalentSync AI**:
1.  **Streamlit Cloud** (Easiest, for demos)
2.  **Local Installation** (Development/Testing)
3.  **Local Production** (On-premise server)
4.  **Air-Gapped / Offline** (High-security environments)
5.  **Docker** (Containerized deployment)

---

## 1. Streamlit Cloud Deployment
**Best for:** Rapid prototyping, public demos.

### Prerequisites
- GitHub account
- [Streamlit Cloud account](https://streamlit.io/cloud)

### Steps
1.  **Push Code to GitHub**:
    ```bash
    git init
    git add .
    git commit -m "Initial release"
    git push origin main
    ```
2.  **Deploy**:
    - Go to [share.streamlit.io](https://share.streamlit.io) -> "New app".
    - Select your repository.
    - Set Main file path: `src/app.py`.
    - Click **Deploy**.

### Configuration
Add secrets in Streamlit Cloud Dashboard (if using external DBs):
```toml
[general]
admin_password = "secure_password"
```

---

## 2. Local Installation
**Best for:** Development and testing.

### Prerequisites
- Python 3.8+
- pip

### Steps
1.  **Clone & Setup**:
    ```bash
    git clone https://github.com/your-org/TalentSyncAI.git
    cd TalentSyncAI
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```
2.  **Install Application**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Dev Server**:
    ```bash
    streamlit run src/app.py
    ```
    Access at `http://localhost:8501`.

---

## 3. Local Production (On-Premise)
**Best for:** Internal corporate networks.

### Recommended Specs
- **CPU**: 4+ Cores
- **RAM**: 8GB+ (16GB recommended for large datasets)
- **OS**: Windows Server 2019+ or Linux (Ubuntu 20.04+)

### Steps
1.  Follow "Local Installation" steps.
2.  **Run on Port 80** (requires admin/root):
    ```bash
    streamlit run src/app.py --server.port 80 --server.address 0.0.0.0
    ```
3.  **Configure Firewall**:
    - Allow inbound traffic on port 80 (or 8501).

---

## 4. 🛡️ Air-Gapped / Offline Deployment
**Best for:** High-security, no-internet zones.

### Phase 1: Preparation (Internet Machine)
1.  **Download Wheels**:
    ```bash
    mkdir wheels
    # For Windows Target
    pip download -r requirements.txt -d wheels/
    
    # For Linux Target (from Windows machine)
    pip download -r requirements.txt -d wheels/ --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.9
    ```
2.  **Package**:
    Zip the `TalentSyncAI` folder and `wheels` folder together. Transfter via secure USB.

### Phase 2: Installation (Secure Machine)
1.  **Extract** to destination (e.g., `C:\Apps\TalentSyncAI`).
2.  **Install Offline**:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install --no-index --find-links=../wheels -r requirements.txt
    ```
3.  **Run**:
    ```bash
    streamlit run src/app.py --server.port 8501
    ```

---

## 5. Docker Deployment
**Best for:** Container orchestration, consistency.

### Dockerfile
Create `Dockerfile` in root:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build & Run
```bash
docker build -t talentsync-ai .
docker run -p 8501:8501 talentsync-ai
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **ModuleNotFoundError** | Ensure `pip install -r requirements.txt` ran successfully. |
| **Model Load Fail** | Check `models/` folder permissions and existence. |
| **Port Clashes** | Use `--server.port XXXX` to specify a different port. |
| **Slow Performance** | Ensure server has enough RAM; Python might be swapping. |

---

**Need Help?** Contact the internal AI team at `ai-support@company.com`.
