# Deployment Guide

## Local Deployment

### Prerequisites
- Python 3.8+
- pip package manager
- 4GB+ RAM recommended

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AIAppointer.git
cd AIAppointer
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare data**
- Place your dataset in `data/` directory
- Update `config.py` with correct data path

5. **Run the application**
```bash
streamlit run src/app.py
```

6. **Access the app**
Open browser at `http://localhost:8501`

---

## Streamlit Cloud Deployment

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at streamlit.io/cloud)

### Steps

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/AIAppointer.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Click "New app"
- Select your repository
- Set main file path: `src/app.py`
- Click "Deploy"

3. **Configure secrets** (if needed)
- In Streamlit Cloud dashboard, go to app settings
- Add secrets in TOML format

4. **Monitor deployment**
- Check logs for any errors
- App will be live at `https://yourapp.streamlit.app`

### Important Notes for Streamlit Cloud
- Models and data files must be included in repo or loaded from external storage
- Keep repository size under 1GB
- Use `.gitignore` to exclude large files
- Consider using Git LFS for model files

---

## Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t ai-appointer .

# Run container
docker run -p 8501:8501 ai-appointer
```

### Docker Compose (Optional)

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
```

Run with: `docker-compose up`

---

## Production Considerations

### Performance
- Use caching (`@st.cache_data`, `@st.cache_resource`)
- Limit concurrent users based on server capacity
- Consider load balancing for high traffic

### Security
- Never commit sensitive data or credentials
- Use environment variables for configuration
- Implement authentication if needed
- Keep dependencies updated

### Monitoring
- Set up logging
- Monitor resource usage (CPU, RAM)
- Track prediction latency
- Monitor model performance drift

### Backup
- Regular backups of models and data
- Version control for models
- Document model training procedures

---

## Troubleshooting

### Common Issues

**Issue**: Module not found errors
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Model files not loading
**Solution**: Check paths in `config.py` and ensure models exist in `models/` directory

**Issue**: Out of memory errors
**Solution**: Reduce batch size, limit data loading, or increase server RAM

**Issue**: Slow predictions
**Solution**: Check if models are cached properly, optimize feature engineering

---

## Environment Variables

Create `.streamlit/secrets.toml` for sensitive configuration:

```toml
[general]
data_path = "/path/to/data"
model_path = "/path/to/models"

[database]
host = "localhost"
port = 5432
```

Access in code:
```python
import streamlit as st
data_path = st.secrets["general"]["data_path"]
```

---

## Scaling

For high-traffic deployments:
1. Use a dedicated server or cloud platform
2. Implement caching strategies
3. Consider horizontal scaling with load balancer
4. Use CDN for static assets
5. Optimize database queries if applicable

---

## Support

For deployment issues:
- Check [Streamlit documentation](https://docs.streamlit.io)
- Open an issue on GitHub
- Review logs for error messages
