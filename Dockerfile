# ── Stage: runtime ───────────────────────────────────────────────────────────
FROM python:3.10-slim

# Metadata
LABEL maintainer="Zhuojun Lyu <lzj2729033776@gmail.com>"
LABEL description="Social Sentiment Tracker — Streamlit web demo"

# Prevents Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only requirements first so Docker layer cache is reused when code changes
COPY requirements.txt .

# Install CPU-only torch to keep the image lean (~1 GB vs ~3 GB for CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Ensure runtime directories exist
RUN mkdir -p data/raw data/processed models reports/figures

# ── Streamlit configuration ───────────────────────────────────────────────────
# Disable Streamlit's browser-open and usage stats for container use
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

# Health check — polls the Streamlit health endpoint every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
