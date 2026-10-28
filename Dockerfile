FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8501 8000
# Default: the web UI. For the API instead: uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
