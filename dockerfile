FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build tools for scientific packages and clean up cache
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libpq-dev pkg-config cargo curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first so Docker can cache installs
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip maturin \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download SBERT model so runtime doesn't rely on HF network
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('keepitreal/vietnamese-sbert')"

# Copy application source
COPY . .

# Ensure artifact folders exist (volumes may mount over them)
RUN mkdir -p /app/artifacts /app/artifacts_implicit_sbert /app/artifacts_neural

ENV APP_MODULE=server_implicit_sbert:app \
    HOST=0.0.0.0 \
    PORT=8001 \
    WORKERS=1

EXPOSE 8001

CMD ["sh", "-c", "uvicorn ${APP_MODULE} --host ${HOST} --port ${PORT} --workers ${WORKERS}"]
