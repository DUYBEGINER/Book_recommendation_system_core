FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create artifacts directory
RUN mkdir -p /app/artifacts

EXPOSE 8001

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8001"]