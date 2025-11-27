# Dockerfile for Inclusive FL
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml /app/
COPY README.md /app/
COPY LICENSE /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy source code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Create data and checkpoint directories
RUN mkdir -p /app/data /app/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WANDB_MODE=offline

# Default command
CMD ["python", "src/harness.py", "--help"]
