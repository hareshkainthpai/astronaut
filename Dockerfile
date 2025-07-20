# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies for vLLM and GPU support
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    vllm \
    transformers \
    accelerate

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/static /app/media /app/logs

# Set permissions
RUN chmod +x manage.py

# Expose ports
EXPOSE 8000 6379

# Create startup script using echo instead of heredoc
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Start Redis server in background' >> /app/start.sh && \
    echo 'redis-server --daemonize yes --bind 0.0.0.0 --port 6379' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Wait for Redis to start' >> /app/start.sh && \
    echo 'sleep 2' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Run Django migrations' >> /app/start.sh && \
    echo 'python3 manage.py makemigrations' >> /app/start.sh && \
    echo 'python3 manage.py migrate' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Collect static files' >> /app/start.sh && \
    echo 'python3 manage.py collectstatic --noinput' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Start Django server' >> /app/start.sh && \
    echo 'python3 manage.py runserver 0.0.0.0:8000' >> /app/start.sh

RUN chmod +x /app/start.sh

# Start the application
CMD ["/app/start.sh"]