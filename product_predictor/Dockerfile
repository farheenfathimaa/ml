FROM python:3.9-slim

WORKDIR /app

# Configure DNS and update package lists with retries
RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf && \
    echo "nameserver 1.1.1.1" >> /etc/resolv.conf

# Install system dependencies with retries and alternative mirrors
RUN for i in 1 2 3; do \
        apt-get update && \
        apt-get install -y \
            gcc \
            g++ \
            build-essential \
            python3-dev \
            curl \
            wget \
            ca-certificates \
        && break || sleep 5; \
    done && \
    rm -rf /var/lib/apt/lists/*

# Create logs directory
RUN mkdir -p /app/logs

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with retries
RUN pip install --no-cache-dir --upgrade pip setuptools wheel --timeout=120 --retries=5
RUN pip install --no-cache-dir -r requirements.txt --timeout=120 --retries=5

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p /app/data

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["python", "app.py"]
