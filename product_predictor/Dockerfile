# Dockerfile in the root of your product_predictor project
# Using Debian 11 (Bullseye) which is more current than Buster (Debian 10)
FROM python:3.9-slim-bullseye

WORKDIR /app

# Set up DNS inside the container for apt-get during the build
# This is a common workaround for "Temporary failure in name resolution" during apt-get update
RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf.new && \
    cat /etc/resolv.conf.new > /etc/resolv.conf && \
    rm /etc/resolv.conf.new

# Install build dependencies for some Python packages (e.g., torch, scikit-learn)
# This is crucial for many scientific Python libraries as they often compile C extensions.
# We also update pip itself.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

# Use a specific index URL for pip install to bypass potential network/DNS issues
# This forces pip to get packages from a reliable source directly.
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://pypi.org/simple/ \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

COPY . .

# Expose the port your Flask app runs on
EXPOSE 5000

# Command to run your Flask application
CMD ["python", "app.py"]