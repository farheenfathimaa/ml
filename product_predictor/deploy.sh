#!/bin/bash
set -e

echo "🚀 Deploying ML API with ELK Stack..."

# Configuration
PROJECT_NAME="product-predictor"
LOG_DIR="./logs"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p $LOG_DIR
mkdir -p logstash/config
mkdir -p logstash/pipeline
mkdir -p data

# Stop existing services
echo "🛑 Stopping existing services..."
docker-compose down -v || true

# Clean up Docker system (optional)
echo "🧹 Cleaning up Docker..."
docker system prune -f || true

# Set proper permissions
echo "🔐 Setting permissions..."
sudo chmod -R 755 $LOG_DIR || chmod -R 755 $LOG_DIR
sudo chmod -R 755 logstash || chmod -R 755 logstash

# Set Elasticsearch memory settings
echo "⚙️ Setting Elasticsearch memory..."
sudo sysctl -w vm.max_map_count=262144 || echo "Could not set vm.max_map_count (may need sudo)"

# Update package lists
echo "📦 Updating system packages..."
sudo apt update || echo "Could not update packages"

# Install docker-compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "Installing docker-compose..."
    sudo apt install -y docker-compose || echo "Please install docker-compose manually"
fi

# Build and start only the ML application first
echo "🔨 Building ML application..."
docker-compose build product-predictor

# Start ML app first to test
echo "🏃 Starting ML application..."
docker-compose up -d product-predictor

# Wait for ML API to be ready
echo "⏳ Waiting for ML API to start..."
for i in {1..30}; do
    if curl -f http://localhost:5000/health > /dev/null 2>&1; then
        echo "✅ ML API is ready"
        break
    fi
    echo "Waiting for ML API... ($i/30)"
    sleep 5
done

# If ML API is working, start ELK stack
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "🔨 Starting ELK Stack..."
    docker-compose up -d elasticsearch
    sleep 30
    
    # Check Elasticsearch
    echo "🔍 Checking Elasticsearch..."
    for i in {1..20}; do
        if curl -s http://localhost:9200/_cluster/health | grep -q "green\|yellow"; then
            echo "✅ Elasticsearch is ready"
            break
        fi
        echo "Waiting for Elasticsearch... ($i/20)"
        sleep 10
    done
    
    # Start remaining services
    docker-compose up -d logstash kibana nginx
    
    # Final health check
    sleep 30
    echo "🔍 Final health check..."
    for i in {1..10}; do
        if curl -f http://localhost/health > /dev/null 2>&1; then
            echo "✅ All services are ready"
            break
        fi
        echo "Waiting for services... ($i/10)"
        sleep 5
    done
else
    echo "❌ ML API failed to start. Check logs:"
    docker-compose logs product-predictor
    exit 1
fi

# Test ML API with sample data
echo "🧪 Testing ML API..."
curl -X POST http://localhost/predict \
    -H "Content-Type: application/json" \
    -d '{"type": "single", "description": "Motor for Conveyor"}' \
    || echo "⚠️ ML API test failed - but service might still be starting"

# Display service status
echo "📊 Service Status:"
docker-compose ps

# Display access information
IP=$(hostname -I | awk '{print $1}' | tr -d '[:space:]')
echo ""
echo "🎉 Deployment Complete!"
echo "==============================================="
echo "🌐 ML API: http://$IP"
echo "📊 Kibana Dashboard: http://$IP/kibana"
echo "🔍 Elasticsearch: http://$IP:9200"
echo "❤️ Health Check: http://$IP/health"
echo "==============================================="
echo ""
echo "📚 Quick Commands:"
echo "  View logs: docker-compose logs -f product-predictor"
echo "  Stop all: docker-compose down"
echo "  Restart: docker-compose restart"
echo ""

# Show logs if there are any issues
if ! curl -f http://localhost/health > /dev/null 2>&1; then
    echo "⚠️ Some services may still be starting. Check logs if needed:"
    echo "docker-compose logs product-predictor"
fi
