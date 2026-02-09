#!/bin/bash
# NeuralBlitz Deployment Script
# Generated: 2026-02-08
# Usage: ./deploy.sh [development|production]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ENV=${1:-development}
APP_DIR="/home/runner/workspace/nb-omnibus-router"
IMAGE_NAME="neuralblitz-router"
CONTAINER_NAME="neuralblitz-router"

echo_blue() { echo -e "${BLUE}[DEPLOY]${NC} $1"; }
echo_green() { echo -e "${GREEN}[OK]${NC} $1"; }
echo_yellow() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_red() { echo -e "${RED}[ERROR]${NC} $1"; }

log() { echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1"; }

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo_red "Docker is not installed!"
        exit 1
    fi
    docker --version
}

check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo_red "Docker Compose is not installed!"
        exit 1
    fi
}

backup_config() {
    log "Creating backup of current configuration..."
    if [ -f "$APP_DIR/config/partners.yaml" ]; then
        cp "$APP_DIR/config/partners.yaml" "$APP_DIR/config/partners.yaml.backup.$(date +%Y%m%d%H%M%S)"
    fi
}

build_image() {
    log "Building Docker image..."
    cd "$APP_DIR"
    
    if [ "$ENV" = "production" ]; then
        docker build -t "$IMAGE_NAME:latest" .
    else
        docker build -t "$IMAGE_NAME:dev" .
    fi
    
    echo_green "Image built successfully!"
}

stop_container() {
    log "Stopping existing containers..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
}

start_container() {
    log "Starting container..."
    cd "$APP_DIR"
    
    if [ "$ENV" = "production" ]; then
        docker-compose up -d
    else
        docker run -d \
            --name "$CONTAINER_NAME" \
            -p 8000:8000 \
            -e NB_ENVIRONMENT=development \
            -e NB_DEBUG=true \
            -v "$APP_DIR/config:/config:ro" \
            "$IMAGE_NAME:dev"
    fi
    
    echo_green "Container started!"
}

verify_deployment() {
    log "Verifying deployment..."
    sleep 5
    
    # Check health
    local health=$(curl -s http://localhost:8000/health 2>/dev/null || echo '{"status":"error"}')
    
    if echo "$health" | grep -q '"status":"healthy"'; then
        echo_green "Health check passed!"
        echo "Response: $health"
    else
        echo_yellow "Health check response: $health"
        echo "Checking logs..."
        docker logs "$CONTAINER_NAME" --tail 50
    fi
}

test_endpoints() {
    log "Testing endpoints..."
    
    # Test main endpoint
    curl -s http://localhost:8000/ | head -c 200
    echo ""
    
    # Test capabilities
    curl -s -H "X-API-Key: nb_pat_xxxxxxxxxxxxxxxxxxxx" \
         http://localhost:8000/api/v1/capabilities | head -c 300
    echo ""
}

show_status() {
    log "Deployment Summary"
    echo "================================"
    echo "Environment: $ENV"
    echo "Container: $CONTAINER_NAME"
    echo "Port: 8000"
    echo ""
    echo "Useful commands:"
    echo "  View logs:    docker logs -f $CONTAINER_NAME"
    echo "  Stop:         docker stop $CONTAINER_NAME"
    echo "  Restart:      docker restart $CONTAINER_NAME"
    echo "  API docs:     http://localhost:8000/docs"
    echo "  Health:       http://localhost:8000/health"
}

usage() {
    echo "NeuralBlitz Deployment Script"
    echo ""
    echo "Usage: $0 [environment]"
    echo ""
    echo "Environments:"
    echo "  development  - Development mode (default)"
    echo "  production   - Production mode with nginx"
    echo ""
    echo "Examples:"
    echo "  $0                  # Deploy in development mode"
    echo "  $0 production       # Deploy in production mode"
}

main() {
    echo "=============================================="
    echo "  NeuralBlitz Deployment Script"
    echo "=============================================="
    echo ""
    
    check_docker
    check_docker_compose
    
    echo_blue "Environment: $ENV"
    echo ""
    
    backup_config
    build_image
    stop_container
    start_container
    verify_deployment
    
    echo ""
    show_status
    
    echo ""
    echo_green "Deployment complete!"
}

main "$@"
