#!/bin/bash

# Docker Build and Test Script for PDF Parser App
# Usage: ./docker-build-test.sh

echo "🐳 PDF Parser App - Docker Build & Test Script"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
print_status "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker."
    exit 1
fi

print_status "Docker is running ✓"

# Build the Docker image
print_status "Building Docker image 'pdf-parser-app'..."
if docker build -t pdf-parser-app .; then
    print_status "Docker image built successfully ✓"
else
    print_error "Failed to build Docker image"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Check if you have internet connection"
    echo "2. Try: docker system prune -f"
    echo "3. Try: docker pull python:3.11-slim"
    echo "4. Check Docker Desktop is running"
    exit 1
fi

# List the created image
print_status "Listing Docker images..."
docker images | grep pdf-parser-app

# Test the container
print_status "Testing the container..."

# Create output directory if it doesn't exist
mkdir -p output

# Run the container
print_status "Running the PDF parser application..."
if docker run --rm \
    -v "$(pwd)/output:/app/output" \
    pdf-parser-app; then
    print_status "Container ran successfully ✓"
else
    print_warning "Container execution completed with warnings or errors"
fi

# Check if output was generated
if [ -f "output.json" ]; then
    print_status "Output file generated ✓"
    echo "Output file size: $(wc -c < output.json) bytes"
else
    print_warning "No output.json file was generated"
fi

print_status "Build and test completed!"

echo ""
echo "📋 Available Docker commands:"
echo "  Build image:           docker build -t pdf-parser-app ."
echo "  Run container:         docker run --rm pdf-parser-app"
echo "  Run with output mount: docker run --rm -v \$(pwd)/output:/app/output pdf-parser-app"
echo "  Run interactively:     docker run --rm -it pdf-parser-app bash"
echo "  View logs:             docker logs <container_id>"
echo "  Remove image:          docker rmi pdf-parser-app"
