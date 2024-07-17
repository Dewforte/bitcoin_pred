#!/bin/bash
set -e

echo "Pulling latest changes..."
git pull

echo "Building Docker image..."
docker-compose build

echo "Starting Docker container..."
docker-compose up -d

echo "Deployment complete!"