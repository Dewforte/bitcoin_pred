#!/bin/bash
set -e

echo "Stopping current container..."
docker-compose down

echo "Pulling latest changes..."
git pull

echo "Building new Docker image..."
docker-compose build

echo "Starting updated container..."
docker-compose up -d

echo "Update complete!"