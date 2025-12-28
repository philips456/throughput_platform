#!/bin/bash

echo "🚀 CI checks (format, lint, test, train)"
make ci

echo "🐳 Building Docker images..."
make build-mlflow
make build-backend
make build-frontend

echo "🏷️ Tagging Docker images..."
make tag

echo "🚀 Launching application with Docker Compose..."
make run-container

echo "✅ Application is up and running!"
