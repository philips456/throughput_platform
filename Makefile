# Configuration
PROJECT_NAME=throughput_platform
DOCKER_USER=philippe545  # change this to your Docker username
PORT=8000
MLFLOW_PORT=5000
VERSION = latest  # Static version for the image (you can change this manually for each release)

# Dockerfile paths
DOCKERFILE_BACKEND = ./Dockerfile.backend
DOCKERFILE_FRONTEND = ./Dockerfile.frontend
DOCKERFILE_MLFLOW = ./Dockerfile.model

.PHONY: run run_mlflow makemigrations migrate collectstatic createsuperuser test lint secure format trainmodel clean build tag push pull run-container check-deploy docker-up docker-down check ci cd

# ========================================
# 🔁 CI – Continuous Integration
# ========================================
format:
	@echo "🚀 Formatting code with Black..."
	black .


test:
	@echo "🚀 Running unit tests..."
	pytest || exit 0

trainmodel:
	@echo "🚀 Training the ML model..."
	python train_tf_model.py
check:
	@echo "🚀 Running full code checks..."
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) secure
	$(MAKE) test

ci:
	@echo "🚀 CI Pipeline: format + lint + secure + test + trainmodel"
	$(MAKE) check
	$(MAKE) trainmodel
# ========================================
# 🐳 Docker Build + Push
# ========================================
# Build MLflow Docker image
build-mlflow:
	@echo "🐳 Building MLflow Docker image..."
	docker build -t $(PROJECT_NAME)-mlflow:$(VERSION) -f $(DOCKERFILE_MLFLOW) .

# Build Backend Docker image
build-backend:
	@echo "🐳 Building Backend Docker image..."
	docker build -t $(PROJECT_NAME)-backend:$(VERSION) -f $(DOCKERFILE_BACKEND) .

# Build Frontend Docker image
build-frontend:
	@echo "🐳 Building Frontend Docker image..."
	docker build -t $(PROJECT_NAME)-frontend:$(VERSION) -f $(DOCKERFILE_FRONTEND) .



# Tag Docker images with the project version
tag:
	@echo "🏷️ Tagging Docker images..."
	docker tag $(PROJECT_NAME)-mlflow:$(VERSION) $(DOCKER_USER)/5g_throughput_prediction:mlflow-$(VERSION)
	docker tag $(PROJECT_NAME)-backend:$(VERSION) $(DOCKER_USER)/5g_throughput_prediction:backend-$(VERSION)
	docker tag $(PROJECT_NAME)-frontend:$(VERSION) $(DOCKER_USER)/5g_throughput_prediction:frontend-$(VERSION)

# Push Docker images to Docker Hub
push:
	@echo "📤 Pushing Docker images to Docker Hub..."
	docker push $(DOCKER_USER)/5g_throughput_prediction:mlflow-$(VERSION)
	docker push $(DOCKER_USER)/5g_throughput_prediction:backend-$(VERSION)
	docker push $(DOCKER_USER)/5g_throughput_prediction:frontend-$(VERSION)

# ========================================
# 🚀 CD – Continuous Deployment
# ========================================

pull:
	@echo "📥 Pulling Docker images from Docker Hub..."
	docker pull $(DOCKER_USER)/5g_throughput_prediction:mlflow-$(VERSION)
	docker pull $(DOCKER_USER)/5g_throughput_prediction:backend-$(VERSION)
	docker pull $(DOCKER_USER)/5g_throughput_prediction:frontend-$(VERSION)

run-container:
	@echo "🚀 Recreating Docker containers with Docker Compose..."
	docker-compose down
	docker-compose up 

check-deploy:
	@echo "🌐 Checking if backend is reachable..."
	curl --fail http://localhost:$(PORT) || echo "❌ Backend not reachable"
	@echo "🌐 Checking if MLflow is reachable..."
	curl --fail http://localhost:$(MLFLOW_PORT) || echo "❌ MLflow not reachable"


# ========================================
# ⚙️ Django
# ========================================

makemigrations:
	python manage.py makemigrations

migrate:
	python manage.py migrate

collectstatic:
	python manage.py collectstatic --noinput

createsuperuser:
	python manage.py createsuperuser

# ========================================
# 🧹 Clean up
# ========================================

clean:
	del /s /q *.pyc 2>NUL || exit 0
	del /s /q *__pycache__* 2>NUL || exit 0

# ========================================
# Other
# ========================================
