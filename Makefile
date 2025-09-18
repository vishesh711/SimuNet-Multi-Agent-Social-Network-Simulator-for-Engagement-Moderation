.PHONY: install install-dev test lint format clean run-api run-dev docker-build docker-run

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,research]"

# Development
test:
	pytest

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term-missing

lint:
	flake8 src tests
	mypy src

format:
	black src tests
	isort src tests

format-check:
	black --check src tests
	isort --check-only src tests

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/

# Running
run-api:
	uvicorn simu_net.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dev: install-dev
	uvicorn simu_net.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# Docker
docker-build:
	docker build -t simu-net:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# Database
setup-db:
	docker run -d --name simu-net-mongo -p 27017:27017 mongo:latest
	docker run -d --name simu-net-redis -p 6379:6379 redis:latest

# Pre-commit
setup-pre-commit:
	pre-commit install

# Help
help:
	@echo "Available commands:"
	@echo "  install       - Install package"
	@echo "  install-dev   - Install with development dependencies"
	@echo "  test          - Run tests"
	@echo "  test-cov      - Run tests with coverage"
	@echo "  lint          - Run linting"
	@echo "  format        - Format code"
	@echo "  clean         - Clean build artifacts"
	@echo "  run-api       - Run API server"
	@echo "  run-dev       - Run API server in development mode"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run with Docker Compose"
	@echo "  setup-db      - Setup local databases with Docker"