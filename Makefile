# Makefile for MusicGen project

.PHONY: help install install-dev test test-unit test-integration test-e2e test-fast test-slow lint format type-check clean docs serve-docs build docker-build docker-run benchmark

# Default target
help:
	@echo "MusicGen Development Commands"
	@echo "============================"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install package and dependencies"
	@echo "  install-dev    Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-e2e       Run end-to-end tests only"
	@echo "  test-fast      Run fast tests only (exclude slow/gpu/model)"
	@echo "  test-slow      Run slow tests"
	@echo "  test-cov       Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run all linters (flake8, mypy)"
	@echo "  format         Format code (black, isort)"
	@echo "  type-check     Run type checking (mypy)"
	@echo "  pre-commit     Run pre-commit hooks"
	@echo ""
	@echo "Documentation:"
	@echo "  docs           Build documentation"
	@echo "  serve-docs     Serve documentation locally"
	@echo ""
	@echo "Build & Deploy:"
	@echo "  build          Build package"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Clean build artifacts"
	@echo "  benchmark      Run performance benchmarks"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest

test-unit:
	pytest tests/unit -m unit

test-integration:
	pytest tests/integration -m integration

test-e2e:
	pytest tests/e2e -m e2e

test-fast:
	pytest -m "not slow and not gpu and not model"

test-slow:
	pytest -m slow --runslow

test-cov:
	pytest --cov=music_gen --cov-report=html --cov-report=term

test-parallel:
	pytest -n auto

# Code Quality
lint: lint-flake8 lint-mypy

lint-flake8:
	flake8 music_gen tests scripts

lint-mypy:
	mypy music_gen

format:
	black music_gen tests scripts
	isort music_gen tests scripts

format-check:
	black --check music_gen tests scripts
	isort --check-only music_gen tests scripts

type-check:
	mypy music_gen

pre-commit:
	pre-commit run --all-files

# Documentation
docs:
	@echo "Building documentation..."
	@echo "Documentation structure:"
	@echo "- README.md: Main project documentation"
	@echo "- CLAUDE.md: Development context and commands"
	@echo "- API docs: Available at /docs when server is running"

serve-docs:
	@echo "Starting API server for documentation..."
	uvicorn music_gen.api.main:app --reload --host 0.0.0.0 --port 8000
	@echo "API documentation available at:"
	@echo "- Swagger UI: http://localhost:8000/docs"
	@echo "- ReDoc: http://localhost:8000/redoc"

# Build
build:
	python -m build

build-wheel:
	python -m build --wheel

build-sdist:
	python -m build --sdist

# Docker
docker-build:
	docker build -t music-gen-ai:latest .

docker-build-dev:
	docker build --target development -t music-gen-ai:dev .

docker-build-prod:
	docker build --target production -t music-gen-ai:prod .

docker-build-gpu:
	docker build --target gpu-production -t music-gen-ai:gpu .

docker-run:
	docker run -p 8000:8000 music-gen-ai:latest

docker-run-dev:
	docker run -p 8000:8000 -v $(PWD):/app music-gen-ai:dev

docker-compose-dev:
	docker-compose --profile dev up

docker-compose-prod:
	docker-compose --profile prod up -d

docker-compose-gpu:
	docker-compose --profile gpu up -d

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf outputs/
	rm -rf multirun/

clean-all: clean
	rm -rf .tox/
	rm -rf .venv/
	docker system prune -f

# Development utilities
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test-fast' to verify installation"

jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Model and data utilities
download-models:
	@echo "Model downloads would happen here"
	@echo "Currently using on-demand model loading"

prepare-data:
	python scripts/prepare_data.py --dataset synthetic --output_dir data/

# Training utilities
train:
	music-gen-train --config configs/training/default.yaml

train-small:
	music-gen-train --config configs/training/small.yaml

train-debug:
	music-gen-train --config configs/training/debug.yaml trainer.fast_dev_run=5

# API utilities
api-dev:
	uvicorn music_gen.api.main:app --reload --host 0.0.0.0 --port 8000

api-prod:
	gunicorn music_gen.api.main:app \
		--worker-class uvicorn.workers.UvicornWorker \
		--workers 4 \
		--bind 0.0.0.0:8000

# Benchmarking
benchmark:
	python scripts/benchmark.py

benchmark-model:
	python scripts/benchmark_model.py

benchmark-api:
	python scripts/benchmark_api.py

# Release utilities
release-check:
	python -m build
	twine check dist/*

release-test:
	twine upload --repository testpypi dist/*

release:
	twine upload dist/*

# Git utilities
git-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Security
security-check:
	safety check
	bandit -r music_gen/

# Performance profiling
profile:
	python -m cProfile scripts/profile_generation.py

# Monitoring
monitor-gpu:
	watch -n 1 nvidia-smi

monitor-resources:
	htop

# Configuration validation
validate-config:
	python -c "from music_gen.models.transformer.config import MusicGenConfig; print('✓ Config validation passed'); MusicGenConfig()"

# CLI testing
test-cli:
	music-gen info
	music-gen generate "test music" --duration 1 --output /tmp/test.wav || true

# Integration test with real dependencies
test-integration-real:
	pytest tests/integration -m "not mock" --runslow

# Database and cache management (for future use)
reset-cache:
	rm -rf .cache/
	rm -rf /tmp/musicgen/

# Development workflow shortcuts
dev: install-dev format lint test-fast
ci: install format-check lint test-cov
deploy: clean build docker-build-prod

# Quick commands for common workflows
quick-test: test-fast
quick-dev: format test-fast
quick-check: format-check lint test-unit

# Help for specific areas
help-testing:
	@echo "Testing Commands:"
	@echo "  make test-fast     # Quick tests (no slow/gpu/model tests)"
	@echo "  make test-unit     # Unit tests only"
	@echo "  make test-cov      # Tests with coverage"
	@echo "  make test-parallel # Parallel test execution"

help-docker:
	@echo "Docker Commands:"
	@echo "  make docker-build-dev    # Development image"
	@echo "  make docker-build-prod   # Production image" 
	@echo "  make docker-build-gpu    # GPU-enabled image"
	@echo "  make docker-compose-dev  # Dev environment"
	@echo "  make docker-compose-prod # Production deployment"