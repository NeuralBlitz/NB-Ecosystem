.PHONY: help install install-dev install-all lint format test test-cov typecheck security clean build docs docker-up docker-down update sync

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

help: ## Display this help message
	@echo "$(BLUE)NeuralBlitz Workspace - Available Commands:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies with Poetry
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	poetry install --no-dev

install-dev: ## Install all dependencies including dev tools
	@echo "$(BLUE)Installing all dependencies (including dev)...$(NC)"
	poetry install

install-all: ## Install with all optional extras
	@echo "$(BLUE)Installing with all optional extras...$(NC)"
	poetry install --extras "all"

install-lrs: ## Install LRS Agents dependencies
	@echo "$(BLUE)Installing LRS Agents dependencies...$(NC)"
	poetry install --extras "lrs-agents"

install-bridge: ## Install Integration Bridge dependencies
	@echo "$(BLUE)Installing Integration Bridge dependencies...$(NC)"
	poetry install --extras "bridge"

install-research: ## Install Advanced Research dependencies
	@echo "$(BLUE)Installing Advanced Research dependencies...$(NC)"
	poetry install --extras "research"

# Development workflow
lint: ## Run all linters (ruff, mypy, bandit)
	@echo "$(BLUE)Running linters...$(NC)"
	@echo "$(YELLOW)Running ruff check...$(NC)"
	poetry run ruff check lrs-agents/lrs neuralblitz-core/src lrs-agents/integration-bridge/src Advanced-Research/src
	@echo "$(YELLOW)Running mypy...$(NC)"
	poetry run mypy lrs-agents/lrs neuralblitz-core/src
	@echo "$(YELLOW)Running bandit security check...$(NC)"
	poetry run bandit -r lrs-agents/lrs -c pyproject.toml

format: ## Format code with black and ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	@echo "$(YELLOW)Running black...$(NC)"
	poetry run black lrs-agents/lrs neuralblitz-core/src lrs-agents/integration-bridge/src Advanced-Research/src aetheria-project
	@echo "$(YELLOW)Running ruff check --fix...$(NC)"
	poetry run ruff check --fix lrs-agents/lrs neuralblitz-core/src lrs-agents/integration-bridge/src Advanced-Research/src

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	poetry run pytest tests/ lrs-agents/tests/ neuralblitz-core/tests/ -v

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	poetry run pytest tests/ lrs-agents/tests/ neuralblitz-core/tests/ --cov=lrs --cov=neuralblitz_core --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	poetry run pytest tests/ lrs-agents/tests/ neuralblitz-core/tests/ -v -m unit

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	poetry run pytest tests/ lrs-agents/tests/ neuralblitz-core/tests/ -v -m integration

typecheck: ## Run static type checking with mypy
	@echo "$(BLUE)Running type checker...$(NC)"
	poetry run mypy lrs-agents/lrs neuralblitz-core/src

security: ## Run security checks with bandit
	@echo "$(BLUE)Running security checks...$(NC)"
	poetry run bandit -r lrs-agents/lrs neuralblitz-core/src -c pyproject.toml

# Code quality - all checks
check: format lint typecheck security test ## Run all code quality checks (format, lint, typecheck, security, test)

# Build and package
build: ## Build packages with Poetry
	@echo "$(BLUE)Building packages...$(NC)"
	poetry build

publish: ## Publish packages to PyPI (requires authentication)
	@echo "$(BLUE)Publishing to PyPI...$(NC)"
	poetry publish

# Documentation
docs: ## Build documentation with MkDocs
	@echo "$(BLUE)Building documentation...$(NC)"
	poetry run mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	poetry run mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(BLUE)Deploying documentation...$(NC)"
	poetry run mkdocs gh-deploy

# Docker operations
docker-build: ## Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose build

docker-up: ## Start all services with Docker Compose
	@echo "$(BLUE)Starting Docker services...$(NC)"
	docker-compose up -d

docker-down: ## Stop all Docker services
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	docker-compose down

docker-logs: ## View Docker logs
	@echo "$(BLUE)Viewing Docker logs...$(NC)"
	docker-compose logs -f

docker-clean: ## Remove Docker containers, networks, and volumes
	@echo "$(YELLOW)Removing Docker containers and volumes...$(NC)"
	docker-compose down -v

# Database operations (if applicable)
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	poetry run alembic upgrade head

db-rollback: ## Rollback database migrations
	@echo "$(YELLOW)Rolling back database migrations...$(NC)"
	poetry run alembic downgrade -1

db-reset: ## Reset database (WARNING: Destructive)
	@echo "$(RED)Resetting database...$(NC)"
	poetry run alembic downgrade base
	poetry run alembic upgrade head

# Pre-commit hooks
pre-commit: ## Install and run pre-commit hooks
	@echo "$(BLUE)Setting up pre-commit hooks...$(NC)"
	poetry run pre-commit install
	poetry run pre-commit run --all-files

# Update and maintenance
update: ## Update all dependencies to latest compatible versions
	@echo "$(BLUE)Updating dependencies...$(NC)"
	poetry update

update-lock: ## Update poetry.lock without installing
	@echo "$(BLUE)Updating poetry.lock...$(NC)"
	poetry lock --no-update

sync: ## Sync dependencies with poetry.lock
	@echo "$(BLUE)Syncing dependencies...$(NC)"
	poetry install --sync

# Workspace management
workspace-status: ## Show workspace status
	@echo "$(BLUE)NeuralBlitz Workspace Status:$(NC)"
	@echo ""
	@echo "$(GREEN)Packages:$(NC)"
	@echo "  - lrs-agents: $(shell cd lrs-agents && git describe --tags --always 2>/dev/null || echo 'N/A')"
	@echo "  - neuralblitz-core: $(shell cd neuralblitz-core && git describe --tags --always 2>/dev/null || echo 'N/A')"
	@echo "  - integration-bridge: $(shell cd lrs-agents/integration-bridge && git describe --tags --always 2>/dev/null || echo 'N/A')"
	@echo ""
	@echo "$(GREEN)Python Version:$(NC) $(shell python --version)"
	@echo "$(GREEN)Poetry Version:$(NC) $(shell poetry --version)"
	@echo ""

workspace-info: ## Show detailed workspace information
	@echo "$(BLUE)NeuralBlitz Workspace Information:$(NC)"
	@echo ""
	@poetry env info
	@echo ""
	@echo "$(GREEN)Installed Packages:$(NC)"
	@poetry show --tree | head -50

# Development servers
run-api: ## Run the Integration Bridge API server
	@echo "$(BLUE)Starting API server on http://localhost:9000$(NC)"
	poetry run opencode-lrs-bridge --reload --port 9000

run-lrs: ## Run LRS Agents CLI
	@echo "$(BLUE)Starting LRS Agents...$(NC)"
	poetry run lrs

run-tests-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(BLUE)Starting test watcher...$(NC)"
	poetry run ptw tests/ lrs-agents/tests/

# Benchmarking and profiling
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	poetry run python -m pytest tests/benchmarks/ -v

profile: ## Profile code execution
	@echo "$(BLUE)Profiling...$(NC)"
	poetry run python -m cProfile -o profile.stats -m pytest tests/ -x
	poetry run python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(30)"

# Cleaning
clean: ## Clean build artifacts and cache files
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf site/  # mkdocs build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "$(GREEN)Clean complete!$(NC)"

clean-all: clean ## Clean everything including virtual environment
	@echo "$(RED)Removing virtual environment...$(NC)"
	poetry env remove python || true
	rm -rf .venv/

# CI/CD helpers
ci-setup: ## Setup CI environment
	@echo "$(BLUE)Setting up CI environment...$(NC)"
	pip install poetry
	poetry config virtualenvs.create false
	poetry install --extras "all"

ci-test: ## Run CI tests with coverage
	@echo "$(BLUE)Running CI tests...$(NC)"
	poetry run pytest tests/ lrs-agents/tests/ neuralblitz-core/tests/ --cov=lrs --cov=neuralblitz_core --cov-report=xml --cov-report=term

ci-lint: ## Run CI linting
	@echo "$(BLUE)Running CI linting...$(NC)"
	poetry run ruff check lrs-agents/lrs neuralblitz-core/src
	poetry run black --check lrs-agents/lrs neuralblitz-core/src
	poetry run mypy lrs-agents/lrs neuralblitz-core/src

# Utility
shell: ## Open a poetry shell
	@echo "$(BLUE)Opening poetry shell...$(NC)"
	poetry shell

version: ## Show current version
	@echo "$(GREEN)NeuralBlitz Workspace Version:$(NC)"
	@poetry version
	@poetry run python -c "from importlib.metadata import version; print('Installed:', version('neuralblitz-workspace'))"

# Combined workflows
dev-setup: install-dev pre-commit ## Complete development environment setup
	@echo "$(GREEN)Development environment setup complete!$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make test' to verify installation"
	@echo "  2. Run 'make format' to format code"
	@echo "  3. Run 'make lint' to check code quality"

quickstart: install-all format test ## Quick start for new contributors
	@echo "$(GREEN)Quickstart complete! You're ready to contribute.$(NC)"
