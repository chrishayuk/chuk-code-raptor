# Makefile for CodeRaptor test project

.PHONY: install test build clean lint format help

# Default target
help:
	@echo "Available targets:"
	@echo "  install  - Install dependencies"
	@echo "  test     - Run tests"
	@echo "  build    - Build project"
	@echo "  clean    - Clean build artifacts"
	@echo "  lint     - Run linters"
	@echo "  format   - Format code"
	@echo "  raptor   - Run CodeRaptor indexing"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	python3 -m pip install -r requirements.txt
	@echo "Installing Node.js dependencies..."
	npm install
	@echo "✅ Dependencies installed"

# Run tests
test:
	@echo "Running Python tests..."
	python3 -m pytest tests/ -v
	@echo "Running JavaScript tests..."
	node tests/test_utils.js
	@echo "✅ Tests completed"

# Build project
build:
	@echo "Building project..."
	bash scripts/build.sh
	@echo "✅ Build completed"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf dist/
	rm -rf build/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	@echo "✅ Cleaned"

# Run linters
lint:
	@echo "Running Python linters..."
	flake8 src/python/ tests/
	mypy src/python/
	@echo "Running JavaScript linters..."
	npx eslint src/frontend/
	@echo "✅ Linting completed"

# Format code
format:
	@echo "Formatting Python code..."
	black src/python/ tests/
	@echo "Formatting JavaScript code..."
	npx prettier --write src/frontend/
	@echo "✅ Code formatted"

# Run CodeRaptor indexing
raptor:
	@echo "Running CodeRaptor indexing..."
	code-raptor index . --config file_processor_config.yaml
	@echo "✅ Indexing completed"

# Development server
dev:
	@echo "Starting development servers..."
	python3 src/python/main.py --config config/development.json &
	npm start &
	@echo "✅ Development servers started"
