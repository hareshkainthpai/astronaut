
# Default configuration - can be overridden
PORT ?= 8000
REDIS_PORT ?= 6379

.PHONY: requirements requirements-dev requirements-exact freeze-requirements update-requirements docker-build docker-up docker-rebuild docker-down docker-logs setup dev-setup kill-all clean-start

# === Python Requirements Management ===
# Generate requirements from current environment
requirements:
	@echo "🔍 Generating requirements.txt from current environment..."
	python -m pip freeze > requirements_frozen.txt
	@echo "✅ Generated requirements_frozen.txt"

# Generate requirements with development dependencies
requirements-dev:
	@echo "🔍 Generating dev requirements..."
	python -m pip freeze | grep -E "(pytest|black|flake8|mypy|pre-commit)" > requirements_dev.txt
	@echo "✅ Generated requirements_dev.txt"

# Generate exact requirements using our script
requirements-exact:
	@echo "🔄 Generating exact requirements..."
	python update_requirements.py
	@echo "✅ Generated requirements_exact.txt and requirements_minimal.txt"

# Show what packages are currently installed
freeze-requirements:
	@echo "📦 Currently installed packages:"
	python -m pip freeze

# Update requirements and test installation
update-requirements: requirements-exact
	@echo "🧪 Testing requirements in a temporary environment..."
	@echo "Creating temporary virtual environment..."
	python -m venv temp_test_env
	./temp_test_env/bin/pip install -r requirements_exact.txt
	@echo "✅ Requirements test passed"
	rm -rf temp_test_env

# Show outdated packages
show-outdated:
	@echo "📊 Checking for outdated packages..."
	python -m pip list --outdated

# Install from requirements
install:
	python -m pip install -r requirements.txt

# Install exact versions
install-exact:
	python -m pip install -r requirements_exact.txt

# === Cleanup Commands ===
# Kill all processes using specified ports
kill-ports:
	@echo "🔫 Killing processes on ports $(PORT) and $(REDIS_PORT)..."
	-sudo fuser -k $(PORT)/tcp 2>/dev/null || echo "No process on port $(PORT)"
	-sudo fuser -k $(REDIS_PORT)/tcp 2>/dev/null || echo "No process on port $(REDIS_PORT)"
	-pkill -f "python.*manage.py.*runserver" 2>/dev/null || echo "No Django runserver processes"
	-pkill -f "redis-server" 2>/dev/null || echo "No Redis processes"
	@echo "✅ Ports cleared"

# Kill all Docker containers and clean up
kill-docker:
	@echo "🐳 Stopping and removing all Docker containers..."
	-docker stop $$(docker ps -aq) 2>/dev/null || echo "No containers to stop"
	-docker rm $$(docker ps -aq) 2>/dev/null || echo "No containers to remove"
	@echo "✅ Docker containers cleaned"

# Complete cleanup - kill everything
kill-all: kill-ports kill-docker
	@echo "🧹 Complete cleanup finished"

# Create static directories
create-dirs:
	@echo "📁 Creating necessary directories..."
	mkdir -p static staticfiles media logs
	@echo "✅ Directories created"

# === Docker Commands ===
# Build and start containers with rebuild
docker-build:
	@echo "🐳 Building and starting Docker containers..."
	PORT=$(PORT) REDIS_PORT=$(REDIS_PORT) docker-compose up --build

# Start containers (after first build)
docker-up:
	@echo "🐳 Starting Docker containers..."
	PORT=$(PORT) REDIS_PORT=$(REDIS_PORT) docker-compose up

# Rebuild containers when you make changes
docker-rebuild:
	@echo "🐳 Rebuilding Docker containers..."
	PORT=$(PORT) REDIS_PORT=$(REDIS_PORT) docker-compose up --build

# Stop and remove containers
docker-down:
	@echo "🐳 Stopping Docker containers..."
	docker-compose down

# View container logs
docker-logs:
	@echo "📋 Showing Docker container logs..."
	docker-compose logs -f

# Clean up Docker resources
docker-clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down --volumes --remove-orphans
	docker system prune -f

# === Setup Commands ===
# Run setup script (development mode - skip collectstatic)
setup:
	@echo "⚙️ Running setup script (development mode) on port $(PORT)..."
	chmod +x start.sh
	PORT=$(PORT) REDIS_PORT=$(REDIS_PORT) ./start.sh

# Run setup script with collectstatic
setup-prod:
	@echo "⚙️ Running setup script (production mode) on port $(PORT)..."
	chmod +x start.sh
	PORT=$(PORT) REDIS_PORT=$(REDIS_PORT) COLLECT_STATIC=true ./start.sh

# Development setup without collectstatic
dev-setup: create-dirs install
	@echo "🚀 Setting up development environment..."
	python manage.py makemigrations
	python manage.py migrate
	@echo "✅ Development environment setup complete!"

# === Combined Commands ===
# Clean start - kill everything and rebuild
clean-start: kill-all create-dirs docker-build

# Quick start for development
dev-start: create-dirs dev-setup
	@echo "🚀 Starting development server on port $(PORT)..."
	python manage.py runserver 0.0.0.0:$(PORT)

# Docker development start with cleanup
docker-dev-start: kill-all docker-build

# Full restart (clean + rebuild)
restart: docker-down docker-build

# Force restart everything
force-restart: kill-all clean-start

# === Status Commands ===
# Check what's running on ports
check-ports:
	@echo "🔍 Checking what's running on ports $(PORT) and $(REDIS_PORT)..."
	-lsof -i :$(PORT) || echo "Nothing on port $(PORT)"
	-lsof -i :$(REDIS_PORT) || echo "Nothing on port $(REDIS_PORT)"
	-docker ps || echo "No Docker containers"

# Show current configuration
show-config:
	@echo "📋 Current Configuration:"
	@echo "  Django Port: $(PORT)"
	@echo "  Redis Port:  $(REDIS_PORT)"

# Help target to show available commands
help:
	@echo "Available targets:"
	@echo ""
	@echo "📋 Current Configuration:"
	@echo "  Django Port: $(PORT) (override with PORT=xxxx)"
	@echo "  Redis Port:  $(REDIS_PORT) (override with REDIS_PORT=xxxx)"
	@echo ""
	@echo "  === Python Requirements ==="
	@echo "  requirements       - Generate requirements.txt from current environment"
	@echo "  requirements-dev   - Generate dev requirements"
	@echo "  requirements-exact - Generate exact requirements"
	@echo "  install           - Install from requirements.txt"
	@echo "  install-exact     - Install exact versions"
	@echo "  show-outdated     - Show outdated packages"
	@echo ""
	@echo "  === Cleanup Commands ==="
	@echo "  kill-ports        - Kill processes on configured ports"
	@echo "  kill-docker       - Stop and remove all Docker containers"
	@echo "  kill-all          - Complete cleanup (ports + docker)"
	@echo "  create-dirs       - Create necessary directories"
	@echo ""
	@echo "  === Docker Commands ==="
	@echo "  docker-build      - Build and start containers (first time)"
	@echo "  docker-up         - Start existing containers"
	@echo "  docker-rebuild    - Rebuild and start containers (after changes)"
	@echo "  docker-down       - Stop containers"
	@echo "  docker-logs       - View container logs"
	@echo "  docker-clean      - Clean up Docker resources"
	@echo ""
	@echo "  === Setup Commands ==="
	@echo "  setup            - Run setup script (development mode)"
	@echo "  setup-prod       - Run setup script (production mode)"
	@echo "  dev-setup        - Development setup without collectstatic"
	@echo ""
	@echo "  === Combined Commands ==="
	@echo "  clean-start      - Kill everything and rebuild (RECOMMENDED)"
	@echo "  dev-start        - Quick start for development"
	@echo "  docker-dev-start - Docker development start with cleanup"
	@echo "  restart          - Full restart (clean + rebuild)"
	@echo "  force-restart    - Force restart everything"
	@echo ""
	@echo "  === Status Commands ==="
	@echo "  check-ports      - Check what's running on configured ports"
	@echo "  show-config      - Show current port configuration"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "📝 Examples:"
	@echo "  make dev-start                    # Use default port 8000"
	@echo "  make dev-start PORT=8080          # Use port 8080"
	@echo "  make docker-build PORT=9000      # Docker with port 9000"
	@echo "  make kill-ports PORT=8080        # Kill processes on port 8080"