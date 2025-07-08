.PHONY: requirements requirements-dev requirements-exact freeze-requirements update-requirements

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