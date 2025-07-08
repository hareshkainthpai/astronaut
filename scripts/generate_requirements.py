#!/usr/bin/env python3
"""
Script to generate comprehensive requirements.txt from currently installed packages
and analyze imports used in the project.
"""

import os
import sys
import subprocess
import importlib
import ast
import re
from pathlib import Path
from typing import Set, Dict, List, Tuple


def get_installed_packages() -> Dict[str, str]:
    """Get all installed packages with their versions using pip freeze"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                                capture_output=True, text=True, check=True)
        packages = {}
        for line in result.stdout.strip().split('\n'):
            if line and '==' in line:
                name, version = line.split('==', 1)
                packages[name.lower()] = version
        return packages
    except subprocess.CalledProcessError as e:
        print(f"Error getting installed packages: {e}")
        return {}


def get_project_imports(project_path: str) -> Set[str]:
    """Scan all Python files in the project to find import statements"""
    imports = set()
    project_path = Path(project_path)

    # Patterns to match import statements
    import_patterns = [
        r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',
    ]

    python_files = list(project_path.rglob('*.py'))

    for py_file in python_files:
        # Skip virtual environment and cache directories
        if any(part in str(py_file) for part in ['.venv', '__pycache__', '.git', 'site-packages']):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse with AST for more accurate parsing
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
            except SyntaxError:
                # Fallback to regex if AST parsing fails
                for line in content.split('\n'):
                    line = line.strip()
                    for pattern in import_patterns:
                        match = re.match(pattern, line)
                        if match:
                            imports.add(match.group(1))

        except (UnicodeDecodeError, PermissionError):
            continue

    return imports


def get_package_mapping() -> Dict[str, str]:
    """Map import names to package names for common packages"""
    return {
        'django': 'Django',
        'torch': 'torch',
        'transformers': 'transformers',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'requests': 'requests',
        'psutil': 'psutil',
        'pynvml': 'pynvml',
        'channels': 'channels',
        'asgiref': 'asgiref',
        'accelerate': 'accelerate',
        'vllm': 'vllm',
        'ray': 'ray',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'websockets': 'websockets',
        'aiofiles': 'aiofiles',
        'jinja2': 'Jinja2',
        'markupsafe': 'MarkupSafe',
        'sqlalchemy': 'SQLAlchemy',
        'alembic': 'alembic',
        'yaml': 'PyYAML',
        'dotenv': 'python-dotenv',
        'jose': 'python-jose',
        'passlib': 'passlib',
        'bcrypt': 'bcrypt',
        'cryptography': 'cryptography',
        'httpx': 'httpx',
        'anyio': 'anyio',
        'starlette': 'starlette',
        'pydantic': 'pydantic',
        'typing_extensions': 'typing_extensions',
        'setuptools': 'setuptools',
        'wheel': 'wheel',
        'pip': 'pip',
    }


def filter_system_packages(packages: Dict[str, str]) -> Dict[str, str]:
    """Filter out system/built-in packages that shouldn't be in requirements"""
    system_packages = {
        'pip', 'setuptools', 'wheel', 'distribute', 'pkg-resources',
        'pytz', 'six', 'certifi', 'urllib3', 'charset-normalizer',
        'idna', 'python-dateutil', 'packaging', 'pyparsing'
    }

    return {name: version for name, version in packages.items()
            if name.lower() not in system_packages}


def categorize_packages(packages: Dict[str, str], project_imports: Set[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Categorize packages into direct dependencies and potential dependencies"""
    package_mapping = get_package_mapping()
    direct_deps = {}
    potential_deps = {}

    # Convert import names to package names
    mapped_imports = set()
    for imp in project_imports:
        if imp in package_mapping:
            mapped_imports.add(package_mapping[imp].lower())
        else:
            mapped_imports.add(imp.lower())

    for pkg_name, version in packages.items():
        pkg_lower = pkg_name.lower()
        if pkg_lower in mapped_imports:
            direct_deps[pkg_name] = version
        else:
            # Check if it's a common dependency
            common_deps = {
                'django', 'torch', 'transformers', 'vllm', 'channels',
                'requests', 'psutil', 'pynvml', 'accelerate', 'numpy',
                'pillow', 'asgiref'
            }
            if pkg_lower in common_deps:
                potential_deps[pkg_name] = version

    return direct_deps, potential_deps


def generate_requirements_content(direct_deps: Dict[str, str], potential_deps: Dict[str, str]) -> str:
    """Generate the requirements.txt content with categories"""
    content = []

    # Header
    content.append("# Generated requirements.txt")
    content.append("# Generated on: " + subprocess.run(['date'], capture_output=True, text=True).stdout.strip())
    content.append("# Python version: " + sys.version.split()[0])
    content.append("")

    # Core Django and web framework dependencies
    django_deps = {k: v for k, v in direct_deps.items() if
                   'django' in k.lower() or 'asgi' in k.lower() or 'channel' in k.lower()}
    if django_deps:
        content.append("# Django and Web Framework")
        for pkg, version in sorted(django_deps.items()):
            content.append(f"{pkg}=={version}")
        content.append("")

    # ML and AI dependencies
    ml_deps = {k: v for k, v in direct_deps.items() if
               any(ml_key in k.lower() for ml_key in ['torch', 'transform', 'vllm', 'acceler', 'numpy', 'tensor'])}
    if ml_deps:
        content.append("# Machine Learning and AI")
        for pkg, version in sorted(ml_deps.items()):
            content.append(f"{pkg}=={version}")
        content.append("")

    # System and utility dependencies
    system_deps = {k: v for k, v in direct_deps.items() if
                   any(sys_key in k.lower() for sys_key in ['request', 'psutil', 'pynvml', 'nvidia'])}
    if system_deps:
        content.append("# System and Utilities")
        for pkg, version in sorted(system_deps.items()):
            content.append(f"{pkg}=={version}")
        content.append("")

    # Other direct dependencies
    other_deps = {k: v for k, v in direct_deps.items()
                  if k not in django_deps and k not in ml_deps and k not in system_deps}
    if other_deps:
        content.append("# Other Dependencies")
        for pkg, version in sorted(other_deps.items()):
            content.append(f"{pkg}=={version}")
        content.append("")

    # Potential dependencies (commented out)
    if potential_deps:
        content.append("# Potential dependencies (uncomment if needed)")
        for pkg, version in sorted(potential_deps.items()):
            content.append(f"# {pkg}=={version}")

    return '\n'.join(content)


def main():
    """Main function to generate requirements.txt"""
    project_root = Path(__file__).parent.parent  # Assuming script is in scripts/ directory

    print("🔍 Analyzing project dependencies...")
    print(f"Project root: {project_root}")

    # Get installed packages
    print("📦 Getting installed packages...")
    installed_packages = get_installed_packages()
    print(f"Found {len(installed_packages)} installed packages")

    # Get project imports
    print("🔎 Scanning project for imports...")
    project_imports = get_project_imports(str(project_root))
    print(f"Found imports: {sorted(project_imports)}")

    # Filter system packages
    filtered_packages = filter_system_packages(installed_packages)
    print(f"Filtered to {len(filtered_packages)} relevant packages")

    # Categorize packages
    direct_deps, potential_deps = categorize_packages(filtered_packages, project_imports)
    print(f"Direct dependencies: {len(direct_deps)}")
    print(f"Potential dependencies: {len(potential_deps)}")

    # Generate requirements content
    requirements_content = generate_requirements_content(direct_deps, potential_deps)

    # Write to file
    requirements_file = project_root / 'requirements.txt'
    with open(requirements_file, 'w') as f:
        f.write(requirements_content)

    print(f"✅ Generated requirements saved to: {requirements_file}")
    print("\n📋 Summary:")
    print(f"  Direct dependencies: {len(direct_deps)}")
    print(f"  Potential dependencies: {len(potential_deps)}")
    print("\n🔧 Next steps:")
    print("  1. Review the generated requirements.txt")
    print("  2. Compare with your current requirements.txt")
    print("  3. Update requirements.txt as needed")
    print("  4. Test in a fresh virtual environment")


if __name__ == "__main__":
    main()