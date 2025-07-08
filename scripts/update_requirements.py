#!/usr/bin/env python3
"""Quick script to update requirements.txt with exact versions"""

import subprocess
import sys
import os


def main():
    print("🔄 Updating requirements.txt with exact versions...")

    try:
        # Get current pip freeze
        result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                                capture_output=True, text=True, check=True)

        # Core packages for your project
        core_packages = {
            'Django', 'channels', 'pynvml', 'nvidia-ml-py', 'vllm',
            'torch', 'transformers', 'accelerate', 'requests', 'psutil', 'asgiref'
        }

        # Parse pip freeze output
        installed = {}
        for line in result.stdout.strip().split('\n'):
            if '==' in line:
                name, version = line.split('==', 1)
                installed[name] = version

        # Generate requirements
        requirements = []
        requirements.append("# Core Django and Web Framework")

        django_deps = ['Django', 'channels', 'asgiref']
        for dep in django_deps:
            if dep in installed:
                requirements.append(f"{dep}=={installed[dep]}")

        requirements.append("\n# NVIDIA and GPU Management")
        gpu_deps = ['pynvml', 'nvidia-ml-py']
        for dep in gpu_deps:
            if dep in installed:
                requirements.append(f"{dep}=={installed[dep]}")

        requirements.append("\n# Machine Learning and AI")
        ml_deps = ['vllm', 'torch', 'transformers', 'accelerate']
        for dep in ml_deps:
            if dep in installed:
                requirements.append(f"{dep}=={installed[dep]}")

        requirements.append("\n# System and Utilities")
        util_deps = ['requests', 'psutil']
        for dep in util_deps:
            if dep in installed:
                requirements.append(f"{dep}=={installed[dep]}")

        # Add any other important packages
        requirements.append("\n# Other Dependencies")
        other_important = ['numpy', 'Pillow', 'setuptools', 'wheel']
        for dep in other_important:
            if dep in installed:
                requirements.append(f"{dep}=={installed[dep]}")

        # Write to file
        with open('requirements_exact.txt', 'w') as f:
            f.write('\n'.join(requirements))

        print("✅ Created requirements_exact.txt with exact versions")
        print(f"📦 Included {len([r for r in requirements if '==' in r])} packages")

        # Also create a minimal requirements.txt
        minimal_reqs = []
        for line in requirements:
            if '==' in line:
                name, version = line.split('==')
                # Use compatible version for core packages
                if name in ['Django', 'torch', 'transformers']:
                    major, minor = version.split('.')[:2]
                    minimal_reqs.append(f"{name}>={major}.{minor},<{int(major) + 1}.0")
                else:
                    minimal_reqs.append(f"{name}=={version}")
            else:
                minimal_reqs.append(line)

        with open('requirements_minimal.txt', 'w') as f:
            f.write('\n'.join(minimal_reqs))

        print("✅ Created requirements_minimal.txt with version ranges")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()