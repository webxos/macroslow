# File Route: /chimera/setup.py
# Purpose: Python package setup for CHIMERA 2048 SDK.
# Description: Configures the CHIMERA SDK as a Python package for distribution on PyPI or private registries.
#              Supports TensorFlow, PyTorch, DSPy, SQLAlchemy, Qiskit, and MongoDB RAG for hybrid quantum workflows.
# Version: 1.0.0
# Publishing Entity: WebXOS Research Group
# Publication Date: August 29, 2025
# Copyright: © 2025 Webxos. All Rights Reserved.

from setuptools import setup, find_packages

setup(
    name="chimera-2048",
    version="1.0.0",  # Semantic versioning
    description="CHIMERA 2048 API Gateway for hybrid quantum workflows with MAML support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="WebXOS Research Group",
    author_email="research@webxos.ai",
    url="https://github.com/webxos/project-dunes",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow==2.15.0",
        "torch==2.0.1",
        "dspy==2.4.0",
        "sqlalchemy==2.0.0",
        "qiskit==0.45.0",
        "pymongo==4.6.0",
        "prometheus-client==0.17.0",
        "[YOUR_SDK_MODULE]"  # Replace with your SDK dependency
    ],
    extras_require={
        "cuda": ["nvidia-cuda-runtime-cu12==12.2.0"],
        "dev": ["pytest==7.4.0", "sphinx==5.3.0"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    license="Apache-2.0",
    include_package_data=True,
    package_data={
        "chimera": ["maml/*.md", "data/*.csv"]
    }
)

# Customization Instructions:
# 1. Replace [YOUR_SDK_MODULE] with your SDK's pip package (e.g., my_sdk==1.0.0).
# 2. Update long_description with your README content or point to /chimera/README.md.
# 3. Adjust install_requires based on your dependencies.
# 4. Ensure /src contains your Python modules (e.g., chimera_analytics_core.py).
# 5. Add /maml and /data files to package_data as needed.
# 6. Build package: `python chimera/setup.py sdist bdist_wheel`.
# 7. Publish to PyPI: `twine upload dist/* --repository-url [YOUR_PYPI_URL]`.
# 8. Include LICENSE file (Apache-2.0) in /chimera/LICENSE.
# 9. Generate docs with Sphinx: `sphinx-build docs docs/_build`.
# 10. Scale to AES-2048 or add Kubernetes support via /chimera/chimera_helm_chart.yaml.