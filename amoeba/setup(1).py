# AMOEBA 2048AES SDK Setup Script
# Description: Python setup script for packaging and distributing the AMOEBA 2048AES SDK as a PyPI-compatible package.

from setuptools import setup, find_packages

setup(
    name="amoeba2048aes-sdk",
    version="1.0.0",
    description="AMOEBA 2048AES SDK for quantum-enhanced, distributed workflows with Dropbox integration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Webxos",
    author_email="contact@webxos.org",
    url="https://github.com/webxos/amoeba2048aes-sdk",
    packages=find_packages(),
    install_requires=[
        "qiskit==1.0.0",
        "torch==2.0.1",
        "pydantic",
        "dropbox==11.36.2",
        "cryptography",
        "click",
        "pytest",
        "prometheus-client",
        "fastapi",
        "uvicorn",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "amoeba2048=amoeba2048_cli:cli",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)