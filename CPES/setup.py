"""
Setup script for CPES package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        # Fallback to hardcoded requirements if requirements.txt is not found
        return [
            "numpy>=1.21.0",
            "networkx>=2.6.0", 
            "faiss-cpu>=1.7.0",
            "sentence-transformers>=2.2.0",
            "loguru>=0.6.0",
            "pydantic>=1.8.0",
            "pyyaml>=6.0",
            "openai>=1.0.0",
            "anthropic>=0.3.0",
            "litellm>=1.0.0",
        ]

setup(
    name="cpes",
    version="0.1.0",
    author="Aron LeiLion",
    author_email="a.leilionas@vk.com",
    description="Composable Persona Emulation Stack - A practical blueprint for building stable behavioral emulation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/IlumCI/cpes",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pytest-cov>=4.0.0",
        ],
        "swarms": [
            "swarms>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cpes=cpes.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cpes": ["examples/*.py", "examples/*.yaml"],
    },
)
