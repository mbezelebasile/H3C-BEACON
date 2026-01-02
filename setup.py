#!/usr/bin/env python3
"""
H3C-BEACON: Hierarchical Hybrid Heterogeneous Control with 
Bayesian-Elite Adaptive COalition Network

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="h3c-beacon",
    version="1.0",
    authors=" Basile BETE MBEZELE, and GHISLAIN ALO'O ABESSOLO",
    author_email="mbezelebetebasile@gmail.com",
    description="Hierarchical Hybrid Heterogeneous Control with Bayesian-Elite Adaptive COalition Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbezelebasile/H3C-BEACON",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "h3c-train=train:main",
            "h3c-ablation=run_ablation:main",
        ],
    },
)
