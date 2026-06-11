#!/usr/bin/env python3
"""
H3C-BEACON: Hierarchical Hybrid Heterogeneous Control with
Bayesian-Elites Adaptive COalition Network

Submitted to Complex & Intelligent Systems (Springer, 2026).

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="h3c-beacon",
    version="2.0.0",
    author="Basile BETE MBEZELE, Ghislain ALO'O ABESSOLO",
    author_email="mbezelebetebasile@gmail.com",
    description=(
        "H3C-BEACON: Hierarchical Hybrid Heterogeneous Control with "
        "Bayesian-Elites Adaptive Coalition Network for MARL"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbezelebasile/H3C-BEACON",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "smac": [
            "smaclite",
        ],
        "hanabi": [
            # hanabi-learning-environment (optional, falls back to built-in)
        ],
    },
    entry_points={
        "console_scripts": [
            "h3c-train=train_mpe_fixed:main",
            "h3c-smac=train_smac_h3c_fixed:main",
            "h3c-hanabi=train_hanabi_full:main",
        ],
    },
)