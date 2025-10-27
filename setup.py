#!/usr/bin/env python3
"""Setup configuration for Incident Intelligence System"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Core dependencies
install_requires = [
    "pandas>=1.3.0",
    "numpy>=1.20.0",
]

# Optional dependencies
extras_require = {
    "all": [
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "duckdb>=0.8.0",
        "PyYAML>=5.4.0",
        "tqdm>=4.62.0",
    ],
    "ml": [
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    "performance": [
        "duckdb>=0.8.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
    ],
}

setup(
    name="incident-intelligence-system",
    version="1.0.0",
    author="Lilun Cheng",
    author_email="your.email@example.com",
    description="Dual-engine anomaly detection and predictive analytics for IT operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/incident-intelligence-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Topic :: System :: Monitoring",
        "Topic :: Software Development :: Quality Assurance",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "incident-ranker=anomaly_ranker_engine:main",
            "incident-predictor=predictive_pattern_miner:main",
        ],
    },
)
