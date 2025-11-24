"""Setup script for etl-studio package."""

from setuptools import setup, find_packages

setup(
    name="etl-studio",
    version="0.1.0",
    description="Lightweight Streamlit platform for collaborative data workflows",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.38.0",
        "pandas>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "pytest-cov>=4.0.0",
        ],
        "test": [
            "pytest>=8.2.0",
            "pytest-cov>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
