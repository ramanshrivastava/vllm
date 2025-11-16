"""
Mini-vLLM: A Learning-Focused LLM Inference Engine

This is a simplified implementation of vLLM for educational purposes.
It implements the core innovations of vLLM in ~10,000 lines of Python.
"""

from setuptools import setup, find_packages

setup(
    name="mini-vllm",
    version="0.1.0",
    description="A learning-focused implementation of vLLM",
    author="Learning Project",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
