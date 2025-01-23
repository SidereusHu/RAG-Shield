from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ragshield",
    version="0.1.0",
    author="RAG-Shield Contributors",
    description="Security framework for Retrieval-Augmented Generation systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RAG-Shield",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "mypy>=1.5.0",
            "ruff>=0.1.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
)
