"""Package setup configuration for DataStudio."""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith('#')
    ]

setup(
    name="DataStudio",
    version="0.1.0",
    description="Config-driven multimodal data processing pipeline for MLLMs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Yi Zhang",
    author_email="yi.zhang.4096@gmail.com",
    url="https://github.com/Open-Bee/DataStudio",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "wandb": ["wandb>=0.16.0"],
        "test": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "all": ["wandb>=0.16.0", "pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
