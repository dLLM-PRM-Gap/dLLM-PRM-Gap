"""Setup for dllm-prm-gap package."""
from setuptools import setup, find_packages

setup(
    name="dllm-prm-gap",
    version="0.1.0",
    description="Process Reward Models for discrete diffusion language models",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        line.strip() for line in open("requirements.txt").read().splitlines()
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
