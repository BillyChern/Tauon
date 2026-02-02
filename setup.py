"""Setup script for soft-muon package."""

from setuptools import setup, find_packages

setup(
    name="soft-muon",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
)
