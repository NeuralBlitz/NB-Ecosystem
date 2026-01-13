"""
Setup script for the NeuralBlitz package.

This script makes the modules within our repository (like core_engine,
subsystems, etc.) installable and discoverable by Python's import system.
This is essential for running tests and for packaging the project.
"""

from setuptools import setup, find_packages

setup(
    name="neuralblitz",
    version="0.1.0",
    description="The Symbiotic Intelligence Framework",
    author="[Your Name]", # Or NuralNexus, etc.
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
