from setuptools import setup, find_packages

setup(
    name="factcheck-pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "datasets>=2.14",
        "transformers>=4.40",
        "torch>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.10",
        "numpy>=1.24",
        "accelerate>=0.30",
    ],
)
